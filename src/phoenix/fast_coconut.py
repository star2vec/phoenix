"""Drop-in performance ports of vendor Coconut.forward and STokenizer.encode.

Semantics are IDENTICAL to vendor/reasoning-by-superposition (same math, same
autograd graph); only the implementation changes:
- Coconut.forward built each pass's inputs_embeds by decomposing the whole
  (bs, len, 768) tensor into Python lists of 1-d tensors and re-stacking
  (~100k tensor ops per step) — replaced by one clone + one advanced-indexing
  assignment per latent pass.
- PreTrainedTokenizer.encode runs the added-token trie machinery per call;
  our texts are whitespace-separated vocab tokens, so a dict lookup suffices.

Equivalence is asserted by tests/test_fast_equivalence.py; run it after any
change to this file and before any training run relies on it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "reasoning-by-superposition"))

import torch
from coconut import Coconut, Outputs
from stokenizer import STokenizer
from torch.nn import CrossEntropyLoss


class FastSTokenizer(STokenizer):
    def __init__(self):
        super().__init__()
        # STokenizer assigns these before super().__init__(), which resets
        # them to None on transformers 4.46.2 — reassign post-init.
        self.pad_token = "<eos>"
        self.eos_token = "<eos>"
        self.bos_token = "<eos>"

    def encode(self, text, add_special_tokens=False, **kwargs):
        assert not add_special_tokens
        return [self.vocab[t] for t in text.replace("\n", " ").strip().split()]


class FastCoconut(Coconut):
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        thought_edit=None,
        attn_eager=False,
        **kwargs,
    ):
        """thought_edit: optional callable (pass_idx, vec) -> vec applied to
        each recycled thought vector at fill time (pass_idx is 0-based, so
        thought step t corresponds to pass_idx == t-1). None (default)
        reproduces vendor behavior bit-exactly. The edited vector propagates
        to all subsequent passes — this is the point where every thought
        intervention in this project is applied.

        attn_eager: when True, every base-model call requests attention
        weights, which makes the SDPA attention class take the eager path
        (the one attn_hooks.AttnHooks wraps). False (default) adds nothing to
        the call and is the bit-exact path."""
        logits = []
        base_kw = {"output_attentions": True} if attn_eager else {}

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        # one host transfer instead of per-element .item() (each forces a
        # device sync — the vendor loop costs ~20s/step on MPS at batch 128)
        latent_lists = [[] for _ in range(input_ids.shape[0])]
        for b, c in latent_indices.detach().cpu().tolist():  # row-major: cols ascend per row
            latent_lists[b].append(c)
        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    **base_kw,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    **base_kw,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # vectorized equivalent of the vendor's tensor_list rebuild:
            # replace embedding at each instance's pass_idx-th latent position
            # with the hidden state at the position immediately before it
            filling = [
                (b, l[pass_idx]) for b, l in enumerate(latent_lists) if len(l) > pass_idx
            ]
            if filling:
                rows = torch.tensor([b for b, _ in filling], device=inputs_embeds.device)
                cols = torch.tensor([c for _, c in filling], device=inputs_embeds.device)
                inputs_embeds = inputs_embeds.clone()
                new_vecs = hidden_states[rows, cols - 1 - hidden_states_offset]
                if thought_edit is not None:
                    new_vecs = torch.stack(
                        [thought_edit(pass_idx, v) for v in new_vecs]
                    )
                inputs_embeds[rows, cols] = new_vecs

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
            **base_kw,
        )
        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
