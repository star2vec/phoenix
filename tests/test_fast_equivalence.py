"""Assert FastCoconut/FastSTokenizer are numerically identical to vendor code.

Run once before any training relies on the fast path:
    .venv/bin/python tests/test_fast_equivalence.py
CPU, eval mode (dropout off) for determinism; checks logits, loss, gradients,
and tokenization on real data at the deepest curriculum stage.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "reasoning-by-superposition"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from coconut import Coconut
from dataset import MyCollator, get_graph_latent_cot_dataset
from stokenizer import STokenizer
from utils import Config, set_seed

from phoenix.fast_coconut import FastCoconut, FastSTokenizer


def main():
    set_seed(0)
    slow_tok = STokenizer()
    slow_tok.pad_token = slow_tok.eos_token = slow_tok.bos_token = "<eos>"
    fast_tok = FastSTokenizer()
    latent_id = slow_tok.convert_tokens_to_ids("<|latent|>")

    # --- tokenizer equivalence on 200 real questions (deepest stage) ------
    cfg = Config(
        {
            "debug": False,
            "uniform_prob": 0.0,
            "train_path": str(VENDOR / "data/prosqa_train_graph_4_coconut.json"),
        }
    )
    set_seed(0)
    ds_slow = get_graph_latent_cot_dataset(cfg.train_path, 4, cfg, slow_tok)
    set_seed(0)
    ds_fast = get_graph_latent_cot_dataset(cfg.train_path, 4, cfg, fast_tok)
    for a, b in zip(ds_slow[:200], ds_fast[:200]):
        assert a["input_ids"] == b["input_ids"], "tokenizer mismatch"
        assert a["labels"] == b["labels"], "labels mismatch"
    print("tokenizer: OK (200 samples identical)")

    # --- model equivalence -------------------------------------------------
    config = AutoConfig.from_pretrained(
        str(VENDOR / "configs/symbol-2layer-8head-768dim.json")
    )
    set_seed(0)
    base = AutoModelForCausalLM.from_config(config)
    ids = [
        latent_id,
        slow_tok.convert_tokens_to_ids("<|start-latent|>"),
        slow_tok.convert_tokens_to_ids("<|end-latent|>"),
        slow_tok.eos_token_id,
    ]
    slow = Coconut(base, *ids)

    set_seed(0)
    base2 = AutoModelForCausalLM.from_config(config)
    fast = FastCoconut(base2, *ids)
    fast.load_state_dict(slow.state_dict())  # belt and suspenders

    collator = MyCollator(slow_tok, latent_id=latent_id, label_pad_token_id=-100)
    batch = collator([{k: v for k, v in s.items()} for s in ds_slow[:8]])

    slow.eval()
    fast.eval()

    out_s = slow(**batch)
    out_f = fast(**batch)
    assert torch.allclose(out_s.logits, out_f.logits, atol=1e-6), (
        f"logits diverge: max abs diff {(out_s.logits - out_f.logits).abs().max()}"
    )
    assert torch.allclose(out_s.loss, out_f.loss, atol=1e-7), "loss diverges"
    assert torch.allclose(
        out_s.inputs_embeds, out_f.inputs_embeds, atol=1e-6
    ), "recycled thought embeddings diverge"
    print(f"forward: OK (loss {out_s.loss.item():.6f} identical; "
          f"max logit diff {(out_s.logits - out_f.logits).abs().max():.2e})")

    # --- hooks importable and inactive: still bit-exact ----------------------
    from phoenix.attn_hooks import AttnHooks, MLPHooks, ResidualHooks

    with AttnHooks(fast.base_causallm), ResidualHooks(fast.base_causallm), MLPHooks(fast.base_causallm):
        out_h = fast(**batch)
    assert torch.equal(out_h.logits, out_f.logits), "inactive hooks broke bit-exactness"
    assert torch.equal(out_h.inputs_embeds, out_f.inputs_embeds)
    print("hooks off: OK (installed-but-inactive hooks are bit-exact)")

    # --- eager attention path (used by the hooks): rounding-level only -------
    # (pad positions carry no labels and the two paths treat fully masked
    # query rows differently, so compare attended positions and the loss)
    out_e = fast(**batch, attn_eager=True)
    attended = batch["attention_mask"].bool()
    diff_e = (out_e.logits - out_f.logits).abs().amax(-1)[attended].max().item()
    assert diff_e < 1e-4, f"eager path diverges at attended positions: {diff_e}"
    assert torch.equal(out_e.loss, out_f.loss) or (out_e.loss - out_f.loss).abs() < 1e-6
    print(f"eager path: OK (max logit diff vs SDPA at attended positions {diff_e:.2e}; loss equal)")

    out_s.loss.backward()
    out_f.loss.backward()
    for (n1, p1), (n2, p2) in zip(slow.named_parameters(), fast.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6), (
            f"grad diverges at {n1}: max {(p1.grad - p2.grad).abs().max()}"
        )
    print("backward: OK (all parameter gradients identical)")
    print("EQUIVALENCE: PASS")


if __name__ == "__main__":
    main()
