"""Loader for the fine-tuned GPT-2 COCONUT checkpoints (experiment 7).

The checkpoints are raw `torch.save` state dicts of the official Coconut
wrapper (facebookresearch/coconut, run.py at commit 27273cb8): GPT-2 small
with three added tokens <|start-latent|>, <|end-latent|>, <|latent|> (ids
50257-50259, in that order), vocabulary resized to 50260, three tied copies
of the embedding matrix (transformer.wte, lm_head, the wrapper's `embedding`).

    ckpts/<run>/best.pt        the checkpoint (copied from Hugging Face)
    ckpts/<run>/SOURCE.json    repo, file, bytes, sha256, url
    ckpts/gpt2_tokenizer/      the GPT-2 tokenizer with the three tokens added

Runs: gpt2_dilgren (connordilgren/gpt2-prosqa-coconut, checkpoint_40) and
gpt2_aswal (darpanaswal/coconut-gpt2-prosqa, checkpoint_best).

The from-scratch loader (harness.Runner) is untouched; drivers that accept
either model take a runner with .model (FastCoconut), .tok, .device, .wte,
.gen, .rng and .u_hat, which both provide.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config  # noqa: E402

from fast_coconut import FastCoconut  # noqa: E402
from nl import bind_tokenizer, special_ids  # noqa: E402
from sets import ROOT, require_checkpoint  # noqa: E402

TOKENIZER_DIR = ROOT / "ckpts" / "gpt2_tokenizer"
GPT2_RUNS = ("gpt2_dilgren", "gpt2_aswal")
_PREFIXES = ("module.", "_fsdp_wrapped_module.")


def load_tokenizer(tokenizer_dir=TOKENIZER_DIR):
    """The snapshot with the three tokens added in the official order; if it
    is missing, build it from the hub and save it."""
    tokenizer_dir = Path(tokenizer_dir)
    if tokenizer_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        for t in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
            tok.add_tokens(t)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(tokenizer_dir))
    sp = special_ids(tok)
    assert (sp["start"], sp["end"], sp["latent"]) == (50257, 50258, 50259), sp
    assert len(tok) == 50260 and tok.eos_token_id == 50256
    return tok


def _strip(key):
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if p in key:
                key = key.replace(p, "")
                changed = True
    return key


def load_coconut_state(model, checkpoint):
    """Strict load of a Coconut-wrapper state dict with the tied-weight
    checks. Returns a small report."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {_strip(k): v for k, v in raw.items()}
    expected = set(model.state_dict().keys())
    missing, unexpected = expected - set(state), set(state) - expected
    assert not missing and not unexpected, f"missing {sorted(missing)[:5]} unexpected {sorted(unexpected)[:5]}"
    ties = ("base_causallm.transformer.wte.weight", "base_causallm.lm_head.weight", "embedding.weight")
    for k in ties[1:]:
        assert torch.equal(state[ties[0]], state[k]), f"tied copies differ: {k}"
    model.load_state_dict(state, strict=True)
    ptrs = {model.base_causallm.transformer.wte.weight.data_ptr(),
            model.base_causallm.lm_head.weight.data_ptr(), model.embedding.weight.data_ptr()}
    assert len(ptrs) == 1, "embedding, lm_head and wte are not tied after loading"
    wte = model.base_causallm.transformer.wte.weight
    special_rows = wte[50257:50260]
    mean_row = wte[:50257].mean(0)
    special_norm_gap = float((special_rows - mean_row).norm(dim=1).min())
    assert special_norm_gap > 1e-3, "the three special rows equal the mean embedding (untrained?)"
    return {"n_keys": len(state), "stripped_prefixes": sorted({k for k in raw if _strip(k) != k})[:3],
            "special_rows_min_gap_to_mean": special_norm_gap}


class GPT2Runner:
    def __init__(self, checkpoint, device="cpu", seed=0, tokenizer_dir=TOKENIZER_DIR, tiny=None):
        """checkpoint=None keeps the random initialisation (tests). tiny: a
        dict of GPT2Config overrides for a small random model in tests."""
        self.device = torch.device(device)
        self.tok = load_tokenizer(tokenizer_dir)
        bind_tokenizer(self.tok)
        sp = special_ids(self.tok)
        self.latent_id = sp["latent"]
        cfg = GPT2Config(**(tiny or {}))
        base = AutoModelForCausalLM.from_config(cfg)
        base.resize_token_embeddings(len(self.tok))  # before wrapping: the wrapper keeps the module
        self.model = FastCoconut(base, sp["latent"], sp["start"], sp["end"], self.tok.eos_token_id)
        self.load_report = None
        if checkpoint is not None:
            self.load_report = load_coconut_state(self.model, checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.wte = base.transformer.wte.weight
        self.rng = random.Random(seed)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def u_hat(self, token_id):
        v = self.wte[token_id].detach()
        return v / v.norm()

    def rand_unit(self):
        v = torch.randn(self.wte.shape[1], generator=self.gen)
        return (v / v.norm()).to(self.device)


def source_info(run_name):
    p = ROOT / "ckpts" / run_name / "SOURCE.json"
    return json.load(open(p)) if p.exists() else {}


def load_gpt2_runner(args):
    ckpt = require_checkpoint(args.run_name)
    return GPT2Runner(ckpt, device=args.device, seed=args.seed)


@torch.no_grad()
def greedy_generate(runner, ids, max_new_tokens=128):
    """Greedy decoding as the official run.py does it (batch one, argmax,
    stop at eos), but with the key/value cache: one base forward over the
    latent-filled input embeddings, then one token at a time."""
    dev = runner.device
    input_ids = torch.tensor([ids], device=dev)
    n = len(ids)
    out = runner.model(input_ids, torch.ones_like(input_ids), input_ids.clone(),
                       torch.arange(n, device=dev).reshape(1, -1))
    base = runner.model.base_causallm
    o = base(inputs_embeds=out.inputs_embeds, use_cache=True)
    past = o.past_key_values
    nxt = int(o.logits[0, -1].argmax())
    tokens = [nxt]
    eos = runner.tok.eos_token_id
    for _ in range(max_new_tokens - 1):
        if nxt == eos:
            break
        o = base(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        tokens.append(nxt)
    if tokens and tokens[-1] == eos:
        tokens = tokens[:-1]
    return tokens
