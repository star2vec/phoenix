"""One forward on explicit token ids; answer split; thought capture and
injection helpers shared by the drivers.

T is on the paper's 0-100 scale here (harness.Runner.measure returns 0-1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from harness import VENDOR  # noqa: F401  (inserts VENDOR on sys.path)
from prompts import N_NODE_TOKENS
from thoughts import inject  # noqa: F401  (re-exported for drivers)

PAD_TOKEN = "<eos>"


@torch.no_grad()
def run_ids(runner, ids, thought_edit=None, attn_eager=False):
    """Returns the logits at the last position (the answer position)."""
    input_ids = torch.tensor([ids], device=runner.device)
    n = len(ids)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "position_ids": torch.arange(n, device=runner.device).reshape(1, -1),
    }
    out = runner.model(**batch, thought_edit=thought_edit, attn_eager=attn_eager)
    return out.logits[0, -1].float()


def answer_split(logits, target, decoy, watch=None):
    """T, e, and where the answer probability went. watch: optional list of
    node tokens whose summed probability is reported as p_watch."""
    p = F.softmax(logits, dim=-1)
    pt, pd = float(p[target]), float(p[decoy])
    T = 100.0 * pt / (pt + pd) if pt + pd > 0 else 50.0
    node_mass = float(p[:N_NODE_TOKENS].sum())
    out = {
        "T": T,
        "e": 1.0 - (pt + pd),
        "p_target": pt,
        "p_decoy": pd,
        "p_other_node": node_mass - pt - pd,
        "p_other_token": 1.0 - node_mass,
        "argmax": int(p.argmax()),
    }
    if watch is not None:
        out["p_watch"] = float(sum(float(p[v]) for v in watch))
        out["watch"] = list(watch)
    return out


def measure(runner, prompt, thought_edit=None, attn_eager=False, watch=None, **text_kw):
    """Render, run, split. text_kw is passed to Prompt.text (e.g. K=0)."""
    ids = prompt.ids(runner.tok, **text_kw)
    logits = run_ids(runner, ids, thought_edit, attn_eager)
    return answer_split(logits, prompt.target, prompt.decoy, watch)


@torch.no_grad()
def capture(runner, ids, attn_eager=False):
    """{pass_idx: recycled thought} from one forward on ids."""
    cap = {}

    def grab(k, v):
        cap[k] = v.detach().clone()
        return v

    run_ids(runner, ids, grab, attn_eager)
    return cap


def intermediates(K):
    """Passes 0..K-2: every recycled thought except the final one."""
    return list(range(K - 1))


def all_passes(K):
    return list(range(K))


def fixed(thoughts, passes):
    """thought_edit that injects `thoughts` at the given passes."""
    return inject(thoughts, set(passes))


def every_pass(fn):
    """thought_edit applying fn(pass_idx, vec) -> vec at every pass."""
    return fn


def at_passes(fn, passes):
    passes = set(passes)

    def f(k, t):
        return fn(k, t) if k in passes else t
    return f
