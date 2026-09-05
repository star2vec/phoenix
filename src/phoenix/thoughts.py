"""Thought capture, transplant, and pinned-serialization measurement.

Vendor expand_data shuffles the edge list and flips candidate order through the
GLOBAL random module, so without pinning every forward sees a different prompt
and a measured change confounds the edit with serialization noise.
pin_serialization seeds the global RNG identically before every condition of a
graph, so all conditions share one prompt; reserial=True gives a second, fixed,
different serialization of the same graph (the noise floor for an unedited
forward). expand_data also shuffles sample["edges"] in place, so callers pass
the original edge list (edges0) and it is restored before every forward.

measure_multi passes thought_edit(pass_idx, vec) straight through to
FastCoconut, so an edit can fire at EVERY latent pass (pass_idx is 0-based).
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from harness import VENDOR  # noqa: F401  (inserts VENDOR on sys.path)
from dataset import expand_data


@torch.no_grad()
def measure_multi(runner, sample, thought_edit):
    """Same forward as Runner.measure, but thought_edit(pass_idx, vec) is passed
    through to FastCoconut directly so an edit can fire at EVERY latent pass."""
    max_steps = len(sample["steps"])
    question, _ = expand_data(sample, max_steps + 1, max_steps, neg_sampling=False)
    ids = runner.tok.encode(question)
    input_ids = torch.tensor([ids], device=runner.device)
    n = len(ids)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "position_ids": torch.arange(n, device=runner.device).reshape(1, -1),
    }
    out = runner.model(**batch, thought_edit=thought_edit)
    p = F.softmax(out.logits[0, -1], dim=-1)
    pt, pd = p[sample["target"]].item(), p[sample["neg_target"]].item()
    T = pt / (pt + pd) if (pt + pd) > 0 else 0.5
    escape = 1.0 - (pt + pd)
    return T, escape


def pin_serialization(gi, base_seed, reserial=False):
    """Seed the global RNG that vendor expand_data draws from. gi is the graph's
    index in its source file; reserial selects a second fixed serialization."""
    random.seed(((gi << 16) ^ base_seed) + (0xABCDEF if reserial else 0))


def pinned_measure(runner, sample, edges0, gi, base_seed, edit, reserial=False):
    """Restore the original edge order, pin the serialization, measure."""
    sample["edges"] = [list(e) for e in edges0]
    pin_serialization(gi, base_seed, reserial)
    return measure_multi(runner, sample, edit)


def capture_thoughts(runner, sample, gi, base_seed, reserial=False):
    """One pinned-serialization forward; returns {pass_idx: thought tensor}."""
    edges0 = [list(e) for e in sample["edges"]]
    sample["edges"] = [list(e) for e in edges0]
    pin_serialization(gi, base_seed, reserial)
    cap = {}

    def grab(k, v):
        cap[k] = v.detach().clone()
        return v

    measure_multi(runner, sample, grab)
    sample["edges"] = [list(e) for e in edges0]
    return cap


def inject(donor_thoughts, passes):
    """thought_edit replacing the recycled thought at the given passes."""
    def f(k, t):
        return donor_thoughts[k] if k in passes else t
    return f


def mix_edit(donor_thoughts, passes, alpha):
    """thought_edit blending the LIVE thought with the donor's at the given
    passes: (1-a)*t + a*donor_k, rescaled to ||t||. a=1 reduces to inject,
    a=0 to no edit."""
    def f(k, t):
        if k not in passes:
            return t
        t2 = (1 - alpha) * t + alpha * donor_thoughts[k]
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return f


def find_donor(pool, target, neg_target, k):
    """First pool graph with the given candidate pair and solution length."""
    for d in pool:
        if (
            d["target"] == target
            and d["neg_target"] == neg_target
            and len(d["steps"]) == k
        ):
            return d
    return None


def find_random_donor(pool, avoid, k):
    """First pool graph of solution length k whose candidates avoid the set."""
    for d in pool:
        if len(d["steps"]) == k and d["target"] not in avoid and d["neg_target"] not in avoid:
            return d
    return None
