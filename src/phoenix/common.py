"""Shared driver plumbing: arguments, runner loading, donors, cell summaries.

Every driver exposes run(runner, ...) -> result dict (so tests can call it on
random weights) and main() (which loads the checkpoint or stops).
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from harness import Runner  # noqa: E402
from measure import capture  # noqa: E402
from prompts import Prompt, covariates  # noqa: E402
from sets import (  # noqa: E402
    MODES, load_train, recipients, require_checkpoint, results_file,
    test_pin, train_pin, write_json,
)
from stats import flag_rows, is_split, redirection, split_by_flag, summarize_cell  # noqa: E402
from thoughts import find_donor  # noqa: E402

SPLIT_COVARIATES = ["K", "n_branches", "first_parent_slot", "first_parent_slot_frac",
                    "n_parent_edges", "n_edges", "target_first"]


def make_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--run-name", default="seed0")
    p.add_argument("--device", default="mps")
    p.add_argument("--mode", choices=MODES, default="pilot",
                   help="pilot = test graphs 400-409; n100 = test graphs 0-99")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed for pinned serializations and random draws")
    return p


def load_runner(args):
    ckpt = require_checkpoint(args.run_name)
    return Runner(ckpt, device=args.device, seed=args.seed)


def header(args, name, **extra):
    return {
        "driver": name, "run_name": args.run_name, "mode": args.mode,
        "seed": args.seed, "device": args.device,
        "checkpoint": str(Path("ckpts") / args.run_name / "best.pt"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra,
    }


def recipient_prompts(mode, base_seed):
    """[(gi, sample, Prompt)] for the test-split recipients of this mode."""
    return [(gi, s, Prompt.from_sample(s, test_pin(gi, base_seed))) for gi, s in recipients(mode)]


def random_donor(train, K, rng, E=None, exclude=()):
    """A random training graph with K latent steps (and E edges when given)."""
    for _ in range(100000):
        gi = rng.randrange(len(train))
        s = train[gi]
        if gi in exclude or len(s["steps"]) != K:
            continue
        if E is not None and len(s["edges"]) != E:
            continue
        return gi, s
    raise RuntimeError(f"no training graph with K={K}, E={E}")


def same_answer_donor(train, target, decoy, K, exclude=()):
    """A training graph with the same two candidates, the same correct answer
    and the same solution length: the standing reference for flips that come
    from a broken search rather than a redirected one. (gi, sample) or None."""
    for j, d in enumerate(train):
        if j in exclude:
            continue
        if d["target"] == target and d["neg_target"] == decoy and len(d["steps"]) == K:
            return j, d
    return None


def donor_run(runner, train, gi, base_seed, attn_eager=False):
    """The training graph gi rendered under its own pinned serialization and
    its recycled thoughts."""
    pr = Prompt.from_sample(train[gi], train_pin(gi, base_seed))
    return pr, capture(runner, pr.ids(runner.tok), attn_eager=attn_eager)


def with_delta(split, base_T):
    out = dict(split)
    out["dT"] = split["T"] - base_T
    return out


REFERENCE = {"all": "same_answer_donor/all", "default": "same_answer_donor/intermediates"}


def reference_for(name):
    """Standing reference cell for redirection: the same-answer donor at all K
    for all-K cells, at the intermediate passes for everything else."""
    return REFERENCE["all"] if name.endswith("/all") else REFERENCE["default"]


def cell_rows_of(rows, name):
    out = []
    for r in rows:
        c = r["cells"].get(name)
        if c is None or c.get("skipped"):
            continue
        cr = dict(c)
        cr["gi"] = r["gi"]
        cr["cov"] = r["cov"]
        out.append(cr)
    return out


def summarize(rows, cell_names, covariate_names=SPLIT_COVARIATES):
    """Per-cell summaries over graphs; redirection counted against the
    same-answer donor's flips on the same graphs; the split analysis wherever
    a cell's graphs divide into moved and unmoved."""
    out = {}
    for name in cell_names:
        cell_rows = cell_rows_of(rows, name)
        summ = summarize_cell(cell_rows)
        summ["n_skipped"] = sum(1 for r in rows if r["cells"].get(name, {}).get("skipped"))
        if cell_rows:
            flag_rows(cell_rows)
            ref = reference_for(name)
            if ref in cell_names and not name.startswith("same_answer_donor"):
                ref_rows = flag_rows(cell_rows_of(rows, ref))
                summ["redirection"] = dict(redirection(cell_rows, ref_rows), reference=ref)
            summ["split"] = {}
            for flag in ("flipped", "escaped", "moved", "redirected"):
                if is_split(cell_rows, flag):
                    summ["split"][flag] = split_by_flag(cell_rows, flag, covariate_names)
        out[name] = summ
    return out


def graph_rng(base_seed, gi):
    return random.Random(base_seed * 1_000_003 + gi)


def graph_gen(base_seed, gi):
    return torch.Generator().manual_seed(base_seed * 1_000_003 + gi)


def finish(args, name, result):
    path = results_file(args.run_name, name, args.mode)
    write_json(path, result)
    return path


__all__ = [
    "make_parser", "load_runner", "header", "recipient_prompts", "random_donor",
    "same_answer_donor", "donor_run", "reference_for", "with_delta", "summarize", "graph_rng", "graph_gen", "finish",
    "load_train", "covariates", "Prompt", "SPLIT_COVARIATES",
]
