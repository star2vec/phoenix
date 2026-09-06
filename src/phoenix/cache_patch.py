"""Experiment 4b: memory-cache patching (after Ding et al.).

Donor = the same graph with edges reordered, run naturally. Recipient = the
pinned original with its own thoughts. At every pass, the donor's keys and/or
values at chosen positions replace the recipient's in the chosen layer(s).

Slices:
  edges                every token of every edge slot
  latents_intermediate latent positions holding thoughts 1..K-1
  latents_all          all K latent positions
Cells: for each slice, keys only / values only / both, in layer 1, layer 2,
and both layers. Controls: self patch (recipient's own cache back into
itself; must be zero), the same slices from a random training graph with the
same number of edges and steps (the on-manifold corruption for Ding's
necessity check), and the thought-level random donor and same-answer donor
at intermediate passes (the standing references for breaking and for
fallback flips). Baseline T here is measured on the eager attention path,
the same path the patched runs use.

    python src/phoenix/cache_patch.py --run-name seed0 --device mps --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attn_hooks import AttnHooks  # noqa: E402
from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_rng, header, load_runner,
    load_train, make_parser, random_donor, recipient_prompts, same_answer_donor,
    summarize, with_delta,
)
from measure import answer_split, fixed, intermediates, measure, run_ids  # noqa: E402
from prompts import reorder  # noqa: E402
from sets import train_pin  # noqa: E402

WHICH = {"k": ("k",), "v": ("v",), "kv": ("k", "v")}


def slices(L, K):
    return {
        "edges": [p for s in L["slots"] for p in s if p is not None],
        "latents_intermediate": L["latents"][: K - 1],
        "latents_all": list(L["latents"]),
    }


def record_kv(runner, ids):
    with AttnHooks(runner.model.base_causallm) as h:
        h.record_kv = True
        logits = run_ids(runner, ids, attn_eager=True)
        kv = dict(h.kv)
    return logits, kv


def kv_slice(kv, layer, which, positions):
    t = kv[layer][0 if which == "k" else 1]
    return t[:, positions, :].clone()


def patched_T(runner, ids, kv, layers, whichs, positions, pr):
    with AttnHooks(runner.model.base_causallm) as h:
        for layer in layers:
            for w in whichs:
                h.add_kv_patch(layer, w, positions, kv_slice(kv, layer, w, positions))
        logits = run_ids(runner, ids, attn_eager=True)
    return answer_split(logits, pr.target, pr.decoy)


def run(runner, recips, train, base_seed=0):
    n_layers = len(runner.model.base_causallm.transformer.h)
    layer_sets = {f"L{i + 1}": [i] for i in range(n_layers)}
    layer_sets["both"] = list(range(n_layers))
    rows, cell_names = [], []
    for gi, sample, pr in recips:
        K, L, E = pr.K, pr.layout(), len(pr.edges)
        ids = pr.ids(runner.tok)
        rng = graph_rng(base_seed, gi)
        logits_base, own_kv = record_kv(runner, ids)
        base = answer_split(logits_base, pr.target, pr.decoy)
        rb, meta = reorder(pr, rng)
        _, donor_kv = record_kv(runner, rb.ids(runner.tok))
        r_gi, _ = random_donor(train, K, rng, E=E)
        rp = Prompt.from_sample(train[r_gi], train_pin(r_gi, base_seed))
        _, rand_kv = record_kv(runner, rp.ids(runner.tok))
        assert len(rp.ids(runner.tok)) == len(ids)

        cells = {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in cell_names:
                cell_names.append(name)

        # standing thought-level references (eager path, same baseline)
        r_gi2, _ = random_donor(train, K, rng)
        cell("random_donor/intermediates", measure(runner, pr, fixed(donor_run(runner, train, r_gi2, base_seed)[1], intermediates(K)), attn_eager=True))
        sad = same_answer_donor(train, pr.target, pr.decoy, K)
        if sad is None:
            cells["same_answer_donor/intermediates"] = {"skipped": True, "reason": "no_same_answer_donor"}
            if "same_answer_donor/intermediates" not in cell_names:
                cell_names.append("same_answer_donor/intermediates")
        else:
            cell("same_answer_donor/intermediates", measure(runner, pr, fixed(donor_run(runner, train, sad[0], base_seed)[1], intermediates(K)), attn_eager=True))

        S = slices(L, K)
        cell("self_patch/edges/kv/both", patched_T(runner, ids, own_kv, layer_sets["both"], ("k", "v"), S["edges"], pr))
        assert abs(cells["self_patch/edges/kv/both"]["dT"]) < 1e-6, "self cache patch is not zero"
        for sname, pos in S.items():
            for wname, whichs in WHICH.items():
                for lname, layers in layer_sets.items():
                    cell(f"donor/{sname}/{wname}/{lname}", patched_T(runner, ids, donor_kv, layers, whichs, pos, pr))
                    cell(f"random_graph/{sname}/{wname}/{lname}", patched_T(runner, ids, rand_kv, layers, whichs, pos, pr))
        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base, "reorder_moved": meta["moved"],
                     "random_graph_gi": r_gi, "cells": cells})
        show = ["donor/edges/k/L2", "donor/edges/v/L2", "donor/latents_intermediate/kv/both",
                "random_graph/edges/k/L2"]
        print(f"graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in show))
    return {"rows": rows, "summary": summarize(rows, cell_names), "cells": cell_names}


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    train = load_train()
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "cache_patch")
    result.update(run(runner, recips, train, args.seed))
    finish(args, "cache_patch", result)


if __name__ == "__main__":
    main()
