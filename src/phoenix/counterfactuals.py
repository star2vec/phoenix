"""Experiment 3: the counterfactual set. Keep the thought fixed, change one
thing about the prompt.

Cells (each run three ways: thoughts free; intermediates fixed, passes
0..K-2; all K fixed):
  reordered              random permutation of edge slots
  renamed                one consistent relabeling of every node token
  unreachable_reordered  only edges with an unreachable source move
  decoy_swap             the edge(s) into the target and edge(s) into the
                         decoy trade slots; the graph is unchanged
  noncandidate_swap      as decoy_swap, but the swapped-in edge leads to an
                         unreachable node that is not a candidate; p_watch is
                         the probability the answer puts on that node, which
                         a broken search cannot produce
  rewrite_last           the cut edge into the target now points to the decoy
  rewrite_d<d>           a cut edge at depth d redirected so the decoy sits at
                         depth K (availability-limited; skips are counted)
Controls per graph: reserialized baseline, self-transplant (must be exactly
zero), random donor and same-answer donor at intermediates and at all K. A
flip counts as redirection only if the same graph did not flip under the
same-answer donor (summary key "redirection").

The change in T is against the graph's pinned baseline. For a renamed prompt
"target" is the renamed label; for a rewritten prompt it stays the original
target, so following the edit shows as T falling toward 0.

    python src/phoenix/counterfactuals.py --run-name seed0 --device mps --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_rng, header, load_runner,
    load_train, make_parser, random_donor, recipient_prompts, same_answer_donor,
    summarize, with_delta,
)
from measure import all_passes, capture, fixed, intermediates, measure  # noqa: E402
from prompts import (  # noqa: E402
    decoy_swap, noncandidate_swap, rename, reorder, reorder_unreachable,
    rewrite_at_depth,
)
from sets import test_pin  # noqa: E402

VARIANTS = ("free", "intermediates", "all")


def counterfactual_set(pr, rng):
    """Ordered list of (name, Prompt or None, meta)."""
    out = [
        ("reordered", *reorder(pr, rng)),
        ("renamed", *rename(pr, rng)),
        ("unreachable_reordered", *reorder_unreachable(pr, rng)),
        ("decoy_swap", *decoy_swap(pr)),
        ("noncandidate_swap", *noncandidate_swap(pr)),
        ("rewrite_last", *rewrite_at_depth(pr, pr.K, rng)),
    ]
    for d in range(1, pr.K):
        out.append((f"rewrite_d{d}", *rewrite_at_depth(pr, d, rng)))
    return out


def run(runner, recips, train, base_seed=0):
    rows = []
    cell_names = []
    for gi, sample, pr in recips:
        K = pr.K
        own = capture(runner, pr.ids(runner.tok))
        base = measure(runner, pr)
        rng = graph_rng(base_seed, gi)
        cells, metas = {}, {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in cell_names:
                cell_names.append(name)

        cell("reserialized", measure(runner, Prompt.from_sample(sample, test_pin(gi, base_seed, reserial=True))))
        cell("self_transplant", measure(runner, pr, fixed(own, all_passes(K))))
        assert cells["self_transplant"]["dT"] == 0.0, "self-transplant is not exactly zero"
        d_gi, _ = random_donor(train, K, rng)
        _, donor = donor_run(runner, train, d_gi, base_seed)
        cell("random_donor/intermediates", measure(runner, pr, fixed(donor, intermediates(K))))
        cell("random_donor/all", measure(runner, pr, fixed(donor, all_passes(K))))
        sad = same_answer_donor(train, pr.target, pr.decoy, K)
        for v, passes in (("intermediates", intermediates(K)), ("all", all_passes(K))):
            if sad is None:
                cells[f"same_answer_donor/{v}"] = {"skipped": True, "reason": "no_same_answer_donor"}
                if f"same_answer_donor/{v}" not in cell_names:
                    cell_names.append(f"same_answer_donor/{v}")
            else:
                cell(f"same_answer_donor/{v}", measure(runner, pr, fixed(donor_run(runner, train, sad[0], base_seed)[1], passes)))

        for name, cf, meta in counterfactual_set(pr, rng):
            metas[name] = meta
            if cf is None:
                for v in VARIANTS:
                    cells[f"{name}/{v}"] = {"skipped": True, "reason": meta.get("reason")}
                    if f"{name}/{v}" not in cell_names:
                        cell_names.append(f"{name}/{v}")
                continue
            edits = {"free": None, "intermediates": fixed(own, intermediates(K)), "all": fixed(own, all_passes(K))}
            for v in VARIANTS:
                cell(f"{name}/{v}", measure(runner, cf, edits[v], watch=meta.get("watch")))

        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base,
                     "random_donor_gi": d_gi, "same_answer_donor_gi": sad[0] if sad else None,
                     "meta": metas, "cells": cells})
        show = ["reordered/intermediates", "renamed/intermediates", "unreachable_reordered/intermediates",
                "decoy_swap/all", "noncandidate_swap/all", "rewrite_last/all", "random_donor/intermediates",
                "same_answer_donor/intermediates"]
        print(f"graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in show if not cells[n].get("skipped")))

    summary = summarize(rows, cell_names)
    # the two swaps can be partial: split their all-K cells by completeness
    for sw in ("decoy_swap", "noncandidate_swap"):
        comp = [dict(r["cells"][f"{sw}/all"], complete=r["meta"][sw].get("complete"))
                for r in rows if not r["cells"].get(f"{sw}/all", {}).get("skipped")]
        summary[f"{sw}/all/by_completeness"] = {
            "complete": summarize_rows([c for c in comp if c["complete"]]),
            "partial": summarize_rows([c for c in comp if not c["complete"]]),
        }
    return {"rows": rows, "summary": summary, "cells": cell_names}


def summarize_rows(rs):
    from stats import summarize_cell
    return summarize_cell(rs)


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    train = load_train()
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "counterfactuals")
    result.update(run(runner, recips, train, args.seed))
    finish(args, "counterfactuals", result)


if __name__ == "__main__":
    main()
