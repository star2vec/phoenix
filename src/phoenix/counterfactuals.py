"""Experiment 3: the counterfactual set. Keep the thought fixed, change one
thing about the prompt.

Cells (each run three ways: thoughts free; intermediates fixed, passes
0..K-2; all K fixed):
  reordered              random permutation of edge slots
  renamed                one consistent relabeling of every node token
                         (exploration only: the model relies on the id-depth
                         convention of ProsQA labels, so this prompt fails
                         with thoughts free; see NOTES.md)
  parent_swap_km2        the target's parent trades labels with a depth K-2
                         node (separating; two labels move)
  parent_swap_same_depth the parent trades labels with another depth K-1
                         node (control: the frontier is unchanged as a set)
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
  qk_subtract/...        the query-key subtraction cells (see qk_cells.py):
                         remove from thought K-1 the direction each layer-2
                         head's query maps onto the answer edge's key
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
    decoy_swap, noncandidate_swap, parent_swap, rename, reorder,
    reorder_unreachable, rewrite_at_depth,
)
from qk_cells import qk_attention_summary, qk_cells  # noqa: E402
from sets import test_pin  # noqa: E402
from common import graph_gen  # noqa: E402

VARIANTS = ("free", "intermediates", "all")


def counterfactual_set(pr, rng):
    """Ordered list of (name, Prompt or None, meta)."""
    out = [
        ("reordered", *reorder(pr, rng)),
        ("renamed", *rename(pr, rng)),
        ("parent_swap_km2", *parent_swap(pr, rng, -1)),
        ("parent_swap_same_depth", *parent_swap(pr, rng, 0)),
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

        # query-key subtraction cells (eager path; dT against the eager baseline)
        qk, qk_meta = qk_cells(runner, pr, own, base_seed, gi, runner.wte.detach(), graph_gen(base_seed, gi))
        metas["qk"] = qk_meta
        for name, sp in qk.items():
            cells[name] = sp
            if name not in cell_names:
                cell_names.append(name)

        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base,
                     "random_donor_gi": d_gi, "same_answer_donor_gi": sad[0] if sad else None,
                     "meta": metas, "cells": cells})
        show = ["reordered/intermediates", "parent_swap_km2/intermediates", "parent_swap_same_depth/intermediates",
                "decoy_swap/all", "noncandidate_swap/all", "rewrite_last/all", "random_donor/intermediates",
                "same_answer_donor/intermediates", "qk_subtract/answer_edge/all_heads"]
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
    summary["qk_attention"] = qk_attention_summary(rows, cell_names)
    summary["beyond_free_twin"] = beyond_free_twin(rows, cell_names)
    return {"rows": rows, "summary": summary, "cells": cell_names}


def beyond_free_twin(rows, cell_names):
    """For every changed prompt run three ways, the paired excess of the fixed
    variants over the free twin: flips and moves (flip or escape) per graph.
    This is the effect of holding the thought fixed, net of what the changed
    prompt does to the model on its own."""
    import numpy as np
    from stats import bootstrap, FLIP_CUT, ESCAPE_CUT
    out = {}
    bases = sorted({c.rsplit("/", 1)[0] for c in cell_names if c.endswith("/free")})
    for b in bases:
        for v in ("intermediates", "all"):
            pairs = []
            for r in rows:
                f, x = r["cells"].get(f"{b}/free"), r["cells"].get(f"{b}/{v}")
                if not f or not x or f.get("skipped") or x.get("skipped"):
                    continue
                fl = lambda c: float(c["dT"] <= FLIP_CUT)
                mv = lambda c: float(c["dT"] <= FLIP_CUT or c["e"] >= ESCAPE_CUT)
                pairs.append((fl(x) - fl(f), mv(x) - mv(f), x["dT"] - f["dT"]))
            if pairs:
                out[f"{b}/{v}"] = {
                    "n": len(pairs),
                    "flips_beyond_free": bootstrap([a for a, _, _ in pairs], np.mean),
                    "moves_beyond_free": bootstrap([m for _, m, _ in pairs], np.mean),
                    "dT_beyond_free_median": bootstrap([d for _, _, d in pairs], np.median),
                }
    return out


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
