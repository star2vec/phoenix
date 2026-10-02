"""Experiment 9a: the restore test. Predictions and reading ranges in NOTES.md.

The every-step query-key removal (as in masking.py: one span of eight
directions per pass 0..K-2, directions from the unedited run) is run, and
then part of layer 2's attention output at every latent query is put back
from the unedited run. A part is the per-head sum over chosen keys of weight
x value, taken before c_proj (linear), so putting it back replaces exactly
what those keys contributed. Edge tokens come before the latents, so their
keys and values are the same in both runs; only the weights differ.

Cells:
  removal               the removal alone (must equal masking_n100.json)
  restore/path          plus the part from the path-edge tokens (every
                        shortest-path edge from depth 1 on, masking.py's set)
  restore/offpath       plus the part from the matched off-path edge tokens
  restore/edges_all     plus the part from every edge token
  restore/all           plus the whole layer-2 attention output (edges,
                        earlier thoughts, question tokens)
  restore/path_unedited the path restore on the unedited run (must be zero)
  reserialized, self_transplant

Summary: per restore, the rescue share (fraction of the removal's flipped
graphs that are not flipped under the restore) with a bootstrap interval,
the new flips among graphs the removal left unflipped, and the paired
rescue differences edges_all - path and all - edges_all.

    python src/phoenix/restore.py --run-name seed0 --device cpu --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from attn_hooks import AttnHooks  # noqa: E402
from common import (  # noqa: E402
    Prompt, covariates, finish, header, load_runner, make_parser, recipient_prompts, with_delta,
)
from masking import path_and_offpath_slots  # noqa: E402
from measure import all_passes, answer_split, capture, fixed, run_ids  # noqa: E402
from prompts import on_path_nodes  # noqa: E402
from qk_cells import _Geometry, per_pass_edit, slot_tokens, span_removal  # noqa: E402
from sets import test_pin  # noqa: E402
from stats import FLIP_CUT, bootstrap  # noqa: E402

LAYER = 1  # layer 2
RESTORES = ("path", "offpath", "edges_all", "all")


def key_sets(L, path, off):
    tok = lambda slots: [p for j in slots for p in slot_tokens(L["slots"][j])]  # noqa: E731
    return {"path": tok(path), "offpath": tok(off), "edges_all": tok(range(len(L["slots"]))), "all": "all"}


def run(runner, recips, base_seed=0):
    rows, cell_names = [], []
    for gi, sample, pr in recips:
        K, L, ids = pr.K, pr.layout(), pr.ids(runner.tok)
        G = _Geometry(runner, pr, ids)
        base = G.base_split
        path, off = path_and_offpath_slots(pr, G)
        keys = key_sets(L, path, off)
        qs = L["latents"]
        cells = {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in cell_names:
                cell_names.append(name)

        cell("reserialized", answer_split(run_ids(runner, Prompt.from_sample(sample, test_pin(gi, base_seed, True)).ids(runner.tok), attn_eager=True), pr.target, pr.decoy))
        own = capture(runner, ids, attn_eager=True)
        cell("self_transplant", answer_split(run_ids(runner, ids, fixed(own, all_passes(K)), attn_eager=True), pr.target, pr.decoy))
        assert abs(cells["self_transplant"]["dT"]) < 1e-6

        on = on_path_nodes(pr)
        steps = []
        for k in range(K - 1):
            pe = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2]
            if not pe:
                steps = None
                break
            slot = max(pe, key=lambda j: G.total_attention(L["latents"][k], j))
            steps.append((k, G.directions(L["latents"][k], slot)))
        row = {"gi": gi, "K": K, "cov": covariates(pr), "baseline": base,
               "meta": {"path_slots": path, "offpath_slots": off, "n_keys": {k: (len(v) if v != "all" else None) for k, v in keys.items()}}}
        if steps is None:
            for nm in ["removal"] + [f"restore/{r}" for r in RESTORES]:
                cells[nm] = {"skipped": True, "reason": "no_path_edge_at_some_step"}
                if nm not in cell_names:
                    cell_names.append(nm)
            row["cells"] = cells
            rows.append(row)
            continue
        removal = per_pass_edit({k: span_removal(dirs) for k, dirs in steps})

        # the unedited run's parts, all four key sets
        with AttnHooks(runner.model.base_causallm) as h:
            for r in RESTORES:
                h.record_partial(r, LAYER, qs, keys[r])
            logits = run_ids(runner, ids, None, attn_eager=True)
            stored = {r: dict(h.partials[r]) for r in RESTORES}
        assert abs(answer_split(logits, pr.target, pr.decoy)["T"] - base["T"]) < 1e-4

        def measure_restore(r, edit):
            with AttnHooks(runner.model.base_causallm) as h:
                if r is not None:
                    h.add_restore(LAYER, qs, keys[r], stored[r])
                logits = run_ids(runner, ids, edit, attn_eager=True)
            return answer_split(logits, pr.target, pr.decoy)

        cell("removal", measure_restore(None, removal))
        for r in RESTORES:
            cell(f"restore/{r}", measure_restore(r, removal))
        cell("restore/path_unedited", measure_restore("path", None))
        assert abs(cells["restore/path_unedited"]["dT"]) < 1e-4, cells["restore/path_unedited"]["dT"]
        row["cells"] = cells
        rows.append(row)
        print(f"graph {gi}: base T {base['T']:.1f}  removal {cells['removal']['dT']:+.1f}  " +
              "  ".join(f"{r} {cells[f'restore/{r}']['dT']:+.1f}" for r in RESTORES), flush=True)
    return {"rows": rows, "summary": summarize(rows), "cells": cell_names, "layer": LAYER + 1,
            "queries": "every latent", "restores": list(RESTORES)}


def flipped(c):
    return c["dT"] <= FLIP_CUT


def summarize(rows):
    rs = [r for r in rows if not r["cells"]["removal"].get("skipped")]
    out = {"n": len(rs), "n_skipped": len(rows) - len(rs), "cutoffs": {"flip_dT": FLIP_CUT}}
    fl = [flipped(r["cells"]["removal"]) for r in rs]
    out["removal_frac_flipped"] = bootstrap([float(x) for x in fl], np.mean)
    out["n_removal_flipped"] = int(sum(fl))
    rescued = {}
    for r in RESTORES:
        res = [not flipped(row["cells"][f"restore/{r}"]) for row, f in zip(rs, fl) if f]
        new = [flipped(row["cells"][f"restore/{r}"]) for row, f in zip(rs, fl) if not f]
        rescued[r] = res
        out[f"restore/{r}"] = {
            "rescue_share": bootstrap([float(x) for x in res], np.mean),
            "n_rescued": int(sum(res)), "n_surviving": int(len(res) - sum(res)),
            "new_flips_among_removal_unflipped": int(sum(new)), "n_removal_unflipped": len(new),
            "frac_flipped": bootstrap([float(flipped(row["cells"][f"restore/{r}"])) for row in rs], np.mean),
        }
    out["paired_rescue_edges_all_minus_path"] = bootstrap([float(a) - float(b) for a, b in zip(rescued["edges_all"], rescued["path"])], np.mean)
    out["paired_rescue_all_minus_edges_all"] = bootstrap([float(a) - float(b) for a, b in zip(rescued["all"], rescued["edges_all"])], np.mean)
    out["paired_rescue_path_minus_offpath"] = bootstrap([float(a) - float(b) for a, b in zip(rescued["path"], rescued["offpath"])], np.mean)
    return out


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "restore")
    result.update(run(runner, recips, args.seed))
    finish(args, "restore", result)


if __name__ == "__main__":
    main()
