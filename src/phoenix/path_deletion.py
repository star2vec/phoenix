"""Experiment 12: whole answer-path deletion at every intermediate step, in the
identity bases (the v1 cell perstep_sub_path_nolast, ported). Predictions in NOTES.md.

At each intermediate pass k = 0..K-2 the thought loses the span of the
identity directions of every node on a shortest path to the target at depth
k+1 (the nodes that step's frontier holds on the answer path), norm
preserved. Bases: input embeddings, and the step-1 probe basis
(probe_basis.pt; nodes without a fitted direction are left out and counted).
Control: the same number of random orthonormal directions at the same passes
(rank-matched to the input-embedding span). Recipients: test graphs (pilot
400-409, n100 0-99). Writes results/<run>/path_deletion_<mode>.json.

    python src/phoenix/path_deletion.py --run-name seed0 --device cpu --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_gen, graph_rng, header, load_runner, load_train,
    make_parser, random_donor, recipient_prompts, same_answer_donor, summarize, with_delta,
)
from measure import all_passes, capture, fixed, intermediates, measure  # noqa: E402
from prompts import on_path_nodes  # noqa: E402
from qk_cells import per_pass_edit, span_removal  # noqa: E402
from sets import ROOT, require_file, test_pin  # noqa: E402


def run(runner, recips, train, probe, base_seed=0):
    wte = runner.wte.detach()
    rows, names = [], []
    for gi, sample, pr in recips:
        K = pr.K
        on = on_path_nodes(pr)
        nodes = {k: sorted(v for v, d in on.items() if d == k + 1) for k in range(K - 1)}
        gen = graph_gen(base_seed, gi)
        emb, prb, rnd, missing = {}, {}, {}, 0
        for k, vs in nodes.items():
            if not vs:
                continue
            emb[k] = span_removal([(wte[v] / wte[v].norm()).to(runner.device) for v in vs])
            pv = [probe[v] for v in vs if probe[v].norm() > 0]
            missing += len(vs) - len(pv)
            if pv:
                prb[k] = span_removal([(p / p.norm()).to(runner.device) for p in pv])
            R, _ = torch.linalg.qr(torch.randn(wte.shape[1], len(vs), generator=gen))
            rnd[k] = span_removal([c.to(runner.device) for c in R[:, :len(vs)].T])
        own = capture(runner, pr.ids(runner.tok))
        base = measure(runner, pr)
        cells = {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in names:
                names.append(name)

        cell("reserialized", measure(runner, Prompt.from_sample(sample, test_pin(gi, base_seed, reserial=True))))
        cell("self_transplant", measure(runner, pr, fixed(own, all_passes(K))))
        assert cells["self_transplant"]["dT"] == 0.0
        sad = same_answer_donor(train, pr.target, pr.decoy, K)
        if sad is not None:
            _, sth = donor_run(runner, train, sad[0], base_seed)
            cell("same_answer_donor/intermediates", measure(runner, pr, fixed(sth, intermediates(K))))
        r_gi, _ = random_donor(train, K, graph_rng(base_seed, gi))
        _, rth = donor_run(runner, train, r_gi, base_seed)
        cell("random_donor/intermediates", measure(runner, pr, fixed(rth, intermediates(K))))

        cell("path_mid/input_embedding", measure(runner, pr, per_pass_edit(emb)))
        cell("path_mid/probe", measure(runner, pr, per_pass_edit(prb)))
        cell("path_mid/random_matched", measure(runner, pr, per_pass_edit(rnd)))
        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base,
                     "path_nodes_by_pass": {str(k): v for k, v in nodes.items()},
                     "probe_missing_nodes": missing, "cells": cells})
        print(f"test graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n.split('/')[-1]} {cells[n]['dT']:+.2f}/e{cells[n]['e']:.2f}" for n in names if n.startswith("path")))
    return {"rows": rows, "summary": summarize(rows, names), "cells": names}


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    probe = torch.load(require_file(ROOT / "results" / args.run_name / "probe_basis.pt", "fit_probes.py"),
                       map_location="cpu", weights_only=True)
    result = header(args, "path_deletion")
    result.update(run(runner, recipient_prompts(args.mode, args.seed), load_train(), probe, args.seed))
    finish(args, "path_deletion", result)


if __name__ == "__main__":
    main()
