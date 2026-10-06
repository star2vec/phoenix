"""Experiment 11: the INLP (amnesic probing) arm. Predictions in NOTES.md.

The recipients and edit of baseline.py's subtraction cell (test graphs whose
target has a unique depth-1 ancestor; that ancestor's identity removed from
the step-1 thought at pass 0, norm preserved), under four deletions:
  subtract_answer/probe                 the probe direction (as in
                                        baseline_subtraction_<mode>.json)
  subtract_answer/inlp_first            the first INLP direction
  subtract_answer/inlp                  the full INLP span of that node
  subtract_answer/inlp_random_matched   as many random orthonormal directions
                                        (rank-matched control, Elazar et al.)
  subtract_answer/inlp_random_sizematched  a random span of the same rank holding
                                        the same share of the step-1 thought's
                                        squared norm as the INLP span
Bases from fit_inlp.py (the v1 full recipe: --k-max 40 --lbfgs-iters 50
--tag _full). Writes results/<run>/inlp_arm_<mode>.json.

    python src/phoenix/inlp_arm.py --run-name seed0 --device cpu --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_gen, graph_rng, header, load_runner, load_train,
    make_parser, random_donor, recipient_prompts, same_answer_donor, summarize, with_delta,
)
from measure import all_passes, at_passes, capture, fixed, intermediates, measure  # noqa: E402
from prompts import unique_answer_branch  # noqa: E402
from qk_cells import span_removal  # noqa: E402
from sets import ROOT, require_file, test_pin  # noqa: E402

TAG = "_full"


def at_step_one(units):
    """Norm-preserving removal of the span of `units` from the step-1 thought (pass 0)."""
    f = span_removal(units)
    return at_passes(lambda k, t: f(t), [0])


def sizematched_span(t, s, k, gen):
    """k orthonormal directions whose span holds exactly the share s of t's
    squared norm: u = sqrt(s) t/|t| + sqrt(1-s) r (r random, orthogonal to t)
    and k-1 random directions orthogonal to both t and r. Drawn from `gen`
    after the rank-matched control, so that control is unchanged."""
    th = (t / t.norm()).cpu()
    G = torch.randn(t.shape[0], k, generator=gen)
    Q, _ = torch.linalg.qr(torch.cat([th[:, None], G], dim=1))
    r = Q[:, 1]
    u = s ** 0.5 * Q[:, 0] + (1 - s) ** 0.5 * r
    return torch.cat([u[:, None], Q[:, 2:k + 1]], dim=1).to(t.device)


def run(runner, recips, train, probe, basis, report, base_seed=0):
    rows, names, skipped = [], [], {"no_unique_ancestor": [], "no_inlp_basis": []}
    for gi, sample, pr in recips:
        ab = unique_answer_branch(pr)
        if ab is None:
            skipped["no_unique_ancestor"].append(gi)
            continue
        v, _ = ab
        if v not in basis:
            skipped["no_inlp_basis"].append(gi)
            continue
        K = pr.K
        B = basis[v].to(runner.device)
        k = B.shape[1]
        gen = graph_gen(base_seed, gi)
        R, _ = torch.linalg.qr(torch.randn(B.shape[0], k, generator=gen))
        R = R[:, :k].to(runner.device)
        own = capture(runner, pr.ids(runner.tok))
        base = measure(runner, pr)
        t1 = own[0].to(runner.device).float()
        share = lambda Q: float(((Q.T @ t1) ** 2).sum() / (t1 @ t1))  # noqa: E731  squared-norm share of the step-1 thought in span(Q)
        S = sizematched_span(t1, share(B), k, gen)
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

        p = probe[v].to(runner.device)
        cell("subtract_answer/probe", measure(runner, pr, at_step_one([p / p.norm()])))
        cell("subtract_answer/inlp_first", measure(runner, pr, at_step_one([B[:, 0]])))
        cell("subtract_answer/inlp", measure(runner, pr, at_step_one(list(B.T))))
        cell("subtract_answer/inlp_random_matched", measure(runner, pr, at_step_one(list(R.T))))
        cell("subtract_answer/inlp_random_sizematched", measure(runner, pr, at_step_one(list(S.T))))
        traj = report["per_node"][str(v)]["auc_trajectory"]
        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base, "answer_branch_root": v,
                     "k": k, "residual_auc": traj[-1],
                     "removed_share": {"inlp": share(B), "random_matched": share(R), "random_sizematched": share(S),
                                       "probe": share((p / p.norm())[:, None])},
                     "cells": cells})
        print(f"test graph {gi}: base T {base['T']:.1f}  node {v} k {k}  " + "  ".join(
            f"{n.split('/')[-1]} {cells[n]['dT']:+.2f}" for n in names if n.startswith("subtract")))
    return {"rows": rows, "summary": summarize(rows, names), "cells": names, "skipped": skipped,
            "basis_tag": TAG, "recipe": report["recipe"]}


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    rdir = ROOT / "results" / args.run_name
    probe = torch.load(require_file(rdir / "probe_basis.pt", "fit_probes.py"), map_location="cpu", weights_only=True)
    basis = torch.load(require_file(rdir / f"inlp_basis{TAG}.pt", "fit_inlp.py --k-max 40 --lbfgs-iters 50 --tag _full"),
                       map_location="cpu", weights_only=True)
    import json
    report = json.load(open(rdir / f"inlp_basis_report{TAG}.json"))
    result = header(args, "inlp_arm")
    result.update(run(runner, recipient_prompts(args.mode, args.seed), load_train(), probe, basis, report, args.seed))
    finish(args, "inlp_arm", result)


if __name__ == "__main__":
    main()
