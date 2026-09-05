"""Thought-norm diagnostic (training graphs only; writes nothing to results/).
Measures, on a trained checkpoint:

 1. the natural norm of a recycled thought;
 2. the answer-branch coefficient <t, u_hat_v> and its fraction of ||t||;
 3. causal load-bearing-ness of the thought (zero it / replace it);
 4. the effect of the unit-renormalization alone vs the subtraction itself.

Usage: .venv/Scripts/python tests/diag_thought_norms.py [--device cuda]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))
sys.path.insert(0, str(ROOT / "vendor" / "reasoning-by-superposition"))

from harness import Runner, bfs_depths, answer_branch_root  # noqa: E402

VENDOR = ROOT / "vendor" / "reasoning-by-superposition"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-name", default="seed0")
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    runner = Runner(
        ROOT / "ckpts" / args.run_name / "best.pt", device=args.device
    )
    data = json.load(
        open(VENDOR / "data/prosqa_train_graph_4_coconut.json")
    )[: args.n]

    norms, coefs, fracs, rows = [], [], [], []
    for sample in data:
        depth = bfs_depths(sample["edges"], sample["root"])
        v = answer_branch_root(sample, depth)
        u = runner.u_hat(v)

        obs = {}

        def observe(t, info, u=u, obs=obs):
            obs["norm"] = float(t.norm())
            obs["coef"] = float(t @ u)
            return t  # unchanged (float-copy noise only)

        T0, _ = runner.measure(sample)
        runner.measure(sample, 1, observe)
        nrm, cf = obs["norm"], obs["coef"]
        norms.append(nrm)
        coefs.append(cf)
        fracs.append(cf / nrm)

        def sub_keepnorm(t, info, u=u):
            t2 = t - (t @ u) * u
            return t2 / t2.norm() * t.norm()

        def zero_thought(t, info):
            return t * 0.0

        def rand_matched_norm(t, info):
            return runner.rand_unit() * t.norm()

        def unit_rescale_only(t, info):
            return t / t.norm()

        def sub_unitnorm(t, info, u=u):
            t2 = t - (t @ u) * u
            return t2 / t2.norm()

        row = {"T0": T0}
        for name, op in [
            ("sub_unitnorm_v1", sub_unitnorm),
            ("sub_keepnorm", sub_keepnorm),
            ("zero", zero_thought),
            ("rand_matched_norm", rand_matched_norm),
            ("unit_rescale_only", unit_rescale_only),
        ]:
            T, _ = runner.measure(sample, 1, op)
            row[f"dT_{name}"] = (T - T0) * 100
        rows.append(row)

    def med(k):
        return statistics.median(r[k] for r in rows)

    print(
        f"thought norm   : median {statistics.median(norms):.3f}  "
        f"min {min(norms):.3f}  max {max(norms):.3f}"
    )
    print(f"answer coef    : median {statistics.median(coefs):.3f}")
    print(f"coef fraction  : median {statistics.median(fracs):.3f}")
    print(f"median T0                  : {med('T0') * 100:.1f}")
    print(f"median dT sub (unit, v1)   : {med('dT_sub_unitnorm_v1'):+.2f}")
    print(f"median dT sub (keep norm)  : {med('dT_sub_keepnorm'):+.2f}")
    print(f"median dT zero thought     : {med('dT_zero'):+.2f}")
    print(f"median dT random @ ||t||   : {med('dT_rand_matched_norm'):+.2f}")
    print(f"median dT unit-rescale only: {med('dT_unit_rescale_only'):+.2f}")


if __name__ == "__main__":
    main()
