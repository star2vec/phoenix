#!/usr/bin/env python3
"""Figure: the BFS-wave readout, one panel per run.

Reads results/<run>/evaluation.json (readout_by_step: per-step, per-group
mean inner products between the thought and each node embedding) written by
src/phoenix/evaluate.py, and writes figs/readout.pdf.

    python scripts/fig_readout.py --runs seed0 seed1 [--labels "seed 0" "seed 1"] [--out PATH]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

CATS = ["NotReachable", "Reachable", "Frontier", "Optimal"]
CAT_LABELS = ["not reachable", "reachable", "frontier", "optimal"]
MARKERS = ["o", "s", "^", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--runs", nargs="+", required=True, help="results/<run> directories")
    ap.add_argument("--labels", nargs="*", default=None, help="panel titles (default: run names)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.repo / "figs" / "readout.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = args.labels or args.runs
    assert len(labels) == len(args.runs), "one label per run"

    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 0.1, 2.6), sharey=True, squeeze=False)
    for ax, run, label in zip(axes[0], args.runs, labels):
        ev = json.load((args.repo / "results" / run / "evaluation.json").open())
        ro = ev["readout_by_step"]
        steps = sorted(ro, key=int)
        for cat, cl, m in zip(CATS, CAT_LABELS, MARKERS):
            ys = [ro[k][cat]["mean"] for k in steps]
            ax.plot([int(k) for k in steps], ys, marker=m, ms=4, lw=1.2, label=cl)
        acc_pct = 100 * ev["test_accuracy"]
        ax.set_title(f"{label}\nacc {acc_pct:.1f}%", fontsize=8)
        ax.set_xlabel("latent step $k$", fontsize=8)
        ax.set_xticks([int(k) for k in steps])
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel(r"mean $\langle \mathbf{z}_k, \mathbf{u}_v\rangle$", fontsize=8)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
