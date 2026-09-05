#!/usr/bin/env python3
"""Figure: INLP removal trajectories.

Reads an inlp_basis_report*.json written by fit_inlp.py: per node, the AUC of
each freshly fitted probe on the projected residual. Writes
figs/inlp_curves.pdf.

    python scripts/fig_inlp.py --report results/<run>/inlp_basis_report.json [--out PATH]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--report", required=True, help="results/<run>/inlp_basis_report*.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.repo / "figs" / "inlp_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    rep = json.load((args.repo / args.report).open())
    trajs = [v["auc_trajectory"] for v in rep["per_node"].values()
             if not v.get("skipped") and v.get("auc_trajectory")]

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for t in trajs:
        ax.plot(range(len(t)), t, color="0.6", lw=0.7, alpha=0.7)
    # median trajectory over nodes, up to the shortest length that >= half survive
    max_len = max(len(t) for t in trajs)
    med = []
    for i in range(max_len):
        vals = sorted(t[i] for t in trajs if len(t) > i)
        if len(vals) < len(trajs) / 2:
            break
        med.append(vals[len(vals) // 2])
    ax.plot(range(len(med)), med, color="C0", lw=1.8, label="median over nodes")
    ax.axhline(0.5, color="0.2", ls=":", lw=1)
    ax.text(0.3, 0.505, "chance", fontsize=7, color="0.2", va="bottom")
    stop = rep["recipe"]["auc_stop"]
    ax.axhline(stop, color="C3", ls="--", lw=0.9)
    ax.text(0.3, stop + 0.005, f"stopping rule (AUC $\\leq$ {stop})",
            fontsize=7, color="C3", va="bottom")
    ax.set_xlabel("directions removed (fresh probe each round)", fontsize=8)
    ax.set_ylabel("held-out AUC of fresh probe", fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}  ({len(trajs)} node trajectories, "
          f"median k={rep['median_k']}, cap={rep['recipe']['k_max']})")


if __name__ == "__main__":
    main()
