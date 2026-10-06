#!/usr/bin/env python3
"""wave.pdf: the breadth-first wave in the thought, seeds 0 to 3.

Mean inner product between thought k and each node's input embedding, by
node group, at each latent step (results/seed<s>/evaluation_ser0.json,
readout_by_step, held-out test split under serialization seed 0).

    .venv/bin/python paper/figs/wave.py [--out PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

RUNS = ["seed0", "seed1", "seed2", "seed3"]
CATS = ["NotReachable", "Reachable", "Frontier", "Optimal"]
CAT_LABELS = ["not reachable", "reachable", "frontier", "optimal"]
MARKERS = ["o", "s", "^", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.REPO / "paper" / "figs" / "wave.pdf")
    args = ap.parse_args()
    S.setup()
    fig, axes = S.plt.subplots(1, 4, figsize=(S.WIDTH, S.HEIGHT), sharey=True)
    for ax, run in zip(axes, RUNS):
        ro = S.load(f"results/{run}/evaluation_ser0.json")["readout_by_step"]
        steps = sorted(ro, key=int)
        for cat, cl, m, c in zip(CATS, CAT_LABELS, MARKERS, S.RAMP):
            ax.plot([int(k) for k in steps], [ro[k][cat]["mean"] for k in steps], marker=m, color=c, label=cl)
        S.ref_line(ax, y=0)
        S.panel_label(ax, f"seed {run[-1]}")
        ax.set_xticks([int(k) for k in steps])
        ax.set_xlabel("latent step")
    axes[0].set_ylabel(r"mean $\langle$thought, node$\rangle$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="center left", bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout(w_pad=0.6)
    S.save(fig, args.out)


if __name__ == "__main__":
    main()
