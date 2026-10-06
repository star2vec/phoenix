#!/usr/bin/env python3
"""winner.pdf: when does the thought know which candidate wins?

Held-out AUC of the learned linear winner probe (thought -> the target's
node token, scored on target vs decoy) at each latent step, last 500
training graphs (results/<run>/winner_probe.json, learned_probe). From
scratch, seed 0 (thick) and seed 1 (thin); fine-tuned GPT-2, the Dilgren
checkpoint (thick) and the Aswal checkpoint (thin). From scratch, step 4
exists only on four-step graphs.

    .venv/bin/python paper/figs/winner.py [--out PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

LINES = (("seed0", S.BLUE, 1.6, "from scratch, seed 0"), ("seed1", S.BLUE, 0.7, "from scratch, seed 1"),
         ("gpt2_dilgren", S.ORANGE, 1.6, "GPT-2, Dilgren"), ("gpt2_aswal", S.ORANGE, 0.7, "GPT-2, Aswal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.REPO / "paper" / "figs" / "winner.pdf")
    args = ap.parse_args()
    S.setup()
    fig, ax = S.plt.subplots(figsize=(2.9, S.HEIGHT))
    for run, color, lw, label in LINES:
        lp = S.load(f"results/{run}/winner_probe.json")["learned_probe"]
        ks = sorted(lp, key=int)
        ax.plot([int(k) for k in ks], [lp[k]["auc"] for k in ks], color=color, lw=lw,
                marker="o" if lw > 1 else None, ms=3, label=label)
    S.ref_line(ax, y=0.5, label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("latent step")
    ax.set_ylabel("winner probe AUC")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2))  # above the chance line
    S.save(fig, args.out)


if __name__ == "__main__":
    main()
