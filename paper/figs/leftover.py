#!/usr/bin/env python3
"""leftover.pdf: what carries the answer once every reader of the path is blocked.

Experiment 10, read in results/gaps_reanalysis.json (key experiment10),
baseline-correct graphs. Left: the share still right with the path slots
isolated, and with the path slots plus every incoming edge of both
candidates isolated; pooled over seeds 0-3 with a 95 percent bootstrap
interval, each seed as a small grey dot, chance (0.5) as a hairline.
Right: the leftover above chance split into three pooled shares with
intervals: the candidates' incoming edges (E), the label-id order (N,
correlational) and the remainder (U).

    .venv/bin/python paper/figs/leftover.py [--out PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

SEEDS = ("seed0", "seed1", "seed2", "seed3")
CELLS = (("s_iso", "path\nisolated"), ("s_1", "path and\ncandidate edges"))
SHARES = (("E", "candidates' edges"), ("N", "label-id order"), ("U", "remainder"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.REPO / "paper" / "figs" / "leftover.pdf")
    args = ap.parse_args()
    S.setup()
    e10 = S.load("results/gaps_reanalysis.json")["experiment10"]
    P = e10["pooled"]["all"]

    fig, (a, b) = S.plt.subplots(1, 2, figsize=(S.WIDTH * 0.72, S.HEIGHT), gridspec_kw={"width_ratios": [1, 1.4]})
    for i, (k, label) in enumerate(CELLS):
        for j, run in enumerate(SEEDS):
            a.scatter(i - 0.18 + 0.06 * j, e10["seeds"][run]["all"][k]["point"], s=6, color=S.GREY, edgecolors="none", zorder=2,
                      label="each seed" if (i, j) == (0, 0) else None)
        p = P[k]
        a.errorbar(i + 0.12, p["point"], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]], fmt="o", ms=4,
                   color=S.BLUE, ecolor=S.BLUE, elinewidth=1.0, capsize=0, zorder=3, label="pooled" if i == 0 else None)
    S.ref_line(a, y=0.5, label="chance")
    a.set_xticks(range(len(CELLS)), [lab for _, lab in CELLS])
    a.set_xlim(-0.5, len(CELLS) - 0.5)
    a.set_ylim(0.4, 1.0)
    a.set_ylabel("share still right")
    a.legend(loc="upper right")
    S.panel_label(a, "survival")

    for i, (k, label) in enumerate(SHARES):
        p = P[k]
        y = len(SHARES) - 1 - i
        b.errorbar(p["point"], y, xerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]], fmt="o", ms=4,
                   color=S.BLUE, ecolor=S.BLUE, elinewidth=1.0, capsize=0, zorder=3)
    S.ref_line(b, x=0)
    b.set_yticks(range(len(SHARES)), [lab for _, lab in SHARES][::-1])
    b.set_ylim(-0.6, len(SHARES) - 0.4)
    b.set_xlim(-0.1, 1.0)
    b.set_xlabel("share of the leftover above chance (pooled)")
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    S.panel_label(b, "what carries it")
    fig.tight_layout(w_pad=1.2)
    S.save(fig, args.out)


if __name__ == "__main__":
    main()
