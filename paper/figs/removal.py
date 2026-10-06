#!/usr/bin/env python3
"""removal.pdf: the query-key removal empties the attention, the answer mostly stays.

On the same graphs (test graphs 0-99 of one seed, the every-step all-heads
removal; results/<run>/counterfactuals_n100.json, cell
qk_every_step/answer_path/all_heads): left, the layer-2 attention from the
last search query onto the answer-path edge, summed over the eight heads,
before and after the removal, paired per graph; right, the change in T per
graph, sorted, with the flip cutoff (-50) as a hairline.

    .venv/bin/python paper/figs/removal.py [--run seed0] [--out PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import style as S  # noqa: E402

CELL = "qk_every_step/answer_path/all_heads"
FLIP_CUT = -50.0  # protects "the answer switched on this graph" (as everywhere)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="seed0")
    ap.add_argument("--out", type=Path, default=S.REPO / "paper" / "figs" / "removal.pdf")
    args = ap.parse_args()
    S.setup()
    rows = [r["cells"][CELL] for r in S.load(f"results/{args.run}/counterfactuals_n100.json")["rows"]
            if not r["cells"][CELL].get("skipped")]
    before = np.array([c["attn_path_before_per_step"][-1] for c in rows])
    after = np.array([c["attn_path_after_per_step"][-1] for c in rows])
    dT = np.sort(np.array([c["dT"] for c in rows]))

    fig, (a, b) = S.plt.subplots(1, 2, figsize=(S.WIDTH * 0.72, S.HEIGHT), gridspec_kw={"width_ratios": [1, 1.6]})
    for x0, x1 in zip(before, after):
        a.plot([0, 1], [x0, x1], color=S.GREY_LIGHT, lw=0.4, zorder=1)
    a.scatter(np.zeros_like(before), before, s=6, color=S.BLUE, edgecolors="none", zorder=2)
    a.scatter(np.ones_like(after), after, s=6, color=S.BLUE, edgecolors="none", zorder=2)
    a.plot([0, 1], [np.median(before), np.median(after)], color=S.INK, lw=1.2, marker="o", ms=3.5, zorder=3, label="median")
    a.set_xticks([0, 1], ["before", "after"])
    a.set_xlim(-0.3, 1.3)
    a.set_ylim(bottom=0)
    a.set_ylabel("attention onto the\nanswer-path edge")
    a.legend(loc="upper right")
    S.panel_label(a, "attention, per graph")

    fl = dT <= FLIP_CUT
    rank = np.arange(len(dT))
    b.scatter(rank[~fl], dT[~fl], s=6, color=S.BLUE, edgecolors="none", label="answer kept")
    b.scatter(rank[fl], dT[fl], s=6, color=S.ORANGE, edgecolors="none", label="answer switched")
    S.ref_line(b, y=0)
    S.ref_line(b, y=FLIP_CUT, label="flip cutoff")
    b.set_ylim(-104, max(8.0, dT.max() + 4))
    b.set_xlabel("graphs, sorted")
    b.set_xticks([])
    b.set_ylabel("change in T")
    b.legend(loc="lower right")
    S.panel_label(b, "the answer, per graph")
    fig.tight_layout(w_pad=1.2)
    S.save(fig, args.out)


if __name__ == "__main__":
    main()
