#!/usr/bin/env python3
"""heads.pdf: do the heads at the latents follow edges by content or by slot?

One dot per edge-reading head at the search latents: content score (x) and
position score (y) after a reorder of the edge list. From-scratch seeds 0
and 1 (results/seed<s>/heads_n100.json, query class intermediate_latent),
layer 2 and layer 1 in two colors, seed by marker; the two fine-tuned GPT-2
checkpoints' heads in grey (results/gpt2_*/cells_n100.json,
head_scores_search_latent). Only heads with at least --mass-cut of their
attention on edge slots are drawn (experiment 5's cutoff, protects "this
head reads edges"; a head that barely looks at edges has no meaningful score).

    .venv/bin/python paper/figs/heads.py [--out PATH] [--mass-cut 0.5]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

SEEDS = (("seed0", "o", "seed 0"), ("seed1", "^", "seed 1"))
GPT2 = ("gpt2_dilgren", "gpt2_aswal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.REPO / "paper" / "figs" / "heads.pdf")
    ap.add_argument("--mass-cut", type=float, default=0.5)
    args = ap.parse_args()
    S.setup()
    fig, ax = S.plt.subplots(figsize=(2.9, 2.5))
    g = [v for run in GPT2 for v in S.load(f"results/{run}/cells_n100.json")["head_scores_search_latent"].values()
         if v["mass"] >= args.mass_cut]
    ax.scatter([v["content"] for v in g], [v["position"] for v in g], s=9, color=S.GREY_LIGHT,
               edgecolors="none", label="GPT-2, both checkpoints", zorder=1)
    for run, marker, sl in SEEDS:
        H = S.load(f"results/{run}/heads_n100.json")["summary"]
        for layer, color in ((2, S.BLUE), (1, S.ORANGE)):
            es = [H[f"L{layer}H{h}/intermediate_latent"] for h in range(8)]
            es = [e for e in es if e["slot_mass_a"] and e["slot_mass_a"]["point"] >= args.mass_cut]
            ax.scatter([e["content_score"]["point"] for e in es], [e["position_score"]["point"] for e in es],
                       s=18, marker=marker, color=color, edgecolors="white", linewidths=0.6, zorder=3,
                       label=f"layer {layer}, {sl}")
    S.ref_line(ax, y=0)
    S.ref_line(ax, x=0)
    ax.set_xlim(-0.15, 1.0)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("content score (follows the edge)")
    ax.set_ylabel("position score (keeps the slot)")
    ax.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    S.save(fig, args.out)


if __name__ == "__main__":
    main()
