"""Shared style for the paper figures in paper/figs/.

One idea per figure, one row of panels, small sans fonts, no titles inside
the plot, recessive axes. Colors are the dataviz reference palette's first
categorical slots (validated all-pairs for CVD: blue and orange worst
protan dE 24.7) plus a neutral grey for references and context, and its
blue ordinal ramp (steps 250, 400, 550, 700; validated --ordinal) for
ordered categories.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
GREY, GREY_LIGHT = "#8a8986", "#c9c8c3"     # context and references
INK, INK_2 = "#0b0b0b", "#52514e"           # text
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal, light to dark

WIDTH = 6.75   # inches, a full text width
HEIGHT = 1.85  # one row


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],  # Arial is a plain TTF on macOS and embeds cleanly
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.edgecolor": GREY, "axes.linewidth": 0.5, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def load(rel):
    return json.load(open(REPO / rel))


def panel_label(ax, text):
    """A small identifier above the panel's top-left corner (not a title)."""
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontsize=7, color=INK_2, ha="left", va="bottom")


def ref_line(ax, y=None, x=None, label=None):
    """A solid hairline reference (chance, zero, a cutoff), optionally labelled at its right end."""
    if y is not None:
        ax.axhline(y, color=GREY_LIGHT, lw=0.6, zorder=0)
        if label:
            ax.text(1.0, y, " " + label, transform=ax.get_yaxis_transform(), fontsize=6, color=INK_2, va="center", ha="left")
    if x is not None:
        ax.axvline(x, color=GREY_LIGHT, lw=0.6, zorder=0)


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
