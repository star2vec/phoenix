"""Effect sizes with bootstrap intervals, cell summaries, and the split
analysis for cells where graphs divide into flipped and unflipped.

Cutoffs are named where they are used (CLAUDE.md rule 3):
  flip_cut   = -50 on the change in T: "the answer switched to the other
               candidate on this graph"
  escape_cut = 0.5 on e: "the model stopped answering the question"
"""

import numpy as np

FLIP_CUT = -50.0
ESCAPE_CUT = 0.5
N_BOOT = 2000


def bootstrap(values, stat=np.median, n_boot=N_BOOT, seed=0, ci=0.95):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = stat(x[idx], axis=1)
    lo, hi = np.percentile(bs, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"point": float(stat(x)), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def summarize_cell(rows, flip_cut=FLIP_CUT, escape_cut=ESCAPE_CUT):
    """rows: dicts with at least dT and e (and the probability split)."""
    if not rows:
        return {"n": 0}
    d = [r["dT"] for r in rows]
    e = [r["e"] for r in rows]
    out = {
        "n": len(rows),
        "median_dT": bootstrap(d, np.median),
        "mean_dT": bootstrap(d, np.mean),
        "median_T": bootstrap([r["T"] for r in rows], np.median),
        "median_e": bootstrap(e, np.median),
        "frac_flipped": bootstrap([float(x <= flip_cut) for x in d], np.mean),
        "frac_escaped": bootstrap([float(x >= escape_cut) for x in e], np.mean),
        "cutoffs": {"flip_dT": flip_cut, "escape_e": escape_cut},
    }
    for key in ("p_target", "p_decoy", "p_other_node", "p_other_token"):
        if key in rows[0]:
            out["mean_" + key] = bootstrap([r[key] for r in rows], np.mean)
    return out


def flag_rows(rows, flip_cut=FLIP_CUT, escape_cut=ESCAPE_CUT):
    for r in rows:
        r["flipped"] = bool(r["dT"] <= flip_cut)
        r["escaped"] = bool(r["e"] >= escape_cut)
        r["moved"] = r["flipped"] or r["escaped"]
    return rows


def split_by_flag(rows, flag, covariates):
    """Compare graphs with flag True vs False on each covariate (mean and
    count). Reported for any cell where graphs split."""
    groups = {True: [], False: []}
    for r in rows:
        groups[bool(r.get(flag))].append(r)
    out = {}
    for g, rs in groups.items():
        entry = {"n": len(rs)}
        for c in covariates:
            vals = [r["cov"][c] for r in rs if r.get("cov", {}).get(c) is not None]
            entry[c] = (float(np.mean([float(v) for v in vals])) if vals else None)
        out["flagged" if g else "unflagged"] = entry
    out["flag"] = flag
    return out


def is_split(rows, flag="moved", lo=0.1, hi=0.9):
    """A cell 'splits' when between 10% and 90% of graphs carry the flag; the
    band protects the claim 'this cell has two populations of graphs'."""
    if not rows:
        return False
    f = np.mean([bool(r.get(flag)) for r in rows])
    return lo <= f <= hi
