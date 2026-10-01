"""Experiment 8a: the winner signal in thoughts K-1 and K, per test graph,
unedited and under the query-key every-step removal. Predictions in NOTES.md.

For each recipient the removal and its matched random control are rebuilt as
in masking.py (directions from the unedited run; one span of eight directions
per pass 0..K-2). Four runs record the recycled thought at every pass as the
model reads it, after the edit, and the layer-2 attention from each search
query onto that step's answer-path edge: unedited ("none"), "removal",
"removal_orth" (added after the pilot: the same span with its component
along the winner readout direction taken out first, so the edit leaves that
direction alone) and "random". On thought K-1 (pass K-2) and thought K
(pass K-1) two readouts give a winner margin, target minus decoy: the
learned per-K winner probe (results/<run>/winner_probe_weights.pt from
winner_probe.py, fit on training graphs) and the input-embedding readout.
Before any margin is read, the share of each readout direction that lies in
the removed span at each pass is recorded (squared norm of its projection
onto the span); eight random orthonormal directions hold about 8/768 = 0.01
of any direction by chance, and the matched random span is recorded as that
reference. The winner direction used for the orthogonal removal at a pass is
the probe's target-minus-decoy direction there, else the input-embedding one.

Summary: flip rates per condition with the paired difference of the
orthogonal removal against the removal; the attention onto the answer-path
edge at the last search step per condition, and the orthogonal removal's
drop as a fraction of the removal's; AUC (bootstrap over graphs) for "the
removal flips this graph" from baseline p_decoy, from the parent out-degree,
from each margin (lower margin, higher score) and from the span share, on
all graphs and on baseline-correct ones; the paired change in the K-1 and K
margins under each edit, on flipped and unflipped graphs; the span shares on
flipped and unflipped graphs; and whether the K margin under the removal
predicts the answer the model gives under the removal.

    python src/phoenix/winner_margin.py --run-name seed0 --device cpu --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attn_hooks import AttnHooks  # noqa: E402
from common import (  # noqa: E402
    covariates, finish, graph_gen, header, load_runner, make_parser, recipient_prompts, with_delta,
)
from edits import rand_orthonormal  # noqa: E402
from measure import answer_split, run_ids  # noqa: E402
from prompts import on_path_nodes  # noqa: E402
from qk_cells import _Geometry, attention_on, per_pass_edit, span_removal  # noqa: E402
from sets import ROOT, require_file  # noqa: E402
from stats import FLIP_CUT, N_BOOT, bootstrap  # noqa: E402

N_DIRECTIONS = 8
CHANCE_SPAN_SHARE = N_DIRECTIONS / 768
CORRECT_T = 50.0  # protects "the baseline answer was the target"
CONDITIONS = ("none", "removal", "removal_orth", "random")
EDITS = ("removal", "removal_orth", "random")
READOUTS = ("probe", "embedding")


def load_probes(run_name):
    path = require_file(ROOT / "results" / run_name / "winner_probe_weights.pt",
                        f"winner_probe.py --model symbol --run-name {run_name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def unit(v):
    return v / v.norm().clamp_min(1e-12)


@torch.no_grad()
def run_recording(runner, pr, ids, edit, query_slots, n_heads):
    """The answer split, the recycled thought at every pass as the model
    reads it (after `edit`, when given), and the layer-2 attention summed
    over heads from each (query, slot) pair in query_slots."""
    cap = {}

    def f(k, t):
        t2 = edit(k, t) if edit is not None else t
        cap[k] = t2.detach().float().cpu().clone()
        return t2

    with AttnHooks(runner.model.base_causallm) as h:
        h.record_weights = True
        logits = run_ids(runner, ids, f, attn_eager=True)
        weights = h.weights
    rec = AttnHooks.__new__(AttnHooks)
    rec.weights = weights
    L = pr.layout()
    attn = [sum(attention_on(rec, q, L["slots"][j], n_heads)) for q, j in query_slots]
    return answer_split(logits, pr.target, pr.decoy), cap, attn


def probe_direction(probe, t, d):
    """Unit target-minus-decoy weight direction of a learned probe, or None."""
    if probe is None or t == d or t not in probe["classes"] or d not in probe["classes"]:
        return None
    ci = {c: i for i, c in enumerate(probe["classes"])}
    return unit(probe["W"][ci[t]] - probe["W"][ci[d]])


def probe_margin(probe, x, t, d):
    if probe is None or t == d or t not in probe["classes"] or d not in probe["classes"]:
        return None
    ci = {c: i for i, c in enumerate(probe["classes"])}
    logit = x @ probe["W"].T + probe["b"]
    return float(logit[ci[t]] - logit[ci[d]])


def span_share(direction, Q):
    """Squared norm of a unit direction's projection onto the span of the
    orthonormal columns of Q."""
    return float(((Q.T @ direction) ** 2).sum())


def orthonormal_span(units):
    Q, _ = torch.linalg.qr(torch.stack(units, dim=1))
    return Q[:, :len(units)]


def orthogonal_units(Q, d):
    """The columns of Q with their component along the unit direction d
    removed (the span to remove when d must be left alone)."""
    d = d.to(Q)
    M = Q - torch.outer(d, d @ Q)
    return [M[:, i] for i in range(M.shape[1])]


def rank_auc(scores, labels):
    """Rank AUC with tied scores averaged; None when one class is empty."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_boot(scores, labels, n_boot=N_BOOT, seed=0):
    """AUC of `scores` for `labels` with a bootstrap interval over graphs;
    graphs with a missing score are dropped and counted."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    n_missing = len(scores) - len(pairs)
    if not pairs:
        return {"point": None, "n": 0, "n_missing": n_missing}
    s = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], bool)
    point = rank_auc(s, y)
    out = {"point": point, "n": int(len(s)), "n_pos": int(y.sum()), "n_missing": n_missing}
    if point is None:
        return out
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        a = rank_auc(s[idx], y[idx])
        if a is not None:
            bs.append(a)
    out["lo"], out["hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return out


def neg(x):
    return None if x is None else -x


def run(runner, recips, probes, base_seed=0):
    rows, cell_names = [], []
    wte = runner.wte.detach().float().cpu()
    for gi, sample, pr in recips:
        K, L, ids = pr.K, pr.layout(), pr.ids(runner.tok)
        G = _Geometry(runner, pr, ids)
        base = G.base_split
        gen = graph_gen(base_seed, gi)
        on = on_path_nodes(pr)
        steps = []
        for k in range(K - 1):
            qk = L["latents"][k]
            pe = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2]
            if not pe:
                steps = None
                break
            slot = max(pe, key=lambda j: G.total_attention(qk, j))
            steps.append((k, slot, G.directions(qk, slot)))
        row = {"gi": gi, "K": K, "cov": covariates(pr), "baseline": base}
        if steps is None:
            row.update({"skipped": True, "reason": "no_path_edge_at_some_step"})
            rows.append(row)
            continue
        spans = {k: orthonormal_span(dirs).cpu() for k, _, dirs in steps}
        rand_spans = {k: rand_orthonormal(N_DIRECTIONS, 768, gen, dirs[0].device) for k, _, dirs in steps}

        # the readout directions, their share inside each span, and the
        # orthogonal removal's span, all before any edited run
        emb_dir = unit(wte[pr.target] - wte[pr.decoy])
        shares, orth_units = {}, {}
        for k, _, _ in steps:
            pdir = probe_direction(probes.get(f"K{K}/k{k}"), pr.target, pr.decoy)
            R = rand_spans[k].cpu()
            wdir = pdir if pdir is not None else emb_dir
            orth_units[k] = orthogonal_units(spans[k], wdir)
            shares[f"pass{k}"] = {
                "probe_in_removed_span": span_share(pdir, spans[k]) if pdir is not None else None,
                "probe_in_random_span": span_share(pdir, R) if pdir is not None else None,
                "embedding_in_removed_span": span_share(emb_dir, spans[k]),
                "embedding_in_random_span": span_share(emb_dir, R),
                "winner_direction_for_orth": "probe" if pdir is not None else "embedding",
                "winner_in_orth_span": span_share(wdir, orthonormal_span(orth_units[k])),
            }
        row["span_shares"] = shares
        edits = {
            "none": None,
            "removal": per_pass_edit({k: span_removal(dirs) for k, _, dirs in steps}),
            "removal_orth": per_pass_edit({k: span_removal(orth_units[k]) for k, _, _ in steps}),
            "random": per_pass_edit({k: span_removal(dirs, rand=rand_spans[k]) for k, _, dirs in steps}),
        }
        query_slots = [(L["latents"][k], slot) for k, slot, _ in steps]
        row["path_slots"] = [slot for _, slot, _ in steps]

        cells = {}
        for name in CONDITIONS:
            split, cap, attn = run_recording(runner, pr, ids, edits[name], query_slots, G.n_heads)
            c = with_delta(split, base["T"])
            for label, k in (("Km1", K - 2), ("K", K - 1)):
                x = cap[k]
                c[f"margin_probe_{label}"] = probe_margin(probes.get(f"K{K}/k{k}"), x, pr.target, pr.decoy)
                c[f"margin_embedding_{label}"] = float(x @ unit(wte[pr.target]) - x @ unit(wte[pr.decoy]))
            c["attn_path_per_step"] = attn
            c["attn_path_last"] = attn[-1]
            cells[name] = c
            if name not in cell_names:
                cell_names.append(name)
        assert abs(cells["none"]["dT"]) < 1e-4, cells["none"]["dT"]
        row["cells"] = cells
        rows.append(row)
        sh = shares[f"pass{K - 2}"]["probe_in_removed_span"]
        fm = lambda v: "None" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"graph {gi}: base T {base['T']:.1f}  dT removal {cells['removal']['dT']:+.1f} orth {cells['removal_orth']['dT']:+.1f} "
              f"random {cells['random']['dT']:+.1f}  attn last {cells['none']['attn_path_last']:.2f} -> {cells['removal']['attn_path_last']:.2f} / "
              f"{cells['removal_orth']['attn_path_last']:.2f}  probe margin K-1 {fm(cells['none']['margin_probe_Km1'])} -> "
              f"{fm(cells['removal']['margin_probe_Km1'])} / {fm(cells['removal_orth']['margin_probe_Km1'])}  "
              f"share {sh if sh is None else round(sh, 3)}", flush=True)
    return {"rows": rows, "summary": summarize(rows), "cells": cell_names,
            "chance_span_share": CHANCE_SPAN_SHARE, "n_directions": N_DIRECTIONS}


def summarize(rows):
    rs = [r for r in rows if not r.get("skipped")]
    out = {"n": len(rs), "n_skipped": len(rows) - len(rs),
           "cutoffs": {"flip_dT": FLIP_CUT, "baseline_correct_T": CORRECT_T},
           "score_sign": "higher score means more likely to flip; margins enter negated"}
    if not rs:
        return out
    flipped = [r["cells"]["removal"]["dT"] <= FLIP_CUT for r in rs]
    correct = [r["baseline"]["T"] > CORRECT_T for r in rs]
    unflipped = [not f for f in flipped]
    sel = lambda vals, mask: [v for v, m in zip(vals, mask) if m]  # noqa: E731
    out["flips"] = {}
    for cond in EDITS:
        f = [r["cells"][cond]["dT"] <= FLIP_CUT for r in rs]
        out["flips"][cond] = {"frac_flipped": bootstrap([float(x) for x in f], np.mean),
                              "frac_flipped_baseline_correct": bootstrap([float(x) for x in sel(f, correct)], np.mean),
                              "frac_escaped": bootstrap([float(r["cells"][cond]["e"] >= 0.5) for r in rs], np.mean),
                              "median_dT": bootstrap([r["cells"][cond]["dT"] for r in rs], np.median)}
    out["flips"]["n_baseline_wrong"] = int(sum(not c for c in correct))
    orth_f = [r["cells"]["removal_orth"]["dT"] <= FLIP_CUT for r in rs]
    out["flips"]["removal_orth_minus_removal"] = bootstrap([float(a) - float(b) for a, b in zip(orth_f, flipped)], np.mean)
    out["flips"]["removal_orth_and_removal_both"] = int(sum(a and b for a, b in zip(orth_f, flipped)))
    # attention onto the answer-path edge at the last search step
    att = {cond: [r["cells"][cond]["attn_path_last"] for r in rs] for cond in CONDITIONS}
    out["attention_last_step"] = {cond: bootstrap(att[cond], np.mean) for cond in CONDITIONS}
    drop = lambda cond: [a - b for a, b in zip(att["none"], att[cond])]  # noqa: E731
    out["attention_drop_last_step"] = {cond: bootstrap(drop(cond), np.mean) for cond in EDITS}
    d_rem, d_orth = drop("removal"), drop("removal_orth")
    out["attention_drop_orth_over_removal"] = bootstrap([o / m for o, m in zip(d_orth, d_rem) if abs(m) > 1e-6], np.median)
    km1 = lambda r: f"pass{r['K'] - 2}"  # noqa: E731
    predictors = {
        "baseline_p_decoy": [r["baseline"]["p_decoy"] for r in rs],
        "parent_out_degree": [r["cov"].get("parent_out_degree") for r in rs],
        "probe_share_in_removed_span_Km1": [r["span_shares"][km1(r)]["probe_in_removed_span"] for r in rs],
    }
    for ro in READOUTS:
        for cond in CONDITIONS:
            for label in ("Km1", "K"):
                predictors[f"neg_margin_{ro}_{label}_{cond}"] = [neg(r["cells"][cond][f"margin_{ro}_{label}"]) for r in rs]
    out["auc_removal_flip"] = {
        name: {"all": auc_boot(v, flipped), "baseline_correct": auc_boot(sel(v, correct), sel(flipped, correct))}
        for name, v in predictors.items()
    }
    for ro in READOUTS:
        for label in ("Km1", "K"):
            for cond in EDITS:
                d = [(r["cells"][cond][f"margin_{ro}_{label}"] - r["cells"]["none"][f"margin_{ro}_{label}"])
                     if r["cells"][cond][f"margin_{ro}_{label}"] is not None and r["cells"]["none"][f"margin_{ro}_{label}"] is not None else None
                     for r in rs]
                out[f"change_margin_{ro}_{label}_{cond}"] = {
                    "all": bootstrap(d, np.mean), "flipped": bootstrap(sel(d, flipped), np.mean),
                    "unflipped": bootstrap(sel(d, unflipped), np.mean),
                }
        out[f"margin_{ro}_Km1_unedited"] = {
            "flipped": bootstrap(sel([r["cells"]["none"][f"margin_{ro}_Km1"] for r in rs], flipped), np.mean),
            "unflipped": bootstrap(sel([r["cells"]["none"][f"margin_{ro}_Km1"] for r in rs], unflipped), np.mean),
        }
    for key in ("probe_in_removed_span", "probe_in_random_span", "embedding_in_removed_span", "embedding_in_random_span", "winner_in_orth_span"):
        v = [r["span_shares"][km1(r)][key] for r in rs]
        out[f"span_share_{key}_Km1"] = {"median": bootstrap(v, np.median), "mean": bootstrap(v, np.mean),
                                        "flipped_mean": bootstrap(sel(v, flipped), np.mean),
                                        "unflipped_mean": bootstrap(sel(v, unflipped), np.mean)}
    out["auc_answer_under_removal"] = {}
    for cond in ("removal", "removal_orth"):
        ans = [r["cells"][cond]["T"] > CORRECT_T for r in rs]
        for ro in READOUTS:
            out["auc_answer_under_removal"][f"margin_{ro}_K_{cond}"] = auc_boot([r["cells"][cond][f"margin_{ro}_K"] for r in rs], ans)
    return out


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    probes = load_probes(args.run_name)
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "winner_margin")
    result.update(run(runner, recips, probes, args.seed))
    finish(args, "winner_margin", result)


if __name__ == "__main__":
    main()
