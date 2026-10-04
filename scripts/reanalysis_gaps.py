"""Experiment 8, step 1: zero-cost re-analysis of the existing n=100 files.
Every number the gap review quotes is recomputed here from the per-graph rows
and written to results/gaps_reanalysis.json, so prose can point to a file.

Per seed (results/seed<s>/masking_n100.json, recovery_n100.json): graphs wrong
at baseline; flip rates of the removal, its random control, the same-answer
donor and the layer-2 mask over all graphs and over the baseline-correct ones;
where the removal's flips go; the per-graph overlap of the removal's flips
with the donor's and with the layer-2 mask's against independence; baseline
p_decoy on flipped and unflipped graphs and its AUC for each cell's flips;
the parent out-degree split; the both-candidates mask as the fraction with T
above 50; the K-1 donor thought's restoration per K. Across seeds: how often
the same test graph flips. Seeds 0 and 1 (counterfactuals_n100.json): the
last-hop rewrite's following split by the decoy's in-degree. If
results/seed<s>/masking_heldout.json exists (test graphs 100-399): the two
pre-registered predictors scored there, and nothing else.

    .venv/bin/python scripts/reanalysis_gaps.py
"""

import json
import sys
from itertools import product
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

from prompts import Prompt, covariates  # noqa: E402
from sets import load_test, load_train, test_pin  # noqa: E402
from stats import ESCAPE_CUT, FLIP_CUT, bootstrap  # noqa: E402
from winner_margin import auc_boot  # noqa: E402

RES = ROOT / "results"
SEEDS = ("seed0", "seed1", "seed2", "seed3")
CORRECT_T = 50.0  # protects "the baseline answer was the target"
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 99))
TEST = None  # the test split, set in main


def load(run, name):
    p = RES / run / name
    return json.load(open(p)) if p.exists() else None


def flips(rows, cell):
    out = []
    for r in rows:
        c = r["cells"].get(cell)
        out.append(None if c is None or c.get("skipped") or c.get("dT") is None else c["dT"] <= FLIP_CUT)
    return out


def frac(vals):
    v = [float(x) for x in vals if x is not None]
    return bootstrap(v, np.mean) if v else None


def hypergeom_tail(N, A, B, k):
    """P(X >= k) for the overlap of a fixed set of A and a random set of B in N."""
    return sum(comb(A, j) * comb(N - A, B - j) / comb(N, B) for j in range(k, min(A, B) + 1))


def overlap(fa, fb):
    idx = [i for i in range(len(fa)) if fa[i] is not None and fb[i] is not None]
    N, A, B = len(idx), sum(fa[i] for i in idx), sum(fb[i] for i in idx)
    both = sum(fa[i] and fb[i] for i in idx)
    return {"n": N, "a": int(A), "b": int(B), "both": int(both), "expected_if_independent": A * B / N if N else None,
            "p_at_least_both": hypergeom_tail(N, A, B, both) if N else None}


def seed_block(run, test):
    mk, rc = load(run, "masking_n100.json"), load(run, "recovery_n100.json")
    if mk is None:
        return None
    rows = mk["rows"]
    cov = {r["gi"]: covariates(Prompt.from_sample(test[r["gi"]], test_pin(r["gi"], mk["seed"]))) for r in rows}
    Tb = [r["baseline"]["T"] for r in rows]
    pd_ = [r["baseline"]["p_decoy"] for r in rows]
    correct = [t >= CORRECT_T for t in Tb]
    out = {"n": len(rows), "n_baseline_wrong": int(sum(not c for c in correct)),
           "n_without_same_answer_donor": int(sum(1 for r in rows if r["same_answer_donor_gi"] is None))}
    cells = {"removal": "qk_removal", "random_directions": "qk_random", "same_answer_donor_intermediates": "same_answer_donor/intermediates",
             "same_answer_donor_all": "same_answer_donor/all", "l2_mask": "l2_latents_mask/path/alone",
             "all_routes_plus_removal": "all_routes/path/plus_removal", "random_donor_intermediates": "random_donor/intermediates"}
    fl = {k: flips(rows, c) for k, c in cells.items()}
    out["flip_rates"] = {}
    for k, c in cells.items():
        f = fl[k]
        out["flip_rates"][k] = {"all": frac(f), "baseline_correct": frac([x for x, ok in zip(f, correct) if ok]),
                                "n_flipped": int(sum(1 for x in f if x)), "n": int(sum(1 for x in f if x is not None))}
    rem = [r["cells"]["qk_removal"] for r in rows]
    fr = [i for i, x in enumerate(fl["removal"]) if x]
    out["removal_where"] = {"n_flipped": len(fr), "escape": int(sum(1 for i in fr if rem[i]["e"] >= ESCAPE_CUT)),
                            "p_decoy_above_half": int(sum(1 for i in fr if rem[i]["p_decoy"] > 0.5)),
                            "mean_p_decoy_on_flipped": float(np.mean([rem[i]["p_decoy"] for i in fr])) if fr else None}
    out["overlap"] = {"removal_vs_same_answer_donor": overlap(fl["removal"], fl["same_answer_donor_intermediates"]),
                      "removal_vs_l2_mask": overlap(fl["removal"], fl["l2_mask"])}
    pdv = np.array(pd_)
    out["baseline_p_decoy"] = {}
    for k in ("removal", "same_answer_donor_intermediates", "l2_mask"):
        f = np.array([bool(x) for x in fl[k]])
        has = np.array([x is not None for x in fl[k]])
        ok = np.array(correct)
        out["baseline_p_decoy"][k] = {
            "median_flipped": float(np.median(pdv[f])) if f.any() else None,
            "median_unflipped": float(np.median(pdv[has & ~f])) if (has & ~f).any() else None,
            "median_unflipped_baseline_correct": float(np.median(pdv[has & ~f & ok])) if (has & ~f & ok).any() else None,
            "auc_for_flip": auc_boot([p if h else None for p, h in zip(pd_, has)], [bool(x) for x in fl[k]]),
            "auc_for_flip_baseline_correct": auc_boot([p for p, h, c in zip(pd_, has, correct) if h and c],
                                                      [bool(x) for x, h, c in zip(fl[k], has, correct) if h and c]),
        }
    pod = [cov[r["gi"]]["parent_out_degree"] for r in rows]
    out["parent_out_degree"] = {
        "auc_for_removal_flip": auc_boot(pod, [bool(x) for x in fl["removal"]]),
        "auc_for_removal_flip_baseline_correct": auc_boot([p for p, c in zip(pod, correct) if c], [bool(x) for x, c in zip(fl["removal"], correct) if c]),
        "auc_for_donor_flip": auc_boot([p if x is not None else None for p, x in zip(pod, fl["same_answer_donor_intermediates"])],
                                       [bool(x) for x in fl["same_answer_donor_intermediates"]]),
        "mean_flipped": float(np.mean([p for p, x in zip(pod, fl["removal"]) if x and p is not None])),
        "mean_unflipped": float(np.mean([p for p, x in zip(pod, fl["removal"]) if not x and p is not None])),
    }
    out["baseline_T_on_removal_flipped"] = {"median": float(np.median([Tb[i] for i in fr])) if fr else None,
                                            "min": float(min(Tb[i] for i in fr)) if fr else None}
    if rc is not None:
        rrows = rc["rows"]
        assert [r["gi"] for r in rrows] == [r["gi"] for r in rows]
        bc = {}
        for cell in ("allq_both_cand_edges/path/alone", "allq_both_cand_edges/path/plus_removal"):
            T = [r["cells"][cell]["T"] for r in rrows]
            bc[cell] = {"frac_T_above_50": frac([t > 50 for t in T]),
                        "frac_T_above_50_baseline_correct": frac([t > 50 for t, c in zip(T, correct) if c]),
                        "one_minus_frac_flipped": 1 - float(np.mean([r["cells"][cell]["dT"] <= FLIP_CUT for r in rrows])),
                        "mean_T": bootstrap(T, np.mean)}
        out["both_candidates_mask"] = bc
        rfl = flips(rrows, "qk_removal")
        assert rfl == fl["removal"]
        per_k = {}
        for cell in ("thoughtK/same_answer/plus_removal", "thoughtKm1/same_answer/plus_removal", "thoughtKm2/same_answer/plus_removal", "thoughtKm1/same_answer/alone"):
            entry = {}
            for K in (3, 4, "all"):
                sel = [r for r, f in zip(rrows, rfl) if f and (K == "all" or r["K"] == K) and r["cells"].get(cell) and not r["cells"][cell].get("skipped")]
                entry[str(K)] = {"restored": int(sum(1 for r in sel if r["cells"][cell]["T"] > 50)), "n_flipped": len(sel)}
            others = [r for r, f in zip(rrows, rfl) if f is False and r["cells"].get(cell) and not r["cells"][cell].get("skipped")]
            entry["flips_among_removal_correct"] = {"n": int(sum(1 for r in others if r["cells"][cell]["T"] <= 50)), "of": len(others)}
            per_k[cell] = entry
        out["donor_thought_restoration"] = per_k
    return out, fl["removal"], fl["same_answer_donor_intermediates"]


def recurrence(flip_sets):
    """How often the same test graph flips across seeds, against independence."""
    M = np.array([[bool(x) for x in f] for f in flip_sets])
    rates = M.mean(1)
    cnt = M.sum(0)

    def expected_at_least(k):
        p = 0.0
        for combo in product([0, 1], repeat=len(rates)):
            if sum(combo) >= k:
                p += float(np.prod([r if c else 1 - r for r, c in zip(rates, combo)]))
        return M.shape[1] * p

    return {"n_seeds": int(M.shape[0]), "graphs_by_number_of_seeds_flipped": [int((cnt == k).sum()) for k in range(M.shape[0] + 1)],
            "observed_at_least_3": int((cnt >= 3).sum()), "expected_at_least_3_if_independent": expected_at_least(3),
            "observed_all": int((cnt == M.shape[0]).sum()), "expected_all_if_independent": expected_at_least(M.shape[0]),
            "graphs_flipped_on_at_least_3_seeds": [int(i) for i in np.where(cnt >= 3)[0]]}


def rewrite_split(run, test):
    cf = load(run, "counterfactuals_n100.json")
    if cf is None:
        return None
    rows = [r for r in cf["rows"] if r["cells"].get("rewrite_last/free") and not r["cells"]["rewrite_last/free"].get("skipped")]
    din = [sum(1 for s, t in test[r["gi"]]["edges"] if t == test[r["gi"]]["neg_target"]) for r in rows]
    followed = [r["cells"]["rewrite_last/free"]["dT"] <= FLIP_CUT for r in rows]
    out = {"n": len(rows), "n_followed": int(sum(followed)), "by_decoy_in_degree": {}}
    for lo, hi in BUCKETS:
        sel = [f for f, d in zip(followed, din) if lo <= d <= hi]
        out["by_decoy_in_degree"][f"{lo}" if lo == hi else f"{lo}+"] = {"followed": int(sum(sel)), "n": len(sel), "frac": frac(sel)}
    out["auc_decoy_in_degree_for_not_followed"] = auc_boot(din, [not f for f in followed])
    for K in (3, 4):
        sel = [f for f, r in zip(followed, rows) if r["K"] == K]
        out[f"K{K}"] = {"followed": int(sum(sel)), "n": len(sel)}
    return out


def heldout_block(run, test):
    mk = load(run, "masking_heldout.json")
    if mk is None:
        return None
    rows = mk["rows"]
    correct = [r["baseline"]["T"] >= CORRECT_T for r in rows]
    rem, rnd, l2 = flips(rows, "qk_removal"), flips(rows, "qk_random"), flips(rows, "l2_latents_mask/path/alone")
    pod = [r["cov"].get("parent_out_degree") for r in rows]
    pd_ = [r["baseline"]["p_decoy"] for r in rows]
    sel = lambda v: [x for x, c in zip(v, correct) if c]  # noqa: E731
    return {
        "n": len(rows), "n_baseline_wrong": int(sum(not c for c in correct)),
        "removal_frac_flipped": frac(rem), "removal_frac_flipped_baseline_correct": frac(sel(rem)),
        "random_frac_flipped": frac(rnd), "l2_mask_frac_flipped": frac(l2),
        "overlap_removal_vs_l2_mask": overlap(rem, l2),
        "auc_baseline_p_decoy_for_removal_flip": {"all": auc_boot(pd_, [bool(x) for x in rem]),
                                                  "baseline_correct": auc_boot(sel(pd_), [bool(x) for x in sel(rem)])},
        "auc_parent_out_degree_for_removal_flip": {"all": auc_boot(pod, [bool(x) for x in rem]),
                                                   "baseline_correct": auc_boot(sel(pod), [bool(x) for x in sel(rem)])},
    }


SUBSTITUTES = ("zero/all", "zero/intermediates", "noise/all", "noise/intermediates", "average/all",
               "average/intermediates", "random_donor/intermediates", "same_answer_donor/intermediates", "removed")


def experiment9_block(run):
    """Experiment 9 zero-cost reads: baseline p_decoy for flips under the
    substitutes; last-step vs every-step removal; the decoy-edge mask on
    graphs the removal leaves correct; the restore test, if run."""
    out = {}
    ne = load(run, "necessity_n100.json")
    if ne is not None:
        rows = ne["rows"]
        ok = [r["baseline"]["T"] >= CORRECT_T for r in rows]
        pdv = [r["baseline"]["p_decoy"] for r in rows]
        out["substitutes_p_decoy_auc"] = {}
        for c in SUBSTITUTES:
            f = flips(rows, c)
            out["substitutes_p_decoy_auc"][c] = auc_boot([p if x is not None and o else None for p, x, o in zip(pdv, f, ok)],
                                                          [bool(x) for x in f])
    cf = load(run, "counterfactuals_n100.json")
    if cf is not None:
        out["last_step_vs_every_step"] = overlap(flips(cf["rows"], "qk_subtract/answer_edge/all_heads"),
                                                 flips(cf["rows"], "qk_every_step/answer_path/all_heads"))
    rc = load(run, "recovery_n100.json")
    if rc is not None:
        rows = rc["rows"]
        ok = [r["baseline"]["T"] >= CORRECT_T for r in rows]
        rem = flips(rows, "qk_removal")
        out["decoy_edges_among_removal_correct"] = {}
        for c in ("allq_decoy_edges/path/plus_removal", "allq_decoy_edges/ctrl/plus_removal", "decoy_edges/path/plus_removal"):
            x = flips(rows, c)
            keep = [i for i in range(len(rows)) if ok[i] and rem[i] is False and x[i] is not None]
            out["decoy_edges_among_removal_correct"][c] = {"n_flip": int(sum(x[i] for i in keep)), "of": len(keep),
                                                            "frac": frac([x[i] for i in keep])}
    rs = load(run, "restore_n100.json")
    if rs is not None:
        out["restore"] = rs["summary"]
        out["restore_rows"] = [{"gi": r["gi"], "removal_flipped": r["cells"]["removal"]["dT"] <= FLIP_CUT,
                                **{k: r["cells"][f"restore/{k}"]["dT"] <= FLIP_CUT for k in ("path", "offpath", "edges_all", "all")}}
                               for r in rs["rows"] if not r["cells"]["removal"].get("skipped")]
    return out


def pooled_restore(blocks):
    """Rescue shares pooled over seeds (each seed's removal-flipped graphs)."""
    rows = [r for b in blocks for r in b.get("restore_rows", []) if r["removal_flipped"]]
    if not rows:
        return None
    out = {"n_removal_flipped": len(rows)}
    for k in ("path", "offpath", "edges_all", "all"):
        out[k] = {"rescue_share": frac([not r[k] for r in rows]), "n_surviving": int(sum(r[k] for r in rows))}
    out["edges_all_minus_path"] = bootstrap([float(not r["edges_all"]) - float(not r["path"]) for r in rows], np.mean)
    out["all_minus_edges_all"] = bootstrap([float(not r["all"]) - float(not r["edges_all"]) for r in rows], np.mean)
    out["path_minus_offpath"] = bootstrap([float(not r["path"]) - float(not r["offpath"]) for r in rows], np.mean)
    return out


def experiment10_zero_cost(run):
    """The two reads experiment 10 cites before it ran: "target id below the
    decoy's" as a predictor of a wrong answer under 6b's both-candidates mask,
    and survival under 8c's path isolation split by id order."""
    out = {}
    rc = load(run, "recovery_n100.json")
    if rc is not None:
        out["auc_target_id_below_for_wrong"] = {}
        for c in ("allq_both_cand_edges/path/alone", "allq_both_cand_edges/path/plus_removal"):
            rows = [r for r in rc["rows"] if not r["cells"][c].get("skipped")]
            below = [float(r["cov"].get("target_id_below_decoy", TEST[r["gi"]]["target"] < TEST[r["gi"]]["neg_target"])) for r in rows]
            wrong = [r["cells"][c]["T"] <= 50 for r in rows]
            ok = [r["baseline"]["T"] >= CORRECT_T for r in rows]
            out["auc_target_id_below_for_wrong"][c] = {
                "all": auc_boot(below, wrong),
                "baseline_correct": auc_boot([b for b, o in zip(below, ok) if o], [w for w, o in zip(wrong, ok) if o])}
    mk = load(run, "masking_n100.json")
    if mk is not None:
        bc = [r for r in mk["rows"] if r["baseline"]["T"] >= CORRECT_T]
        below = lambda r: TEST[r["gi"]]["target"] < TEST[r["gi"]]["neg_target"]  # noqa: E731
        right = lambda r: r["cells"]["isolation/path/alone"]["T"] > 50  # noqa: E731
        out["isolation_path_survival_by_id_order"] = {
            "all_baseline_correct": frac([right(r) for r in bc]),
            "target_id_above": frac([right(r) for r in bc if not below(r)]),
            "target_id_below": frac([right(r) for r in bc if below(r)]),
            "n_above": sum(1 for r in bc if not below(r)), "n_below": sum(1 for r in bc if below(r))}
    return out


E10_ISO, E10_CAND, E10_OFF = "isolation/path/alone", "isolation/path_cand/alone", "isolation/offpath_matched/alone"


def e10_units(run):
    """One record per baseline-correct graph of masking_leftover_n100.json."""
    ml = load(run, "masking_leftover_n100.json")
    if ml is None:
        return None, None
    mk = load(run, "masking_n100.json")
    old = {r["gi"]: r["cells"]["isolation/path/alone"]["T"] for r in mk["rows"]}
    rerun_diff = max(abs(old[r["gi"]] - r["cells"][E10_ISO]["T"]) for r in ml["rows"])
    units = []
    for r in ml["rows"]:
        if r["baseline"]["T"] < CORRECT_T:
            continue
        c = r["cells"]
        units.append({"run": run, "gi": r["gi"], "K": r["K"], "below": bool(r["cov"]["target_id_below_decoy"]),
                      "target_first": bool(r["cov"]["target_first"]),
                      "decoy_only": not r["meta"]["target_extra_in_edges"],
                      "iso": c[E10_ISO]["T"] > 50, "cand": c[E10_CAND]["T"] > 50, "off": c[E10_OFF]["T"] > 50,
                      "e": {k: c[n]["e"] for k, n in (("iso", E10_ISO), ("cand", E10_CAND), ("off", E10_OFF))},
                      "T": {k: c[n]["T"] for k, n in (("iso", E10_ISO), ("cand", E10_CAND))}})
    meta = {"rerun_max_abs_dT_vs_masking_n100": rerun_diff,
            "invariance_max_abs_logit_diff": max(r["meta"]["invariance_max_abs_logit_diff"] for r in ml["rows"]),
            "matched_shortfall_graphs": sum(1 for r in ml["rows"] if r["meta"]["matched_shortfall"] > 0),
            "T_distribution_all_graphs": {k: t_distribution([r["cells"][n]["T"] for r in ml["rows"]]) for k, n in (("isolation_path", E10_ISO), ("path_cand", E10_CAND))},
            "n_graphs": len(ml["rows"])}
    return units, meta


def t_distribution(T):
    T = np.asarray(T, float)
    return {"n": int(len(T)), "median_T": bootstrap(T, np.median), "mean_T": bootstrap(T, np.mean),
            "frac_T_between_20_and_80": frac([20 <= t <= 80 for t in T]),
            "histogram_20pt_bins": {f"{lo}-{lo + 20}": int(((T >= lo) & ((T < lo + 20) if lo < 80 else (T <= 100))).sum()) for lo in range(0, 100, 20)}}


def e10_stats(u):
    """Survivals, the decomposition E + N + U = 1, G and the id-order gaps on
    one set of units (any field may be None where a group is empty)."""
    a = lambda k, sel=lambda x: True: (lambda v: float(np.mean(v)) if v else None)([x[k] for x in u if sel(x)])  # noqa: E731
    s_iso, s_1, s_off = a("iso"), a("cand"), a("off")
    s_a, s_b = a("cand", lambda x: not x["below"]), a("cand", lambda x: x["below"])
    i_a, i_b = a("iso", lambda x: not x["below"]), a("iso", lambda x: x["below"])
    out = {"s_iso": s_iso, "s_1": s_1, "s_offpath_matched": s_off, "s_above": s_a, "s_below": s_b,
           "G": float(np.mean([x["cand"] == (not x["below"]) for x in u])) if u else None,
           "gap_cell1": None if s_a is None or s_b is None else s_a - s_b,
           "gap_isolation_path": None if i_a is None or i_b is None else i_a - i_b,
           "s1_minus_s_iso": None if s_1 is None else s_1 - s_iso}
    left = None if s_iso is None else s_iso - 0.5
    if left and left > 0:
        out["E"] = (s_iso - s_1) / left
        if s_a is not None and s_b is not None:
            s_bal = (s_a + s_b) / 2
            out["N"], out["U"] = (s_1 - s_bal) / left, (s_bal - 0.5) / left
    return out


def e10_block(u, n_boot=2000, seed=0):
    """Point values and paired bootstrap intervals over graphs."""
    point = e10_stats(u)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        st = e10_stats([u[i] for i in rng.integers(0, len(u), len(u))])
        for k in draws:
            if st.get(k) is not None:
                draws[k].append(st[k])
    out = {}
    for k, v in point.items():
        out[k] = {"point": v, "lo": float(np.percentile(draws[k], 2.5)) if draws[k] else None,
                  "hi": float(np.percentile(draws[k], 97.5)) if draws[k] else None}
    wrong = [x for x in u if not x["cand"]]
    out["n_baseline_correct"] = len(u)
    out["n_target_id_below"] = sum(x["below"] for x in u)
    for k in ("iso", "cand", "off"):
        w = [x for x in u if not x[k]]
        out[f"wrong_{k}"] = {"n": len(w), "escape": sum(x["e"][k] >= ESCAPE_CUT for x in w), "switch": sum(x["e"][k] < ESCAPE_CUT for x in w)}
    out["escape_share_cell1"] = frac([(not x["cand"]) and x["e"]["cand"] >= ESCAPE_CUT for x in u])
    out["cell1_by_target_first"] = {"target_first": frac([x["cand"] for x in u if x["target_first"]]),
                                    "target_second": frac([x["cand"] for x in u if not x["target_first"]])}
    out["cell1_T_distribution_baseline_correct"] = t_distribution([x["T"]["cand"] for x in u])
    out["n_cell1_wrong"] = len(wrong)
    out["cell1_lost"] = sum(x["iso"] and not x["cand"] for x in u)  # right under 8c isolation, wrong under cell 1
    out["cell1_gained"] = sum((not x["iso"]) and x["cand"] for x in u)
    return out


def experiment10():
    out, allu = {"seeds": {}}, []
    for run in SEEDS:
        u, meta = e10_units(run)
        if u is None:
            continue
        allu += u
        out["seeds"][run] = {"checks": meta, "all": e10_block(u),
                             "decoy_only_graphs": {"n": sum(x["decoy_only"] for x in u)}}
    if allu:
        out["pooled"] = {"all": e10_block(allu), "decoy_only_graphs": e10_block([x for x in allu if x["decoy_only"]]),
                         "target_extra_in_edge_graphs": e10_block([x for x in allu if not x["decoy_only"]])}
        for run in out["seeds"]:
            out["seeds"][run]["decoy_only_graphs"] = e10_block([x for x in allu if x["run"] == run and x["decoy_only"]])
    return out


def main():
    global TEST
    test, train = load_test(), load_train()
    TEST = test
    result = {"cutoffs": {"flip_dT": FLIP_CUT, "escape_e": ESCAPE_CUT, "baseline_correct_T": CORRECT_T}, "seeds": {}, "heldout": {}}
    rem_sets, don_sets = [], []
    for run in SEEDS:
        blk = seed_block(run, test)
        if blk is None:
            continue
        result["seeds"][run], rem, don = blk
        rem_sets.append(rem)
        don_sets.append(don)
        ho = heldout_block(run, test)
        if ho is not None:
            result["heldout"][run] = ho
    result["recurrence_across_seeds"] = {"removal": recurrence(rem_sets), "same_answer_donor_intermediates": recurrence(don_sets)}
    result["experiment9"] = {run: experiment9_block(run) for run in SEEDS}
    result["experiment9_restore_pooled"] = pooled_restore(list(result["experiment9"].values()))
    result["rewrite_following_by_decoy_in_degree"] = {run: rewrite_split(run, test) for run in ("seed0", "seed1")}
    counts = []
    for gi in range(100):
        g = test[gi]
        counts.append(sum(1 for d in train if d["target"] == g["target"] and d["neg_target"] == g["neg_target"] and len(d["steps"]) == len(g["steps"])))
    result["same_answer_donors_available_per_recipient"] = {"min": int(min(counts)), "median": float(np.median(counts)), "max": int(max(counts)),
                                                            "n_zero": int(sum(c == 0 for c in counts)), "n_below_5": int(sum(c < 5 for c in counts))}
    result["experiment10_zero_cost"] = {run: experiment10_zero_cost(run) for run in SEEDS}
    result["experiment10"] = experiment10()
    out = RES / "gaps_reanalysis.json"
    json.dump(result, open(out, "w"), indent=1)
    print(f"written: {out}")
    for run, b in result["seeds"].items():
        fr = b["flip_rates"]
        print(f"{run}: wrong at baseline {b['n_baseline_wrong']}; removal flips {fr['removal']['n_flipped']} "
              f"(over correct {fr['removal']['baseline_correct']['point']:.2f}); random {fr['random_directions']['n_flipped']}; "
              f"donor {fr['same_answer_donor_intermediates']['n_flipped']}; overlap removal/donor {b['overlap']['removal_vs_same_answer_donor']['both']} "
              f"(expected {b['overlap']['removal_vs_same_answer_donor']['expected_if_independent']:.1f}); "
              f"p_decoy AUC for removal flip {b['baseline_p_decoy']['removal']['auc_for_flip']['point']:.2f}, for donor flip "
              f"{b['baseline_p_decoy']['same_answer_donor_intermediates']['auc_for_flip']['point']:.2f}; parent out-degree AUC "
              f"{b['parent_out_degree']['auc_for_removal_flip']['point']:.2f}; flips escaping {b['removal_where']['escape']}")
    rc = result["recurrence_across_seeds"]["removal"]
    print(f"removal flips on at least 3 seeds: {rc['observed_at_least_3']} (expected {rc['expected_at_least_3_if_independent']:.1f})")
    for run, ho in result["heldout"].items():
        print(f"heldout {run}: removal flips {ho['removal_frac_flipped']['point']:.2f}, random {ho['random_frac_flipped']['point']:.2f}, "
              f"p_decoy AUC {ho['auc_baseline_p_decoy_for_removal_flip']['baseline_correct']['point']:.2f}, "
              f"parent out-degree AUC {ho['auc_parent_out_degree_for_removal_flip']['baseline_correct']['point']:.2f}")


if __name__ == "__main__":
    main()
