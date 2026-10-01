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


def main():
    test, train = load_test(), load_train()
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
    result["rewrite_following_by_decoy_in_degree"] = {run: rewrite_split(run, test) for run in ("seed0", "seed1")}
    counts = []
    for gi in range(100):
        g = test[gi]
        counts.append(sum(1 for d in train if d["target"] == g["target"] and d["neg_target"] == g["neg_target"] and len(d["steps"]) == len(g["steps"])))
    result["same_answer_donors_available_per_recipient"] = {"min": int(min(counts)), "median": float(np.median(counts)), "max": int(max(counts)),
                                                            "n_zero": int(sum(c == 0 for c in counts)), "n_below_5": int(sum(c < 5 for c in counts))}
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
