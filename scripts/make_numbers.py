#!/usr/bin/env python3
"""Emit paper/numbers.tex from result files.

Every statistic quoted in a write-up is a LaTeX macro whose value is read from
a results/ JSON or JSONL file, never hand-typed. Register macros in build().

Naming. Macros kept from the v1 draft (its make_numbers.py, RRR repository)
keep their names and are read from results/seed0/; each has a seed-1 twin
with the suffix SeedB (the v1 convention: A, B, C, D are seeds 0, 1, 2, 3).
New macros end in SeedA, SeedB, SeedC, SeedD, Pooled, Dilgren or Aswal; an
interval is the same name with Lo or Hi before that suffix. Digits are not
allowed in macro names, so steps and passes are spelled out (StepOne, PassSix).

Some values are derived from per-graph rows (paired differences, restoration
counts); each such producer names the file whose rows it reads. Result files
are only read.

Missing fields are emitted as visible ??MISSING?? placeholders and reported
(exit 1), so a stale result file can never silently produce a wrong number.

    python scripts/make_numbers.py [--repo PATH] [--out PATH] [--table PATH]

Defaults: repo = parent of this script's directory; out = <repo>/paper/numbers.tex;
table = <repo>/paper/numbers_table.md (macro, value, source file).
"""

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

import numpy as np

SEEDS = {"seed0": "SeedA", "seed1": "SeedB", "seed2": "SeedC", "seed3": "SeedD"}
GPT2 = {"gpt2_dilgren": "Dilgren", "gpt2_aswal": "Aswal"}
WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
FLIP_CUT, ESC_CUT, CORRECT_T = -50.0, 0.5, 50.0  # as in src/phoenix/stats.py and reanalysis_gaps.py
N_BOOT = 2000


def load(repo: Path, rel: str):
    p = repo / rel
    if not p.exists() and (repo / (rel + ".gz")).exists():
        return json.load(gzip.open(repo / (rel + ".gz"), "rt"))
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.open()]
    return json.load(p.open())


def fmt(x, nd=None, pct=False):
    if pct:
        x = x * 100.0
    if nd is not None:
        # fixed decimals, so 0.90 stays 0.90; a negative value that rounds to
        # zero keeps its sign (-0.0), as in the v1 draft
        return f"{float(x):.{nd}f}"
    if isinstance(x, float) and x == 0:
        return "-0.0" if str(x).startswith("-") else "0.0"
    return f"{x:g}"


def ensure_math(s: str) -> str:
    return "\\ensuremath{%s}" % s if any(c in s for c in "-.0123456789") else s


def boot_mean(vals, seed=0):
    """Mean with a 95 percent bootstrap interval over graphs (src/phoenix/stats.bootstrap)."""
    x = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"point": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def flipped(cell):
    return cell["dT"] <= FLIP_CUT


def present(r, name):
    c = r["cells"].get(name)
    return c is not None and not c.get("skipped")


def paired_flips(rows, a, b):
    """Flip rate of cell a minus cell b, paired per graph (graphs with both)."""
    return boot_mean([float(flipped(r["cells"][a])) - float(flipped(r["cells"][b]))
                      for r in rows if present(r, a) and present(r, b)])


def build(repo: Path):
    """Return {macro_name: (producer, is_text, source)}. producer() reads from
    the loaded result files and returns a string; is_text=True skips math wrapping."""
    cache = {}

    def F(rel):
        if rel not in cache:
            cache[rel] = load(repo, rel)
        return cache[rel]

    N = {}

    def macro(name, producer, src, text=False):
        assert name not in N, name
        assert name.isalpha(), name
        N[name] = (producer, text, src)

    def trio(base, suf, getter, src, nd=2, pct=False):
        """name, nameLo, nameHi from a {point, lo, hi} dict."""
        macro(base + suf, lambda: fmt(getter()["point"], nd, pct), src)
        macro(base + "Lo" + suf, lambda: fmt(getter()["lo"], nd, pct), src)
        macro(base + "Hi" + suf, lambda: fmt(getter()["hi"], nd, pct), src)

    def S(run, name):
        return F(f"results/{run}/{name}")["summary"]

    # ==== 1. Names kept from the v1 draft (seed 0) with seed-1 twins =========
    # per-seed accuracy: accA..accD are seeds 0..3 (serialization seed 0)
    acc = {run: f"results/{run}/evaluation_ser0.json" for run in SEEDS}
    for run, letter in zip(SEEDS, "ABCD"):
        macro(f"acc{letter}", lambda r=run: fmt(F(acc[r])["test_accuracy"], 1, pct=True), acc[run])
    macro("accbandlo", lambda: fmt(min(F(acc[r])["test_accuracy"] for r in SEEDS), 1, pct=True), "results/seed{0,1,2,3}/evaluation_ser0.json")
    macro("accbandhi", lambda: fmt(max(F(acc[r])["test_accuracy"] for r in SEEDS), 1, pct=True), "results/seed{0,1,2,3}/evaluation_ser0.json")

    def kept(name, rel_of_seed, producer, text=False):
        """name from seed 0, nameSeedB from seed 1; producer(data) -> str."""
        for run, suf in (("seed0", ""), ("seed1", "SeedB")):
            rel = rel_of_seed(run)
            macro(name + suf, lambda rel=rel: producer(F(rel)), rel, text)

    probe = lambda run: f"results/{run}/probe_basis_report.json"  # noqa: E731
    kept("aucFrontier", probe, lambda d: fmt(d["median_holdout_auc"], 4))
    kept("aucFrontierMin", probe, lambda d: fmt(d["min_holdout_auc"], 2))
    kept("nProbedNodes", probe, lambda d: fmt(d["n_nodes_probed"]))

    tr = lambda run: f"results/{run}/baseline_transplant_n100.json"  # noqa: E731
    kept("nEval", tr, lambda d: fmt(len(d["rows"])))
    cell = lambda c, k="median_dT", nd=1: (lambda d: fmt(d["summary"][c][k]["point"], nd))  # noqa: E731
    kept("dTbaseReserial", tr, cell("reserialized"))
    kept("dTselfT", tr, cell("self_transplant"))
    kept("dTdonorDiff", tr, cell("matched_donor/intermediates"))
    kept("escDonorDiff", tr, cell("matched_donor/intermediates", "median_e", 4))
    kept("dTsingleStep", tr, cell("matched_donor/first"))
    kept("escSingleStep", tr, cell("matched_donor/first", "median_e", 4))
    kept("dTdonorFinal", tr, cell("matched_donor/final"))
    kept("dTdonorAll", tr, cell("matched_donor/all"))
    kept("dTrandomDonor", tr, cell("random_donor/intermediates"))
    kept("escRandomDonor", tr, cell("random_donor/intermediates", "median_e", 2))
    kept("dTplaceboM", tr, cell("placebo_swap/intermediates"))
    kept("dTinteriorM", tr, cell("interior_swap/intermediates"))
    kept("dTinterior", tr, cell("interior_swap/intermediates"))
    # label swap: its v1 names (gateAdT, dTswapDonor, gateAfrac) are on the drop list, so new names
    kept("dTlabelSwap", tr, cell("label_swap/intermediates"))
    kept("fracLabelSwap", tr, cell("label_swap/intermediates", "frac_flipped", 2))

    sw = lambda run: f"results/{run}/baseline_swap_n100.json"  # noqa: E731
    kept("jswapFinalDT", sw, cell("swap_final", nd=2))
    kept("jswapFinalEsc", sw, cell("swap_final", "median_e", 4))

    jl = lambda run: f"results/{run}/jlens_basis_report.json"  # noqa: E731
    kept("jconcMid", jl, lambda d: fmt(d["per_step"]["0"]["median_concentration"], 2))
    kept("jconcFinal", jl, lambda d: fmt(d["per_step"]["3"]["median_concentration"], 2))

    # every recycled thought zeroed (necessity zero/all on test graphs 0-99)
    nec = lambda run: f"results/{run}/necessity_n100.json"  # noqa: E731
    kept("dTzeroHeld", nec, cell("zero/all"))
    kept("escZero", nec, cell("zero/all", "median_e", 2))
    macro("dTzeroSeedB", lambda: fmt(S("seed1", "necessity_n100.json")["zero/all"]["median_dT"]["point"], 1), nec("seed1"))

    # INLP, amnesic probing (experiment 11): the v1 recipe on this repo's seeds 0 and 1
    ia = lambda run: f"results/{run}/inlp_arm_n100.json"  # noqa: E731
    ir = lambda run: f"results/{run}/inlp_basis_report_full.json"  # noqa: E731
    kept("inlpDT", ia, cell("subtract_answer/inlp", nd=2))
    kept("inlpTpost", ia, cell("subtract_answer/inlp", "median_T", 2))
    kept("inlpRandDT", ia, cell("subtract_answer/inlp_random_matched", nd=2))
    kept("inlpMedianK", ir, lambda d: fmt(d["median_k"]))
    kept("inlpKmax", ir, lambda d: fmt(d["recipe"]["k_max"]))
    kept("inlpAUCstop", ir, lambda d: fmt(d["recipe"]["auc_stop"], 2))

    def resid_range(d):
        last = [v["auc_trajectory"][-1] for v in d["per_node"].values() if v.get("auc_trajectory")]
        return "%s--%s" % (fmt(min(last), 2), fmt(max(last), 2))
    kept("inlpAUCresid", ir, resid_range, text=True)

    # whole answer-path deletion at every intermediate step (experiment 12), test graphs 0-99
    pd_ = lambda run: f"results/{run}/path_deletion_n100.json"  # noqa: E731
    kept("dTpathNolastEmb", pd_, cell("path_mid/input_embedding", nd=2))
    kept("dTpathNolastProbe", pd_, cell("path_mid/probe", nd=2))

    for run, suf in (("seed0", "SeedA"), ("seed1", "SeedB")):
        src = ia(run)
        sm = lambda src=src: F(src)["summary"]  # noqa: E731
        macro(f"inlpN{suf}", lambda src=src: fmt(len(F(src)["rows"])), src)
        for c, nm in (("subtract_answer/inlp", "Inlp"), ("subtract_answer/inlp_random_matched", "InlpRand"),
                      ("subtract_answer/inlp_random_sizematched", "InlpSize"),
                      ("subtract_answer/inlp_first", "InlpFirst"), ("subtract_answer/probe", "InlpProbe")):
            trio(f"amn{nm}Flips", suf, lambda sm=sm, c=c: sm()[c]["frac_flipped"], src)
            trio(f"amn{nm}Beyond", suf, lambda sm=sm, c=c: sm()[c]["redirection"]["flips_beyond_reference"], src)
            trio(f"amn{nm}Esc", suf, lambda sm=sm, c=c: sm()[c]["frac_escaped"], src)
            macro(f"amn{nm}DT{suf}", lambda sm=sm, c=c: fmt(sm()[c]["median_dT"]["point"], 2), src)
        trio("amnSameAnswerFlips", suf, lambda sm=sm: sm()["same_answer_donor/intermediates"]["frac_flipped"], src)
        trio("amnInlpVsRand", suf, lambda src=src: paired_flips(F(src)["rows"], "subtract_answer/inlp", "subtract_answer/inlp_random_matched"), src)
        trio("amnInlpVsSize", suf, lambda src=src: paired_flips(F(src)["rows"], "subtract_answer/inlp", "subtract_answer/inlp_random_sizematched"), src)
        trio("amnSizeVsRand", suf, lambda src=src: paired_flips(F(src)["rows"], "subtract_answer/inlp_random_sizematched", "subtract_answer/inlp_random_matched"), src)
        macro(f"amnShareMismatchMax{suf}", lambda src=src: "%.0e" % max(abs(r["removed_share"]["inlp"] - r["removed_share"]["random_sizematched"]) for r in F(src)["rows"]), src, text=True)
        for k, nm in (("inlp", "Inlp"), ("random_matched", "Rand"), ("random_sizematched", "Size"), ("probe", "Probe")):
            macro(f"amnRemovedShare{nm}{suf}", lambda src=src, k=k: fmt(statistics.median(r["removed_share"][k] for r in F(src)["rows"]), 2), src)
        src = pd_(run)
        sm = lambda src=src: F(src)["summary"]  # noqa: E731
        macro(f"pathNolastRandDT{suf}", lambda sm=sm: fmt(sm()["path_mid/random_matched"]["median_dT"]["point"], 2), src)
        for c, nm in (("path_mid/input_embedding", "Emb"), ("path_mid/probe", "Probe"), ("path_mid/random_matched", "Rand")):
            trio(f"pathNolast{nm}Flips", suf, lambda sm=sm, c=c: sm()[c]["frac_flipped"], src)
            trio(f"pathNolast{nm}Beyond", suf, lambda sm=sm, c=c: sm()[c]["redirection"]["flips_beyond_reference"], src)
            trio(f"pathNolast{nm}Esc", suf, lambda sm=sm, c=c: sm()[c]["frac_escaped"], src)
        trio("pathNolastSameAnswerFlips", suf, lambda sm=sm: sm()["same_answer_donor/intermediates"]["frac_flipped"], src)
        for c, nm in (("path_mid/input_embedding", "Emb"), ("path_mid/probe", "Probe")):
            trio(f"pathNolast{nm}VsRand", suf, lambda src=src, c=c: paired_flips(F(src)["rows"], c, "path_mid/random_matched"), src)
        # the label-swap donors on their own prompts: T for the swapped answer
        # (the candidate swap keeps the recipient's target, so it is 100 - T)
        src = tr(run)
        sw_t = lambda src=src: [100 - r["info"]["label_swap"]["donor_T_on_own_prompt"]  # noqa: E731
                                for r in F(src)["rows"] if "label_swap" in r["info"]]
        macro(f"labelSwapDonorT{suf}", lambda f=sw_t: fmt(statistics.median(f()), 1), src)
        macro(f"labelSwapDonorFrac{suf}", lambda f=sw_t: fmt(sum(t > 50 for t in f()) / len(f()), 0, pct=True) + r"\%", src, text=True)
        macro(f"labelSwapDonorN{suf}", lambda f=sw_t: fmt(len(f())), src)

    # ==== 3. Experiments 0 to 10, per seed and pooled ========================
    for run, suf in SEEDS.items():
        R = f"results/{run}/"

        def have(name, R=R):
            return (repo / R / name).exists()

        # -- experiment 0: subtraction on natural test graphs (seeds 0, 1)
        if have("baseline_subtraction_n100.json"):
            src = R + "baseline_subtraction_n100.json"
            for c, nm in (("subtract_answer/input_embedding", "Emb"), ("subtract_answer/probe", "Probe"),
                          ("subtract_answer/random_matched", "Rand"), ("subtract_sibling/input_embedding", "Sibling")):
                macro(f"subDT{nm}{suf}", lambda src=src, c=c: fmt(F(src)["summary"][c]["median_dT"]["point"], 2), src)
            macro(f"subN{suf}", lambda src=src: fmt(F(src)["summary"]["subtract_answer/input_embedding"]["n"]), src)
        if have("baseline_transplant_n100.json"):
            src = R + "baseline_transplant_n100.json"
            for c, nm in (("matched_donor/intermediates", "Matched"), ("matched_donor/final", "Final"), ("matched_donor/first", "First")):
                trio(f"transBeyondRef{nm}", suf, lambda src=src, c=c: F(src)["summary"][c]["redirection"]["flips_beyond_reference"], src)
            trio("transSameAnswerFlips", suf, lambda src=src: F(src)["summary"]["same_answer_donor/intermediates"]["frac_flipped"], src)

        # -- experiment 1: necessity
        src = R + "necessity_n100.json"
        for c, nm in (("zero/all", "ZeroAll"), ("zero/intermediates", "ZeroMid"), ("noise/all", "NoiseAll"),
                      ("noise/intermediates", "NoiseMid"), ("average/all", "AverageAll"), ("average/intermediates", "AverageMid"),
                      ("random_donor/all", "RandomAll"), ("random_donor/intermediates", "RandomMid"),
                      ("removed", "Removed"), ("removed_length_kept", "Pad"),
                      ("same_answer_donor/all", "SameAnswerAll"), ("same_answer_donor/intermediates", "SameAnswerMid")):
            trio(f"nec{nm}DT", suf, lambda src=src, c=c: F(src)["summary"][c]["median_dT"], src, 1)
            macro(f"nec{nm}Flips{suf}", lambda src=src, c=c: fmt(F(src)["summary"][c]["frac_flipped"]["point"], 2), src)
            macro(f"nec{nm}Esc{suf}", lambda src=src, c=c: fmt(F(src)["summary"][c]["frac_escaped"]["point"], 2), src)
            if c not in ("same_answer_donor/all", "same_answer_donor/intermediates", "removed", "removed_length_kept"):
                trio(f"nec{nm}Beyond", suf, lambda src=src, c=c: F(src)["summary"][c]["redirection"]["flips_beyond_reference"], src)
        for nm in ("Removed",):
            macro(f"nec{nm}PDecoy{suf}", lambda src=src: fmt(F(src)["summary"]["removed"]["mean_p_decoy"]["point"], 2), src)

        # -- experiment 2: heads, per layer, at the intermediate latents
        src = R + "heads_n100.json"
        for layer in (1, 2):
            def heads_at(layer=layer, src=src, cut=0.5):  # experiment 5's cutoff: protects "this head reads edges"
                Hs = F(src)["summary"]
                es = [Hs[f"L{layer}H{h}/intermediate_latent"] for h in range(8)]
                return [e for e in es if e["slot_mass_a"] and e["slot_mass_a"]["point"] >= cut]
            L = "LOne" if layer == 1 else "LTwo"
            macro(f"heads{L}Readers{suf}", lambda f=heads_at: fmt(len(f())), src)
            for q, nm in (("content_score", "Content"), ("position_score", "Position"), ("slot_mass_a", "Mass")):
                macro(f"heads{L}{nm}Min{suf}", lambda f=heads_at, q=q: fmt(min(e[q]["point"] for e in f()), 2), src)
                macro(f"heads{L}{nm}Max{suf}", lambda f=heads_at, q=q: fmt(max(e[q]["point"] for e in f()), 2), src)
        # Zhu et al.'s layer-1 copy: share of attention from the separator inside its own slot,
        # over the layer-1 heads that copy (at least 0.5 inside the slot)
        def copiers(src=src):
            Hs = F(src)["summary"]
            return [Hs[f"L1H{h}/edge_sep"]["own_slot_a"]["point"] for h in range(8) if Hs[f"L1H{h}/edge_sep"]["own_slot_a"]["point"] >= 0.5]
        macro(f"headsSepCopyN{suf}", lambda f=copiers: fmt(len(f())), src)
        macro(f"headsSepCopyMin{suf}", lambda f=copiers: fmt(min(f()), 2), src)
        macro(f"headsSepCopyMax{suf}", lambda f=copiers: fmt(max(f()), 2), src)

        # -- experiment 3: counterfactual cells and the query-key removal (seeds 0, 1)
        if have("counterfactuals_n100.json"):
            src = R + "counterfactuals_n100.json"
            cs = lambda src=src: F(src)["summary"]  # noqa: E731
            for c, nm in (("qk_subtract/answer_edge/all_heads", "Last"), ("qk_every_step/answer_path/all_heads", "Every")):
                macro(f"qk{nm}AttnBefore{suf}", lambda cs=cs, c=c: fmt(cs()["qk_attention"][c]["attn_answer_total_before"]["point"], 2), src)
                macro(f"qk{nm}AttnAfter{suf}", lambda cs=cs, c=c: fmt(cs()["qk_attention"][c]["attn_answer_total_after"]["point"], 2), src)
                trio(f"qk{nm}AttnDrop", suf, lambda cs=cs, c=c: cs()["qk_attention"][c]["attn_answer_total_drop"], src)
                trio(f"qk{nm}DT", suf, lambda cs=cs, c=c: cs()[c]["median_dT"], src, 1)
                trio(f"qk{nm}Flips", suf, lambda cs=cs, c=c: cs()[c]["frac_flipped"], src)
                trio(f"qk{nm}Beyond", suf, lambda cs=cs, c=c: cs()[c]["redirection"]["flips_beyond_reference"], src)
            macro(f"qkEveryPathDropMean{suf}", lambda cs=cs: fmt(cs()["qk_attention"]["qk_every_step/answer_path/all_heads"]["attn_path_drop_mean_over_steps"]["point"], 2), src)
            macro(f"qkLastRandAttnAfter{suf}", lambda cs=cs: fmt(cs()["qk_attention"]["qk_subtract/random_matched/all_heads"]["attn_answer_total_after"]["point"], 2), src)
            macro(f"qkLastRandFlips{suf}", lambda cs=cs: fmt(cs()["qk_subtract/random_matched/all_heads"]["frac_flipped"]["point"], 2), src)
            for c, nm in (("reordered/intermediates", "ReorderedMid"), ("reordered/all", "ReorderedAll"),
                          ("decoy_swap/all", "DecoySwapAll"), ("noncandidate_swap/all", "NoncandSwapAll"),
                          ("rewrite_last/all", "RewriteLastAll"), ("rewrite_last/free", "RewriteLastFree"),
                          ("rewrite_last/intermediates", "RewriteLastMid")):
                macro(f"cf{nm}Flips{suf}", lambda cs=cs, c=c: fmt(cs()[c]["frac_flipped"]["point"], 2), src)
                macro(f"cf{nm}N{suf}", lambda cs=cs, c=c: fmt(cs()[c]["n"]), src)
            macro(f"cfReorderedMidDT{suf}", lambda cs=cs: fmt(cs()["reordered/intermediates"]["median_dT"]["point"], 1), src)

        # -- experiment 4a: causal tracing (seeds 0, 1)
        if have("tracing_n100.json"):
            src = R + "tracing_n100.json"
            for cf_, cn in (("candidate_swap", "Swap"), ("rewrite_last", "Rewrite")):
                for cls, lvl, nm in (("latent_final", "level_0", "FinalLatentLZero"), ("latent_final", "level_1", "FinalLatentLOne"),
                                     ("latent_intermediate", "level_2", "MidLatentLTwo"), ("latent_intermediate", "level_0", "MidLatentLZero"),
                                     ("slot_changed_target", "level_0", "TargetTokLZero"), ("slot_changed_sep", "level_1", "SepLOne"),
                                     ("slot_changed_source", "level_0", "SourceLZero")):
                    trio(f"trace{cn}{nm}", suf, lambda src=src, cf_=cf_, cls=cls, lvl=lvl: F(src)["summary"][cf_][cls][lvl], src)

        # -- experiment 4b: cache patching (seeds 0, 1)
        if have("cache_patch_n100.json"):
            src = R + "cache_patch_n100.json"
            for c, nm in (("donor/edges/k/L2", "KeysLTwo"), ("donor/edges/v/L2", "ValuesLTwo"), ("donor/edges/kv/both", "KVBoth"),
                          ("random_graph/latents_all/v/L1", "RandLatentsVLOne"), ("random_graph/edges/kv/both", "RandEdgesKV")):
                macro(f"cache{nm}DT{suf}", lambda src=src, c=c: fmt(F(src)["summary"][c]["median_dT"]["point"], 1), src)
                macro(f"cache{nm}Flips{suf}", lambda src=src, c=c: fmt(F(src)["summary"][c]["frac_flipped"]["point"], 2), src)
                macro(f"cache{nm}Esc{suf}", lambda src=src, c=c: fmt(F(src)["summary"][c]["frac_escaped"]["point"], 2), src)
                trio(f"cache{nm}Beyond", suf, lambda src=src, c=c: F(src)["summary"][c]["redirection"]["flips_beyond_reference"], src)

        # -- experiments 5 and 8c: masks (all seeds)
        src = R + "masking_n100.json"
        ms = lambda src=src: F(src)["summary"]  # noqa: E731
        rows = lambda src=src: F(src)["rows"]  # noqa: E731
        trio("maskRemovalFlips", suf, lambda ms=ms: ms()["qk_removal"]["frac_flipped"], src)
        trio("maskRemovalBeyond", suf, lambda ms=ms: ms()["qk_removal"]["redirection"]["flips_beyond_reference"], src)
        macro(f"maskRandomFlips{suf}", lambda ms=ms: fmt(ms()["qk_random"]["frac_flipped"]["point"], 2), src)
        macro(f"maskSameAnswerFlips{suf}", lambda ms=ms: fmt(ms()["same_answer_donor/intermediates"]["frac_flipped"]["point"], 2), src)
        for c, nm in (("l1_latents/path/alone", "LOneAlone"), ("answer_heads/path/alone", "AnswerAlone"),
                      ("all_routes/path/alone", "AllRoutesAlone"), ("l2_latents_mask/path/alone", "LTwoAlone"),
                      ("l2_latents_mask/offpath/alone", "LTwoOffAlone"), ("all_masks/path/alone", "AllMasks"),
                      ("all_masks/offpath/alone", "AllMasksOff"), ("isolation/path/alone", "Iso"),
                      ("isolation/offpath/alone", "IsoOff"), ("isolation/path/plus_removal", "IsoPlusRemoval")):
            trio(f"mask{nm}Flips", suf, lambda ms=ms, c=c: ms()[c]["frac_flipped"], src)
        for c, nm in (("l1_latents/path/plus_removal", "LOne"), ("answer_heads/path/plus_removal", "Answer"),
                      ("all_routes/path/plus_removal", "AllRoutes")):
            trio(f"mask{nm}PlusRemovalVsRemoval", suf, lambda rows=rows, c=c: paired_flips(rows(), c, "qk_removal"), src)
        trio("maskIsoVsLTwo", suf, lambda rows=rows: paired_flips(rows(), "isolation/path/alone", "l2_latents_mask/path/alone"), src)
        trio("maskAllMasksVsLTwo", suf, lambda rows=rows: paired_flips(rows(), "all_masks/path/alone", "l2_latents_mask/path/alone"), src)
        trio("maskIsoVsAllMasks", suf, lambda rows=rows: paired_flips(rows(), "isolation/path/alone", "all_masks/path/alone"), src)
        trio("maskIsoPlusRemovalVsRemoval", suf, lambda rows=rows: paired_flips(rows(), "isolation/path/plus_removal", "qk_removal"), src)

        # -- experiments 6 and 6b: recovery (all seeds)
        src = R + "recovery_n100.json"
        rs = lambda src=src: F(src)["summary"]  # noqa: E731
        rrows = lambda src=src: F(src)["rows"]  # noqa: E731
        for c, nm in (("mlp/answer/L1/mean/alone", "MlpMean"), ("mlp/answer/L1/same_answer/alone", "MlpSameAnswer"),
                      ("mlp/answer/L1/random/alone", "MlpRandom"), ("mlp/answer/L2/mean/alone", "MlpLTwoMean"),
                      ("mlp/latent/both/mean/alone", "MlpLatentMean"),
                      ("thoughtK/same_answer/alone", "ThoughtKSameAnswer"), ("thoughtK/random/alone", "ThoughtKRandom"),
                      ("allq_decoy_edges/path/alone", "DecoyEdges"), ("allq_both_cand_edges/path/alone", "BothCandEdges")):
            macro(f"rec{nm}DT{suf}", lambda rs=rs, c=c: fmt(rs()[c]["median_dT"]["point"], 1), src)
            macro(f"rec{nm}Flips{suf}", lambda rs=rs, c=c: fmt(rs()[c]["frac_flipped"]["point"], 2), src)
            macro(f"rec{nm}Esc{suf}", lambda rs=rs, c=c: fmt(rs()[c]["frac_escaped"]["point"], 2), src)

        def restored(cname, rrows=rrows):
            fl = [r for r in rrows() if present(r, "qk_removal") and flipped(r["cells"]["qk_removal"]) and present(r, cname)]
            return sum(1 for r in fl if r["cells"][cname]["T"] > 50), len(fl)
        for c, nm in (("mlp/answer/L1/same_answer/plus_removal", "MlpSameAnswer"), ("thoughtK/same_answer/plus_removal", "ThoughtK"),
                      ("thoughtKm1/same_answer/plus_removal", "ThoughtKmOne"), ("thoughtKm2/same_answer/plus_removal", "ThoughtKmTwo")):
            macro(f"carry{nm}Restored{suf}", lambda c=c, f=restored: fmt(f(c)[0]), src)
            macro(f"carry{nm}Of{suf}", lambda c=c, f=restored: fmt(f(c)[1]), src)
        macro(f"carryRemovalPDecoy{suf}", lambda rs=rs: fmt(rs()["fallback_line"]["removal_p_decoy"]["point"], 2), src)
        for c, nm in (("allq_decoy_edges/path/plus_removal", "DecoyEdges"), ("allq_both_cand_edges/path/plus_removal", "BothCandEdges"),
                      ("allq_decoy_edges/ctrl/plus_removal", "DecoyEdgesCtrl")):
            trio(f"rec{nm}PlusRemovalVsRemoval", suf, lambda rrows=rrows, c=c: paired_flips(rrows(), c, "qk_removal"), src)
        if run in ("seed0", "seed1"):  # the T distribution under this mask was added for seeds 0 and 1 only
            for k, nm in (("mean_T", "MeanT"), ("median_T", "MedianT")):
                trio(f"recBothCandEdges{nm}", suf, lambda rs=rs, k=k: rs()["both_candidates_T_distribution"][k], src, 1)

        # -- experiment 8a: the removal's flipped graphs (all seeds)
        src = R + "winner_margin_n100.json"
        ws = lambda src=src: F(src)["summary"]  # noqa: E731
        for c, nm in (("removal", "Removal"), ("removal_orth", "Orth"), ("random", "Random")):
            trio(f"wm{nm}Flips", suf, lambda ws=ws, c=c: ws()["flips"][c]["frac_flipped"], src)
        trio("wmAttnRatio", suf, lambda ws=ws: ws()["attention_drop_orth_over_removal"], src)
        for c, nm in (("change_margin_probe_Km1_removal", "KmOneRemoval"), ("change_margin_probe_Km1_removal_orth", "KmOneOrth"),
                      ("change_margin_probe_Km1_random", "KmOneRandom"), ("change_margin_probe_K_removal", "KRemoval"),
                      ("change_margin_probe_K_removal_orth", "KOrth")):
            trio(f"wmMargin{nm}", suf, lambda ws=ws, c=c: ws()[c]["all"], src, 1)
        trio("wmSpanShare", suf, lambda ws=ws: ws()["span_share_probe_in_removed_span_Km1"]["median"], src, 3)
        trio("wmSpanShareRandom", suf, lambda ws=ws: ws()["span_share_probe_in_random_span_Km1"]["median"], src, 3)
        trio("wmAucPDecoy", suf, lambda ws=ws: ws()["auc_removal_flip"]["baseline_p_decoy"]["all"], src)
        trio("wmAucMarginKmOne", suf, lambda ws=ws: ws()["auc_removal_flip"]["neg_margin_probe_Km1_none"]["all"], src)
        trio("wmAucAnswerUnderRemoval", suf, lambda ws=ws: ws()["auc_answer_under_removal"]["margin_probe_K_removal"], src)

        # -- experiment 8b: held-out check, test graphs 100-399 (read in gaps_reanalysis.json)
        gsrc = "results/gaps_reanalysis.json"
        ho = lambda run=run: F(gsrc)["heldout"][run]  # noqa: E731
        trio("heldRemovalFlips", suf, lambda ho=ho: ho()["removal_frac_flipped"], gsrc)
        macro(f"heldRandomFlips{suf}", lambda ho=ho: fmt(ho()["random_frac_flipped"]["point"], 2), gsrc)
        macro(f"heldLTwoFlips{suf}", lambda ho=ho: fmt(ho()["l2_mask_frac_flipped"]["point"], 2), gsrc)
        trio("heldAucPDecoy", suf, lambda ho=ho: ho()["auc_baseline_p_decoy_for_removal_flip"]["baseline_correct"], gsrc)
        trio("heldAucOutDegree", suf, lambda ho=ho: ho()["auc_parent_out_degree_for_removal_flip"]["baseline_correct"], gsrc)

        # -- experiment 9: the restore test (all seeds)
        src = R + "restore_n100.json"
        macro(f"restoreNFlipped{suf}", lambda src=src: fmt(F(src)["summary"]["n_removal_flipped"]), src)
        for k, nm in (("path", "Path"), ("offpath", "Offpath"), ("edges_all", "EdgesAll"), ("all", "All")):
            trio(f"restore{nm}", suf, lambda src=src, k=k: F(src)["summary"][f"restore/{k}"]["rescue_share"], src)
            macro(f"restore{nm}Surviving{suf}", lambda src=src, k=k: fmt(F(src)["summary"][f"restore/{k}"]["n_surviving"]), src)

        # -- experiment 10: the isolation leftover (read in gaps_reanalysis.json)
        e10 = lambda run=run: F(gsrc)["experiment10"]["seeds"][run]["all"]  # noqa: E731
        for k, nm in (("s_iso", "SIso"), ("s_1", "SOne"), ("s_offpath_matched", "SOff"), ("s_above", "SAbove"),
                      ("s_below", "SBelow"), ("E", "E"), ("N", "N"), ("U", "U"), ("G", "G"),
                      ("gap_cell1", "GapOne"), ("gap_isolation_path", "GapIso")):
            trio(f"lft{nm}", suf, lambda e10=e10, k=k: e10()[k], gsrc)
        macro(f"lftLost{suf}", lambda e10=e10: fmt(e10()["cell1_lost"]), gsrc)
        macro(f"lftGained{suf}", lambda e10=e10: fmt(e10()["cell1_gained"]), gsrc)
        macro(f"lftNCorrect{suf}", lambda e10=e10: fmt(e10()["n_baseline_correct"]), gsrc)
        macro(f"lftEscapes{suf}", lambda e10=e10: fmt(e10()["wrong_cand"]["escape"]), gsrc)
        macro(f"lftWrong{suf}", lambda e10=e10: fmt(e10()["wrong_cand"]["n"]), gsrc)
        trio("lftDecoyOnlyE", suf, lambda run=run: F(gsrc)["experiment10"]["seeds"][run]["decoy_only_graphs"]["E"], gsrc)
        macro(f"lftLabelAuc{suf}", lambda run=run: fmt(F(gsrc)["experiment10_zero_cost"][run]["auc_target_id_below_for_wrong"]
                                                     ["allq_both_cand_edges/path/alone"]["baseline_correct"]["point"], 2), gsrc)

    # pooled over the four seeds
    gsrc = "results/gaps_reanalysis.json"
    P = lambda: F(gsrc)["experiment10"]["pooled"]  # noqa: E731
    for k, nm in (("s_iso", "SIso"), ("s_1", "SOne"), ("s_offpath_matched", "SOff"), ("s_above", "SAbove"),
                  ("s_below", "SBelow"), ("E", "E"), ("N", "N"), ("U", "U"), ("G", "G"),
                  ("gap_cell1", "GapOne"), ("gap_isolation_path", "GapIso")):
        trio(f"lft{nm}", "Pooled", lambda k=k: P()["all"][k], gsrc)
    for k, nm in (("cell1_lost", "Lost"), ("cell1_gained", "Gained"), ("n_baseline_correct", "NCorrect"), ("n_target_id_below", "NBelow")):
        macro(f"lft{nm}Pooled", lambda k=k: fmt(P()["all"][k]), gsrc)
    macro("lftEscapesPooled", lambda: fmt(P()["all"]["wrong_cand"]["escape"]), gsrc)
    macro("lftWrongPooled", lambda: fmt(P()["all"]["wrong_cand"]["n"]), gsrc)
    trio("lftDecoyOnlyE", "Pooled", lambda: P()["decoy_only_graphs"]["E"], gsrc)
    macro("lftDecoyOnlyNPooled", lambda: fmt(P()["decoy_only_graphs"]["n_baseline_correct"]), gsrc)
    trio("lftEscShare", "Pooled", lambda: P()["all"]["escape_share_cell1"], gsrc)
    trio("lftFracMid", "Pooled", lambda: P()["all"]["cell1_T_distribution_baseline_correct"]["frac_T_between_20_and_80"], gsrc)
    RP = lambda: F(gsrc)["experiment9_restore_pooled"]  # noqa: E731
    macro("restoreNFlippedPooled", lambda: fmt(RP()["n_removal_flipped"]), gsrc)
    for k, nm in (("path", "Path"), ("offpath", "Offpath"), ("edges_all", "EdgesAll"), ("all", "All")):
        trio(f"restore{nm}", "Pooled", lambda k=k: RP()[k]["rescue_share"], gsrc)
        macro(f"restore{nm}SurvivingPooled", lambda k=k: fmt(RP()[k]["n_surviving"]), gsrc)
    for k, nm in (("edges_all_minus_path", "EdgesAllMinusPath"), ("all_minus_edges_all", "AllMinusEdgesAll"), ("path_minus_offpath", "PathMinusOffpath")):
        trio(f"restore{nm}", "Pooled", lambda k=k: RP()[k], gsrc)

    # ==== winner separability per step, both models (experiments 7 and 8d) ===
    for run, suf in list(SEEDS.items()) + list(GPT2.items()):
        src = f"results/{run}/winner_probe.json"
        wp = lambda src=src: F(src)  # noqa: E731
        kmax = 4 if run in SEEDS else 6
        for k in range(1, kmax + 1):
            w = WORD[k]
            macro(f"wpAucPass{w}{suf}", lambda wp=wp, k=k: fmt(wp()["learned_probe"][str(k)]["auc"], 3), src)
            for b, nm in (("input_embedding", "Emb"), ("jlens", "Jac"), ("probe", "ProbeBasis")):
                macro(f"wpSep{nm}Pass{w}{suf}", lambda wp=wp, k=k, b=b: fmt(wp()["separation"][b][str(k)]["frac_target_higher"]["point"], 2), src)
        if run in SEEDS:
            # per K, thought K-1 and the final thought; training holdout and test split
            for split, sn in (("winner_probe.json", "Train"), ("winner_probe_test.json", "Test")):
                s2 = f"results/{run}/{split}"
                for K in (3, 4):
                    for step, stn in ((K - 1, "KmOne"), (K, "Final")):
                        macro(f"wp{sn}K{WORD[K]}{stn}{suf}", lambda s2=s2, K=K, step=step: fmt(F(s2)["by_K"][str(K)]["learned_probe"][str(step)]["auc"], 3), s2)
                        if sn == "Test":
                            macro(f"wp{sn}K{WORD[K]}{stn}Correct{suf}", lambda s2=s2, K=K, step=step: fmt(F(s2)["by_K"][str(K)]["learned_probe"][str(step)]["auc_correct_only"], 3), s2)

    # ==== experiment 7: GPT-2 =================================================
    for run, suf in GPT2.items():
        src = f"results/{run}/evaluation.json"
        g = lambda src=src: F(src)["generation"]  # noqa: E731
        for split, sn in (("original_test", "Orig"), ("vendor_test", "Vendor")):
            for cond, cn in (("six", "Latents"), ("zero_markers", "MarkersOnly"), ("none", "NoMarkers")):
                trio(f"gptAcc{sn}{cn}", suf, lambda g=g, split=split, cond=cond: g()[split][cond]["accuracy"], src, 1, pct=True)
        tc = lambda src=src: F(src)["two_candidate"]  # noqa: E731
        trio("gptAccOwnThoughts", suf, lambda tc=tc: tc()["accuracy_own_thoughts"], src, 1, pct=True)
        trio("gptAccZeroed", suf, lambda tc=tc: tc()["accuracy_zeroed"], src, 1, pct=True)
        trio("gptZeroDT", suf, lambda tc=tc: tc()["median_dT_zeroed"], src, 1)
        macro(f"gptZeroFlips{suf}", lambda tc=tc: fmt(tc()["frac_flipped_zeroed"]["point"], 2), src)
        csrc = f"results/{run}/cells_n100.json"
        ce = lambda csrc=csrc: F(csrc)  # noqa: E731
        readers = lambda ce=ce: {k: v for k, v in ce()["head_scores_search_latent"].items() if v["mass"] >= 0.5}  # noqa: E731
        macro(f"gptReaders{suf}", lambda r=readers: fmt(len(r())), csrc)
        macro(f"gptContentMedian{suf}", lambda r=readers: fmt(statistics.median(v["content"] for v in r().values()), 2), csrc)
        macro(f"gptContentMax{suf}", lambda r=readers: fmt(max(v["content"] for v in r().values()), 2), csrc)
        macro(f"gptContentMaxHead{suf}", lambda r=readers: max(r().items(), key=lambda kv: kv[1]["content"])[0], csrc, text=True)
        macro(f"gptPositionMedian{suf}", lambda r=readers: fmt(statistics.median(v["position"] for v in r().values()), 2), csrc)
        a = lambda ce=ce: ce()["summary"]["attention"]["qk_removal"]  # noqa: E731
        macro(f"gptAttnBefore{suf}", lambda a=a: fmt(a()["last_step_before"]["point"], 2), csrc)
        macro(f"gptAttnAfter{suf}", lambda a=a: fmt(a()["last_step_after"]["point"], 2), csrc)
        macro(f"gptNDirections{suf}", lambda a=a: fmt(a()["n_directions"]), csrc)
        for c, nm in (("qk_removal", "Removal"), ("latents_mask/path/alone", "Mask"), ("thoughtK/random/alone", "ThoughtKRandom"),
                      ("thoughtK/same_answer/plus_removal", "ThoughtKSameAnswerPlusRemoval"), ("reserialized", "Reserialized")):
            macro(f"gpt{nm}Flips{suf}", lambda ce=ce, c=c: fmt(ce()["summary"][c]["frac_flipped"]["point"], 2), csrc)
            macro(f"gpt{nm}DT{suf}", lambda ce=ce, c=c: fmt(ce()["summary"][c]["median_dT"]["point"], 1), csrc)
        trio("gptRemovalBeyond", suf, lambda ce=ce: ce()["summary"]["qk_removal"]["redirection"]["flips_beyond_reference"], csrc)
        macro(f"gptProbeBasisAuc{suf}", lambda run=run: fmt(F(f"results/{run}/probe_basis_report.json")["median_holdout_auc"], 2),
              f"results/{run}/probe_basis_report.json")

    # ==== relabel retrains (DECISIONS.md, amendments 1 and 2) =================
    for sub, an in (("relabel", "relabel"), ("relabel_names", "relabelNames")):
        for run in ("seed0", "seed1"):
            suf = SEEDS[run]
            for ser in range(4):
                src = f"results/{run}/{sub}/evaluation_ser{ser}.json"
                macro(f"{an}AccSer{['Zero', 'One', 'Two', 'Three'][ser]}{suf}", lambda src=src: fmt(F(src)["test_accuracy"], 1, pct=True), src)
            src = f"results/{run}/{sub}/best.json"
            macro(f"{an}ValAcc{suf}", lambda src=src: fmt(F(src)["val_acc"], 1, pct=True), src)

    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--table", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.repo / "paper" / "numbers.tex"
    table = args.table or args.repo / "paper" / "numbers_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    N = build(args.repo)
    lines = ["% AUTO-GENERATED by scripts/make_numbers.py -- do not edit by hand."]
    rows = ["<!-- AUTO-GENERATED by scripts/make_numbers.py -- do not edit by hand. -->",
            "", "| Macro | Value | Source |", "|---|---|---|"]
    missing = []
    for name, (producer, text, src) in N.items():
        try:
            val = producer()
            body = val if text else ensure_math(val)
        except Exception as e:  # noqa: BLE001
            missing.append((name, repr(e)))
            val, body = "??MISSING??", r"\textbf{??MISSING??}"
        lines.append(rf"\newcommand{{\{name}}}{{{body}}}")
        rows.append(f"| `\\{name}` | {val} | `{src}` |")
    out.write_text("\n".join(lines) + "\n")
    table.write_text("\n".join(rows) + "\n")
    print("\n".join(rows[2:]))
    print(f"\nwrote {out} ({len(N)} macros) and {table}")
    if missing:
        print(f"\n{len(missing)} MISSING -- fix before using:", file=sys.stderr)
        for name, err in missing:
            print(f"  \\{name}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
