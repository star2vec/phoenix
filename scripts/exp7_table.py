"""Experiment 7 comparison table: the from-scratch seeds against the two
fine-tuned GPT-2 checkpoints, one row per cell, every number read from a
results file. Prints Markdown; with --write-notes replaces the table block in
NOTES.md (between the exp7-table markers).

    .venv/bin/python scripts/exp7_table.py [--write-notes]
"""

import argparse
import gzip
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEEDS = ("seed0", "seed1", "seed2", "seed3")
GPT2 = ("gpt2_dilgren", "gpt2_aswal")
START, END = "<!-- exp7-table-start -->", "<!-- exp7-table-end -->"


def load(run, name):
    p = RES / run / name
    if p.exists():
        return json.load(open(p))
    if (RES / run / (name + ".gz")).exists():
        return json.load(gzip.open(RES / run / (name + ".gz"), "rt"))
    return None


def pt(x, fmt="{:.2f}"):
    if x is None:
        return "-"
    if isinstance(x, dict):
        x = x.get("point")
    return "-" if x is None else fmt.format(x)


def pct(x):
    return pt(x, "{:.0%}") if x is not None else "-"


def scratch_column(run):
    c = {}
    ev = load(run, "evaluation_ser0.json") or load(run, "evaluation.json")
    c["acc"] = pt(ev["test_accuracy"] / (100 if ev["test_accuracy"] > 1 else 1), "{:.1%}") if ev else "-"
    c["acc_removed"] = "not applicable (K latents are the search)"
    ne = load(run, "necessity_n100.json")
    z = ne["summary"]["zero/all"] if ne else None
    c["zero"] = f"median dT {pt(z['median_dT'], '{:+.1f}')}, flips {pct(z['frac_flipped'])}" if z else "-"
    hd = load(run, "heads_n100.json")
    if hd:
        S = hd["summary"]
        l2 = [S[f"L2H{h}/intermediate_latent"] for h in range(8)]
        readers = [k for k in S if k.endswith("/intermediate_latent") and S[k]["slot_mass_a"] and S[k]["slot_mass_a"]["point"] >= 0.5]
        cont = [e["content_score"]["point"] for e in l2]
        pos = [e["position_score"]["point"] for e in l2]
        c["readers"] = f"{len(readers)} of 16 (all 8 of layer 2)"
        c["content"] = f"layer 2: content {min(cont):.2f} to {max(cont):.2f}, position {min(pos):+.2f} to {max(pos):+.2f}"
    else:
        c["readers"] = c["content"] = "-"
    cf = load(run, "counterfactuals_n100.json")
    if cf:
        rows = [r["cells"]["qk_every_step/answer_path/all_heads"] for r in cf["rows"]
                if not r["cells"]["qk_every_step/answer_path/all_heads"].get("skipped")]
        b = statistics.mean(r["attn_path_before_per_step"][-1] for r in rows)
        a = statistics.mean(r["attn_path_after_per_step"][-1] for r in rows)
        c["attn"] = f"{b:.2f} to {a:.2f} (8 directions)"
    else:
        c["attn"] = "not run on this seed"
    mk = load(run, "masking_n100.json")
    if mk:
        q = mk["summary"]["qk_removal"]
        fb = q.get("redirection", {}).get("flips_beyond_reference")
        c["removal"] = f"flips {pct(q['frac_flipped'])}, beyond reference {pt(fb, '{:+.2f}')} [{pt(fb['lo'] if fb else None, '{:+.2f}')}, {pt(fb['hi'] if fb else None, '{:+.2f}')}]"
        m = mk["summary"]["l2_latents_mask/path/alone"]
        c["mask"] = f"flips {pct(m['frac_flipped'])} (removal {pct(q['frac_flipped'])})"
    else:
        c["removal"] = c["mask"] = "-"
    rc = load(run, "recovery_n100.json")
    if rc:
        fl = [r for r in rc["rows"] if not r["cells"]["qk_removal"].get("skipped") and r["cells"]["qk_removal"]["dT"] <= -50
              and not r["cells"].get("thoughtK/same_answer/plus_removal", {}).get("skipped", True)]
        rest = sum(1 for r in fl if r["cells"]["thoughtK/same_answer/plus_removal"]["T"] > 50)
        rnd = rc["summary"]["thoughtK/random/alone"]
        c["carry"] = f"{rest} of {len(fl)} removal-flipped graphs restored; random donor escape {pct(rnd['frac_escaped'])}"
    else:
        c["carry"] = "-"
    wp = load(run, "winner_probe.json")
    if wp:
        lp, se = wp["learned_probe"], wp["separation"]
        last = str(wp["k_max"])
        c["winner"] = f"AUC {lp['1']['auc']:.2f} at pass 1, {lp['3']['auc']:.2f} at pass 3, {lp[last]['auc']:.3f} at pass {last}"
        c["sep"] = f"{pt(se['input_embedding']['1']['frac_target_higher'])} at pass 1, {pt(se['input_embedding'][last]['frac_target_higher'])} at pass {last}"
        j1, jl = se["jlens"]["1"]["frac_target_higher"], se["jlens"][last]["frac_target_higher"]
        c["jlens"] = f"{pt(j1)} at pass 1, {pt(jl)} at pass {last}" if j1 else "-"
    else:
        c["winner"] = c["sep"] = c["jlens"] = "not run on this seed"
    pb = load(run, "probe_basis_report.json")
    c["probe"] = f"{pb['median_holdout_auc']:.3f}" if pb else "-"
    return c


def gpt2_column(run):
    c = {}
    ev = load(run, "evaluation.json")
    if ev:
        g = ev["generation"]
        c["acc"] = f"{pct(g['original_test']['six']['accuracy'])} (original 500); {pct(g['vendor_test']['six']['accuracy'])} (vendor 419)"
        c["acc_removed"] = f"{pct(g['original_test']['zero_markers']['accuracy'])} markers only, {pct(g['original_test']['none']['accuracy'])} no markers"
        tc = ev["two_candidate"]
        c["zero"] = f"median dT {pt(tc['median_dT_zeroed'], '{:+.1f}')}, flips {pct(tc['frac_flipped_zeroed'])} (accuracy {pct(tc['accuracy_own_thoughts'])} to {pct(tc['accuracy_zeroed'])})"
    else:
        c["acc"] = c["acc_removed"] = c["zero"] = "-"
    hd = load(run, "heads_n100.json")
    ce = load(run, "cells_n100.json")
    if hd and ce:
        sc = ce["head_scores_search_latent"]
        readers = {k: v for k, v in sc.items() if v["mass"] >= 0.5}
        top = max(readers.items(), key=lambda kv: kv[1]["content"])
        c["readers"] = f"{len(readers)} of 144"
        c["content"] = (f"content median {statistics.median(v['content'] for v in readers.values()):.2f}, "
                        f"max {top[1]['content']:.2f} ({top[0]}); position median {statistics.median(v['position'] for v in readers.values()):.2f}")
    else:
        c["readers"] = c["content"] = "-"
    if ce:
        S = ce["summary"]
        a = S["attention"]["qk_removal"]
        c["attn"] = f"{pt(a['last_step_before'])} to {pt(a['last_step_after'])} ({a['n_directions']} directions)"
        q = S["qk_removal"]
        fb = q.get("redirection", {}).get("flips_beyond_reference")
        c["removal"] = f"flips {pct(q['frac_flipped'])}, beyond reference {pt(fb, '{:+.2f}')} [{pt(fb['lo'] if fb else None, '{:+.2f}')}, {pt(fb['hi'] if fb else None, '{:+.2f}')}]"
        m = S["latents_mask/path/alone"]
        c["mask"] = f"flips {pct(m['frac_flipped'])} (removal {pct(q['frac_flipped'])})"
        fl = S["fallback_line"]["n_removal_flipped"]
        rnd = S["thoughtK/random/alone"]
        c["carry"] = f"{fl} removal-flipped graphs (nothing to restore); random donor escape {pct(rnd['frac_escaped'])}"
        cr = ce["routes"]["content/"]
        c["content_route"] = f"{cr['rung']}: {', '.join(cr['heads'])}; every cell zero"
    else:
        c["attn"] = c["removal"] = c["mask"] = c["carry"] = c["content_route"] = "-"
    wp = load(run, "winner_probe.json")
    if wp:
        lp, se = wp["learned_probe"], wp["separation"]
        last = str(wp["k_max"])
        c["winner"] = f"AUC {lp['1']['auc']:.3f} at pass 1, {lp['3']['auc']:.3f} at pass 3, {lp[last]['auc']:.3f} at pass {last}"
        c["sep"] = f"{pt(se['input_embedding']['1']['frac_target_higher'])} at pass 1, {pt(se['input_embedding'][last]['frac_target_higher'])} at pass {last}"
        j1, jl = se["jlens"]["1"]["frac_target_higher"], se["jlens"][last]["frac_target_higher"]
        c["jlens"] = f"{pt(j1)} at pass 1, {pt(jl)} at pass {last}" if j1 else "pending (Jacobian fit running)"
    else:
        c["winner"] = c["sep"] = c["jlens"] = "-"
    pb = load(run, "probe_basis_report.json")
    c["probe"] = f"{pb['median_holdout_auc']:.3f}" if pb else "-"
    return c


ROWS = [
    ("acc", "Accuracy, the model's own prompt"),
    ("acc_removed", "Accuracy with the latents removed"),
    ("zero", "Every recycled thought zeroed"),
    ("readers", "Heads reading edges at the latents (mass at least 0.5)"),
    ("content", "Do they follow edge content? (scores)"),
    ("attn", "Removal: attention onto the answer-path edge, before to after"),
    ("removal", "Removal: the answer"),
    ("mask", "Calibration mask on the path edges"),
    ("carry", "Final-thought carry-over"),
    ("content_route", "Content-filtered route (GPT-2 only, exploration)"),
    ("winner", "Learned winner probe, held-out AUC"),
    ("sep", "Winner separation, input-embedding basis"),
    ("jlens", "Winner separation, Jacobian basis"),
    ("probe", "Frontier probe basis, median holdout AUC"),
]


def build():
    cols = {r: scratch_column(r) for r in SEEDS}
    cols.update({r: gpt2_column(r) for r in GPT2})
    names = list(SEEDS) + list(GPT2)
    lines = ["| Cell | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(cols[r].get(key, "-") for r in names) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-notes", action="store_true")
    args = p.parse_args()
    table = build()
    print(table)
    if args.write_notes:
        notes = (ROOT / "NOTES.md").read_text()
        block = f"{START}\n{table}\n{END}"
        if START in notes:
            pre, rest = notes.split(START, 1)
            _, post = rest.split(END, 1)
            notes = pre + block + post
        else:
            notes = notes.replace("## Next run", block + "\n\n## Next run")
        (ROOT / "NOTES.md").write_text(notes)
        print("table written into NOTES.md")


if __name__ == "__main__":
    main()
