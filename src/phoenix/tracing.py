"""Experiment 4a: causal tracing by position.

Clean run = the pinned original. Corrupted run = an on-manifold
counterfactual whose answer differs: the candidate swap (target and decoy
tokens transposed in the edge list; question line unchanged) and the last-hop
rewrite (the cut edge into the target now points at the decoy). For each
level (0 = input to layer 1, which at a latent position is the recycled
thought plus its position embedding; 1 = after layer 1; 2 = after layer 2)
and each absolute position, the clean activation is restored into the
corrupted run and T is recorded.

Recovery = (T_restored - T_corrupted) / (T_clean - T_corrupted), reported
when the clean and corrupted runs differ by more than 10 points of T (the
cutoff protects "there was an effect to recover").

Granularity: `token` restores one token position at a time; `slot`
restores a whole edge slot at once (cheaper; special positions stay single).

    python src/phoenix/tracing.py --run-name seed0 --device mps --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from attn_hooks import ResidualHooks  # noqa: E402
from common import (  # noqa: E402
    covariates, finish, graph_rng, header, load_runner, make_parser,
    recipient_prompts,
)
from measure import answer_split, run_ids  # noqa: E402
from prompts import candidate_swap, rewrite_at_depth  # noqa: E402
from stats import bootstrap  # noqa: E402

CORRUPTIONS = ("candidate_swap", "rewrite_last")
MIN_GAP = 10.0


def position_classes(L, K, ids_clean, ids_corr):
    """abs position -> (class, slot or None); differing positions flagged."""
    cls = {0: ("eos", None)}
    for j, (ps, pt, sep) in enumerate(L["slots"]):
        cls[ps] = ("slot_source", j)
        cls[pt] = ("slot_target", j)
        if sep is not None:
            cls[sep] = ("slot_sep", j)
    cls[L["q"]] = ("q", None)
    cls[L["c1"]] = ("cand_1", None)
    cls[L["c2"]] = ("cand_2", None)
    cls[L["r"]] = ("r", None)
    cls[L["root"]] = ("root", None)
    for i, p in enumerate(L["latents"]):
        cls[p] = (f"latent_{i + 1}", None)
    cls[L["a"]] = ("answer", None)
    differs = {p: ids_clean[p] != ids_corr[p] for p in range(len(ids_clean))}
    return cls, differs


def groups(L, granularity):
    """List of (label, [positions]) restore groups."""
    out = [("eos", [0])]
    for j, (ps, pt, sep) in enumerate(L["slots"]):
        if granularity == "slot":
            out.append((f"slot_{j}", [p for p in (ps, pt, sep) if p is not None]))
        else:
            out.append((f"slot_{j}_source", [ps]))
            out.append((f"slot_{j}_target", [pt]))
            if sep is not None:
                out.append((f"slot_{j}_sep", [sep]))
    out += [("q", [L["q"]]), ("cand_1", [L["c1"]]), ("cand_2", [L["c2"]]), ("r", [L["r"]]), ("root", [L["root"]])]
    out += [(f"latent_{i + 1}", [p]) for i, p in enumerate(L["latents"])]
    out.append(("answer", [L["a"]]))
    return out


def clean_store(runner, ids):
    base = runner.model.base_causallm
    with ResidualHooks(base) as rh:
        rh.record = True
        logits = run_ids(runner, ids)
        store = {lvl: dict(d) for lvl, d in rh.store.items()}
    return logits, store


def trace(runner, pr, cf, granularity):
    base = runner.model.base_causallm
    L = pr.layout()
    ids_clean, ids_corr = pr.ids(runner.tok), cf.ids(runner.tok)
    assert len(ids_clean) == len(ids_corr)
    logits_clean, store = clean_store(runner, ids_clean)
    clean = answer_split(logits_clean, pr.target, pr.decoy)
    corr = answer_split(run_ids(runner, ids_corr), pr.target, pr.decoy)
    gap = clean["T"] - corr["T"]
    levels = sorted(store)
    restored = {}  # label -> {level: T}
    for label, poss in groups(L, granularity):
        restored[label] = {}
        for lvl in levels:
            with ResidualHooks(base) as rh:
                for p in poss:
                    rh.patches[(lvl, p)] = store[lvl][p]
                T = answer_split(run_ids(runner, ids_corr), pr.target, pr.decoy)["T"]
            restored[label][lvl] = T
    cls, differs = position_classes(L, pr.K, ids_clean, ids_corr)
    return {
        "clean": clean, "corrupted": corr, "gap": gap, "levels": levels,
        "restored_T": restored,
        "differing_positions": [p for p, d in differs.items() if d],
        "differing_slots": sorted({cls[p][1] for p, d in differs.items() if d and cls[p][1] is not None}),
    }


def recovery(entry, T):
    if abs(entry["gap"]) < MIN_GAP:
        return None
    return (T - entry["corrupted"]["T"]) / entry["gap"]


def aggregate(rows, corruption):
    """Mean recovery per position class per level, bootstrapped over graphs."""
    per_graph = {}  # (class, level) -> list over graphs of the graph's mean recovery
    for r in rows:
        e = r["traces"].get(corruption)
        if e is None or e.get("skipped"):
            continue
        L = r["layout"]
        acc = {}
        for label, by_lvl in e["restored_T"].items():
            if label.startswith("slot_"):
                j = int(label.split("_")[1])
                if j in e["differing_slots"]:
                    cls = "slot_changed"
                else:
                    cls = "slot_unchanged"
                if label.endswith("_source"):
                    cls += "_source"
                elif label.endswith("_target"):
                    cls += "_target"
                elif label.endswith("_sep"):
                    cls += "_sep"
            elif label.startswith("latent_"):
                i = int(label.split("_")[1])
                cls = "latent_final" if i == r["K"] else "latent_intermediate"
            else:
                cls = label
            for lvl, T in by_lvl.items():
                rec = recovery(e, T)
                if rec is not None:
                    acc.setdefault((cls, int(lvl)), []).append(rec)
        for key, vals in acc.items():
            per_graph.setdefault(key, []).append(float(np.mean(vals)))
    out = {}
    for (cls, lvl), vals in sorted(per_graph.items()):
        out.setdefault(cls, {})[f"level_{lvl}"] = bootstrap(vals, np.mean)
    return out


def run(runner, recips, base_seed=0, granularity="token", corruptions=CORRUPTIONS, rows_path=None):
    """rows_path: optional JSONL file; finished graphs are appended as they
    complete and skipped on a rerun, so an interrupted run resumes."""
    import json
    rows, done = [], {}
    if rows_path is not None and Path(rows_path).exists():
        for line in open(rows_path):
            r = json.loads(line)
            done[r["gi"]] = r
    for gi, sample, pr in recips:
        if gi in done:
            rows.append(done[gi])
            continue
        rng = graph_rng(base_seed, gi)
        traces = {}
        for name in corruptions:
            cf, meta = candidate_swap(pr) if name == "candidate_swap" else rewrite_at_depth(pr, pr.K, rng)
            if cf is None:
                traces[name] = {"skipped": True, "reason": meta.get("reason")}
                continue
            traces[name] = trace(runner, pr, cf, granularity)
            traces[name]["meta"] = meta
            t = traces[name]
            print(f"graph {gi} {name}: clean T {t['clean']['T']:.1f} corrupted T {t['corrupted']['T']:.1f} "
                  f"(changed slots {t['differing_slots']})")
        row = {"gi": gi, "K": pr.K, "layout": pr.layout(), "cov": covariates(pr), "traces": traces}
        rows.append(row)
        if rows_path is not None:
            from sets import _default
            with open(rows_path, "a") as f:
                f.write(json.dumps(row, default=_default) + "\n")
    rows.sort(key=lambda r: r["gi"])
    summary = {name: aggregate(rows, name) for name in corruptions}
    summary["n_skipped"] = {name: sum(1 for r in rows if r["traces"].get(name, {}).get("skipped")) for name in corruptions}
    return {"rows": rows, "summary": summary, "granularity": granularity, "min_gap": MIN_GAP}


def main():
    p = make_parser(__doc__.split("\n")[0])
    p.add_argument("--granularity", choices=("token", "slot"), default="token")
    args = p.parse_args()
    runner = load_runner(args)
    recips = recipient_prompts(args.mode, args.seed)
    from sets import results_dir
    rows_path = results_dir(args.run_name) / f"tracing_{args.mode}_rows.jsonl"
    result = header(args, "tracing", granularity=args.granularity)
    result.update(run(runner, recips, args.seed, args.granularity, rows_path=rows_path))
    finish(args, "tracing", result)


if __name__ == "__main__":
    main()
