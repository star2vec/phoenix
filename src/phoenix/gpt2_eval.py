"""Regime check for a fine-tuned GPT-2 COCONUT checkpoint (experiment 7).

Accuracy exactly as the official run.py scores it (greedy decoding, exact
match of the text after the last '#'), with the key/value cache, under three
prompt conditions:
  six           <|start-latent|> + six <|latent|> + <|end-latent|>  (what the model saw)
  zero_markers  the markers with no latent between them (the stage-0 format,
                so the model may write its reasoning steps in text first)
  none          no markers, no latents
on the original ProsQA test set (500; the literature's numbers) and the
vendor test set (419; our held-out graphs). Also, on the vendor test set at
the readout position: the teacher-forced two-candidate accuracy with the
model's own thoughts and with every recycled thought replaced by the zero
vector, and the change in T under zeroing (against the from-scratch
necessity cell zero/all). And the held-out check: how many recipient
questions occur in the original training and test sets.

    python src/phoenix/gpt2_eval.py --run-name gpt2_dilgren --device cpu
    python src/phoenix/gpt2_eval.py --run-name gpt2_dilgren --device cpu --limit 3 --sets original
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpt2 import GPT2Runner, greedy_generate, source_info  # noqa: E402
from measure import answer_split, run_ids  # noqa: E402
from nl import NLPrompt, special_ids  # noqa: E402
from sets import N100, PILOT, ROOT, load_test, require_checkpoint, require_file, write_json  # noqa: E402
from stats import bootstrap  # noqa: E402

ORIGINAL = ROOT / "data" / "prosqa_original"
CONDITIONS = ("six", "zero_markers", "none")


def latent_block(tok, condition):
    sp = special_ids(tok)
    if condition == "six":
        return [sp["start"]] + [sp["latent"]] * 6 + [sp["end"]]
    if condition == "zero_markers":
        return [sp["start"], sp["end"]]
    return []


def score(text, answer):
    pred = text.split("#")[-1].replace(",", "").strip()
    return pred == answer.replace(",", "").strip(), pred


def generation_accuracy(runner, data, condition, limit=None, tag=""):
    tok = runner.tok
    rows = []
    t0 = time.time()
    items = data if limit is None else data[:limit]
    for i, s in enumerate(items):
        ids = tok.encode(s["question"] + "\n", add_special_tokens=True) + latent_block(tok, condition)
        toks = greedy_generate(runner, ids, max_new_tokens=128)
        text = tok.decode(toks, skip_special_tokens=True)
        ok, pred = score(text, s["answer"])
        steps_text = text.split("###")[0] if "###" in text else text
        rows.append({"i": i, "correct": bool(ok), "pred": pred, "n_generated": len(toks),
                     "n_step_sentences": steps_text.count("."), "text": text})
        if (i + 1) % 50 == 0:
            acc = np.mean([r["correct"] for r in rows])
            print(f"  {tag}/{condition} {i + 1}/{len(items)} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    acc = [float(r["correct"]) for r in rows]
    return {"n": len(rows), "accuracy": bootstrap(acc, np.mean) if acc else None,
            "mean_generated": float(np.mean([r["n_generated"] for r in rows])) if rows else None,
            "frac_with_step_sentences": float(np.mean([r["n_step_sentences"] > 0 for r in rows])) if rows else None,
            "rows": rows}


@torch.no_grad()
def two_candidate(runner, data, limit=None):
    rows = []
    zero = lambda k, t: torch.zeros_like(t)  # noqa: E731
    items = data if limit is None else data[:limit]
    for i, s in enumerate(items):
        pr = NLPrompt.from_sample(s)
        ids = pr.ids(runner.tok)
        ro = pr.readout()
        base = answer_split(run_ids(runner, ids), ro["target"], ro["decoy"], node_ids=ro["nodes"])
        zs = answer_split(run_ids(runner, ids, zero), ro["target"], ro["decoy"], node_ids=ro["nodes"])
        rows.append({"i": i, "L": pr.L, "base": base, "zeroed": zs, "dT": zs["T"] - base["T"],
                     "n_collided": ro["n_collided"], "prefix_len": ro["prefix_len"]})
        if (i + 1) % 100 == 0:
            print(f"  two-candidate {i + 1}/{len(items)}", flush=True)
    return {
        "n": len(rows),
        "accuracy_own_thoughts": bootstrap([float(r["base"]["T"] > 50) for r in rows], np.mean),
        "accuracy_zeroed": bootstrap([float(r["zeroed"]["T"] > 50) for r in rows], np.mean),
        "median_dT_zeroed": bootstrap([r["dT"] for r in rows], np.median),
        "mean_dT_zeroed": bootstrap([r["dT"] for r in rows], np.mean),
        "frac_flipped_zeroed": bootstrap([float(r["dT"] <= -50) for r in rows], np.mean),
        "median_e_zeroed": bootstrap([r["zeroed"]["e"] for r in rows], np.median),
        "mean_e_zeroed": bootstrap([r["zeroed"]["e"] for r in rows], np.mean),
        "median_e_own": bootstrap([r["base"]["e"] for r in rows], np.median),
        "n_prefix_not_frame": sum(1 for r in rows if r["prefix_len"] != rows[0]["prefix_len"]),
        "n_graphs_with_collisions": sum(1 for r in rows if r["n_collided"] > 0),
        "rows": rows,
    }


def held_out_check(vendor_test):
    train = json.load(open(require_file(ORIGINAL / "prosqa_train.json", "download (see plan)")))
    test = json.load(open(require_file(ORIGINAL / "prosqa_test.json", "download (see plan)")))
    trq = {s["question"] for s in train}
    teq = {s["question"] for s in test}
    rec = [vendor_test[gi]["question"] for gi in list(N100) + list(PILOT)]
    allq = [s["question"] for s in vendor_test]
    return {"recipients": len(rec), "recipients_in_original_train": sum(q in trq for q in rec),
            "recipients_in_original_test": sum(q in teq for q in rec),
            "vendor_test_in_original_train": sum(q in trq for q in allq),
            "vendor_test_in_original_test": sum(q in teq for q in allq), "n_vendor_test": len(allq)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-name", default="gpt2_dilgren")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sets", default="original,vendor")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--out-name", default="evaluation")
    args = p.parse_args()
    ckpt = require_checkpoint(args.run_name)
    runner = GPT2Runner(ckpt, device=args.device, seed=args.seed)
    print("load:", runner.load_report)
    vendor_test = load_test()
    sets = {}
    if "original" in args.sets:
        sets["original_test"] = json.load(open(require_file(ORIGINAL / "prosqa_test.json", "download (see plan)")))
    if "vendor" in args.sets:
        sets["vendor_test"] = vendor_test
    run_dir = ROOT / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{args.out_name}.json"
    out = {"run_name": args.run_name, "checkpoint_source": source_info(args.run_name), "device": args.device,
           "limit": args.limit, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "generation": {}}
    if path.exists() and args.limit is None:
        out = json.load(open(path))  # resume: conditions already present are kept
        print(f"resuming from {path}: {[(k, sorted(v)) for k, v in out['generation'].items()]}")
    if (ORIGINAL / "prosqa_train.json").exists():
        out["held_out_check"] = held_out_check(vendor_test)
        print("held-out check:", out["held_out_check"])
        assert out["held_out_check"]["recipients_in_original_train"] == 0, "a recipient is in the model's training set"
    for name, data in sets.items():
        out["generation"].setdefault(name, {})
        for cond in args.conditions.split(","):
            if cond in out["generation"][name] and args.limit is None:
                print(f"{name}/{cond}: already done, kept")
                continue
            r = generation_accuracy(runner, data, cond, args.limit, tag=name)
            out["generation"][name][cond] = r
            write_json(path, out)
            acc = r["accuracy"]["point"] if r["accuracy"] else float("nan")
            print(f"{name}/{cond}: accuracy {acc:.3f} (n {r['n']}), mean generated {r['mean_generated']:.1f} tokens, "
                  f"with step sentences {r['frac_with_step_sentences']:.2f}", flush=True)
            if args.limit is not None and args.limit <= 5:
                for row in r["rows"]:
                    print("   ", repr(row["text"]))
    if "vendor_test" in sets and ("two_candidate" not in out or args.limit is not None):
        out["two_candidate"] = two_candidate(runner, vendor_test, args.limit)
        write_json(path, out)
    if "two_candidate" in out:
        tc = out["two_candidate"]
        print(f"two-candidate accuracy own {tc['accuracy_own_thoughts']['point']:.3f}, zeroed {tc['accuracy_zeroed']['point']:.3f}, "
              f"median dT zeroed {tc['median_dT_zeroed']['point']:+.1f}, flipped {tc['frac_flipped_zeroed']['point']:.2f}")
    out["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(path, out)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
