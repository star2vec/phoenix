"""Per-step winner probe, one script for both model kinds (experiments 7, 8).

The from-scratch numbers in NOTES.md (check 3 of experiment 6) were computed
ad hoc; this recomputes them so every number in the comparison table points
to a results file.

Scored graphs: the last EVAL_LAST training graphs (the probe holdout; --eval
train, the default) or the 419 test graphs (--eval test, experiment 8). Per
pass k, three readouts and one learned probe:
  separation/<basis>   fraction of scored graphs where <t_k, u_target> >
                       <t_k, u_decoy> in the input-embedding basis, the probe
                       basis and the Jacobian basis (results/<run>/)
  learned_probe        multinomial logistic regression from t_k to the
                       target's node token, fit on train[FIT], scored as the
                       AUC over the scored candidate instances (target 1,
                       decoy 0, score = that candidate's logit); pairs whose
                       candidates share a node token are dropped and counted
Both are reported pooled over K at each absolute pass (the experiment 7
numbers, unchanged) and, under by_K, fit and scored per solution length K so
that a K=3 graph's final thought is never pooled with a K=4 graph's thought
K-1 (experiment 8). When results/<run>/jlens_basis_K<K>.pt exists (fit_jlens.py
--only-K) the per-K Jacobian readout uses it, else the pooled basis; the file
records which. The per-K probe weights are saved to
results/<run>/winner_probe_weights.pt for winner_margin.py.
Thoughts are cached in results/<run>/thought_cache.pt (training graphs) and
thought_cache_test.pt (test graphs), both gitignored.

    python src/phoenix/winner_probe.py --model symbol --run-name seed0 --device cpu
    python src/phoenix/winner_probe.py --model symbol --run-name seed0 --device cpu --eval test
    python src/phoenix/winner_probe.py --model gpt2 --run-name gpt2_dilgren --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from measure import capture  # noqa: E402
from prompts import Prompt  # noqa: E402
from sets import ROOT, load_test, load_train, require_checkpoint, test_pin, train_pin, write_json  # noqa: E402
from stats import bootstrap  # noqa: E402

FIT = (500, 2500)
EVAL_LAST = 500
L2 = 1e-3
ITERS = 200
BASES = ("input_embedding", "probe", "jlens")
WEIGHTS_FILE = "winner_probe_weights.pt"


def prompt_for(model_kind, sample, gi, seed, split="train"):
    if model_kind == "symbol":
        pin = train_pin(gi, seed) if split == "train" else test_pin(gi, seed)
        return Prompt.from_sample(sample, pin)
    from nl import NLPrompt
    return NLPrompt.from_sample(sample)


def node_tokens(model_kind, pr):
    """(target token, decoy token) in the model's vocabulary. For GPT-2 a
    name's node token is its first token after the frame "### Root is a";
    pairs whose names share it are dropped by the caller."""
    if model_kind == "symbol":
        return pr.target, pr.decoy
    from nl import tokenizer
    tok = tokenizer()
    P = len(tok.encode(f"### {pr.names[pr.root]} is a", add_special_tokens=False))
    first = lambda v: tok.encode(pr.answer_text(v), add_special_tokens=False)[P]  # noqa: E731
    return first(pr.target), first(pr.decoy)


@torch.no_grad()
def thoughts_for(runner, model_kind, data, idx, seed, cache_path, split="train"):
    """{gi: (K_gi, d) tensor} for the given graphs of `data`, cached on disk."""
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = torch.load(cache_path, weights_only=True)
    missing = [gi for gi in idx if gi not in cache]
    t0 = time.time()
    for n, gi in enumerate(missing):
        pr = prompt_for(model_kind, data[gi], gi, seed, split)
        cap = capture(runner, pr.ids(runner.tok))
        cache[gi] = torch.stack([cap[k].float().cpu() for k in range(pr.K)])
        if (n + 1) % 200 == 0:
            print(f"  thoughts {n + 1}/{len(missing)} ({time.time() - t0:.0f}s)", flush=True)
    if missing and cache_path is not None:
        torch.save(cache, cache_path)
    return {gi: cache[gi] for gi in idx}


def graph_info(model_kind, data, idx, seed, split="train"):
    info = {}
    for gi in idx:
        pr = prompt_for(model_kind, data[gi], gi, seed, split)
        t, d = node_tokens(model_kind, pr)
        names = getattr(pr, "names", None)
        info[gi] = {"t": t, "d": d, "K": pr.K,
                    "t_name": names[pr.target] if names else str(pr.target),
                    "d_name": names[pr.decoy] if names else str(pr.decoy)}
    return info


@torch.no_grad()
def answers_for(runner, model_kind, data, idx, seed, split="train"):
    """{gi: the model's own answer is the target} for symbol models (one
    forward per graph); empty for GPT-2. Lets every readout also be reported
    over the graphs the model answers correctly, since on a wrongly answered
    graph the thought carries the decoy and the probe reads that."""
    if model_kind != "symbol":
        return {}
    from measure import measure
    out = {}
    for gi in idx:
        pr = prompt_for(model_kind, data[gi], gi, seed, split)
        out[gi] = bool(measure(runner, pr)["T"] > 50)
    return out


def load_bases(run_dir):
    out = {}
    for name in ("probe", "jlens"):
        p = run_dir / f"{name}_basis.pt"
        if p.exists():
            out[name] = torch.load(p, map_location="cpu", weights_only=True)
    for K in (3, 4):
        p = run_dir / f"jlens_basis_K{K}.pt"
        if p.exists():
            out[f"jlens_K{K}"] = torch.load(p, map_location="cpu", weights_only=True)
    return out


def basis_direction(kind, obj, model_kind, tok_id, name, k):
    """Unit direction for a node in a basis, or None when the basis has none."""
    if model_kind == "symbol":
        v = obj[tok_id] if kind == "probe" else (obj[k, tok_id] if k < obj.shape[0] else None)
    else:
        if kind == "probe":
            if name not in obj["names"]:
                return None
            v = obj["vectors"][obj["names"].index(name)]
        else:
            if tok_id not in obj["ids"] or k >= obj["vectors"].shape[0]:
                return None
            v = obj["vectors"][k, obj["ids"].index(tok_id)]
    if v is None or float(v.norm()) == 0:
        return None
    return v / v.norm()


def fit_softmax(X, y, n_classes, l2=L2, iters=ITERS):
    W = torch.zeros(n_classes, X.shape[1], requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")
    ce = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(X @ W.T + b, y) + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def auc(pos, neg):
    s = torch.cat([pos, neg])
    ranks = s.argsort().argsort().float() + 1
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def separation_block(th, info, ev, k, basis, obj, model_kind, wte, correct=None):
    """Fraction of scored graphs whose thought at pass k scores the target
    above the decoy in one basis; also over the correctly answered graphs
    when `correct` is given."""
    wins, wins_correct, n_missing = [], [], 0
    for gi in ev:
        i = info[gi]
        if basis == "input_embedding":
            ut, ud = wte[i["t"]], wte[i["d"]]
            ut, ud = ut / ut.norm(), ud / ud.norm()
        else:
            if obj is None:
                n_missing += 1
                continue
            ut = basis_direction(basis, obj, model_kind, i["t"], i["t_name"], k)
            ud = basis_direction(basis, obj, model_kind, i["d"], i["d_name"], k)
            if ut is None or ud is None:
                n_missing += 1
                continue
        if i["t"] == i["d"]:
            n_missing += 1
            continue
        x = th[gi][k]
        wins.append(float(x @ ut > x @ ud))
        if correct and correct.get(gi):
            wins_correct.append(wins[-1])
    out = {"frac_target_higher": bootstrap(wins, np.mean) if wins else None,
           "n": len(wins), "n_without_direction": n_missing}
    if correct:
        out["frac_target_higher_correct_only"] = bootstrap(wins_correct, np.mean) if wins_correct else None
        out["n_correct"] = len(wins_correct)
    return out


def probe_block(th_fit, info_fit, fit, th_ev, info_ev, ev, k, classes, correct=None):
    """Fit the learned winner probe on `fit` at pass k and score it on `ev`
    (and, when `correct` is given, on the correctly answered graphs alone).
    Returns (entry, weights)."""
    cls_index = {c: i for i, c in enumerate(classes)}
    X = torch.stack([th_fit[gi][k] for gi in fit])
    y = torch.tensor([cls_index[info_fit[gi]["t"]] for gi in fit])
    W, b = fit_softmax(X, y, len(classes))
    pos, neg, ok, dropped = [], [], [], 0
    for gi in ev:
        i = info_ev[gi]
        if i["t"] not in cls_index or i["d"] not in cls_index or i["t"] == i["d"]:
            dropped += 1
            continue
        logit = th_ev[gi][k] @ W.T + b
        pos.append(logit[cls_index[i["t"]]])
        neg.append(logit[cls_index[i["d"]]])
        ok.append(bool(correct.get(gi)) if correct else False)
    if pos:
        pos, neg = torch.stack(pos), torch.stack(neg)
        entry = {"auc": auc(pos, neg), "frac_target_higher": bootstrap((pos > neg).float().tolist(), np.mean)}
        if correct:
            sel = torch.tensor(ok)
            entry["auc_correct_only"] = auc(pos[sel], neg[sel]) if sel.any() else None
            entry["n_eval_correct"] = int(sel.sum())
            # experiment 9: the margin against the model's own answer, all graphs
            m = pos - neg
            entry["auc_own_answer"] = auc(m[sel], m[~sel]) if sel.any() and (~sel).any() else None
            entry["own_answer_sign_agrees"] = int(((m > 0) == sel).sum())
    else:
        entry = {"auc": None, "frac_target_higher": None}
    entry.update({"n_eval": int(len(pos)), "n_dropped": dropped, "n_fit": len(fit), "n_classes": len(classes)})
    return entry, {"W": W, "b": b, "classes": list(classes)}


def run(runner, model_kind, train, fit_idx, eval_idx, bases, cache, seed=0,
        eval_data=None, eval_split="train", eval_cache=None):
    """Returns (results, weights). The scored graphs come from eval_data
    (default: the training file, i.e. the probe holdout) under eval_split's
    pinned serialization, cached at eval_cache (default: `cache`)."""
    fit_idx, eval_idx = list(fit_idx), list(eval_idx)
    if eval_data is None:
        eval_data, eval_split, eval_cache = train, "train", cache
    th_fit = thoughts_for(runner, model_kind, train, fit_idx, seed, cache, "train")
    th_ev = thoughts_for(runner, model_kind, eval_data, eval_idx, seed, eval_cache, eval_split)
    info_fit = graph_info(model_kind, train, fit_idx, seed, "train")
    info_ev = graph_info(model_kind, eval_data, eval_idx, seed, eval_split)
    k_max = max(info_fit[gi]["K"] for gi in fit_idx)
    wte = runner.wte.detach().float().cpu()
    classes = sorted({info_fit[gi]["t"] for gi in fit_idx})
    correct = answers_for(runner, model_kind, eval_data, eval_idx, seed, eval_split)
    res = {"separation": {b: {} for b in BASES}, "learned_probe": {}, "by_K": {},
           "n_classes": len(classes), "k_max": k_max, "eval_split": eval_split, "n_eval_graphs": len(eval_idx),
           "n_eval_correct": int(sum(correct.values())) if correct else None}
    weights = {}
    # pooled over K at each absolute pass (the experiment 7 numbers)
    for k in range(k_max):
        fit_k = [gi for gi in fit_idx if info_fit[gi]["K"] > k]
        ev_k = [gi for gi in eval_idx if info_ev[gi]["K"] > k]
        for basis in BASES:
            res["separation"][basis][str(k + 1)] = separation_block(th_ev, info_ev, ev_k, k, basis, bases.get(basis), model_kind, wte, correct)
        entry, w = probe_block(th_fit, info_fit, fit_k, th_ev, info_ev, ev_k, k, classes, correct)
        res["learned_probe"][str(k + 1)] = entry
        weights[f"pooled/k{k}"] = w
        sep = res["separation"]["input_embedding"][str(k + 1)]["frac_target_higher"]
        print(f"step {k + 1} (pooled): input-embedding separation {sep['point'] if sep else float('nan'):.3f} "
              f"(n {len(ev_k)}), learned probe AUC {entry['auc'] if entry['auc'] is not None else float('nan'):.3f}", flush=True)
    # per solution length K (experiment 8): no pooling of K=3 and K=4
    for K in sorted({info_fit[gi]["K"] for gi in fit_idx}):
        fit_K = [gi for gi in fit_idx if info_fit[gi]["K"] == K]
        ev_K = [gi for gi in eval_idx if info_ev[gi]["K"] == K]
        classes_K = sorted({info_fit[gi]["t"] for gi in fit_K})
        jl = f"jlens_K{K}" if f"jlens_K{K}" in bases else ("jlens" if "jlens" in bases else None)
        blk = {"n_fit": len(fit_K), "n_eval": len(ev_K), "n_classes": len(classes_K), "jlens_basis": jl,
               "separation": {b: {} for b in BASES}, "learned_probe": {}}
        for k in range(K):
            for basis in BASES:
                obj = bases.get(jl) if basis == "jlens" and jl else bases.get(basis)
                blk["separation"][basis][str(k + 1)] = separation_block(th_ev, info_ev, ev_K, k, basis, obj, model_kind, wte, correct)
            entry, w = probe_block(th_fit, info_fit, fit_K, th_ev, info_ev, ev_K, k, classes_K, correct)
            blk["learned_probe"][str(k + 1)] = entry
            weights[f"K{K}/k{k}"] = w
        res["by_K"][str(K)] = blk
        aucs = ", ".join(f"{blk['learned_probe'][str(k + 1)]['auc'] if blk['learned_probe'][str(k + 1)]['auc'] is not None else float('nan'):.3f}" for k in range(K))
        aucs_ok = ", ".join(f"{blk['learned_probe'][str(k + 1)].get('auc_correct_only') or float('nan'):.3f}" for k in range(K)) if correct else "-"
        print(f"K={K} (n_fit {len(fit_K)}, n_eval {len(ev_K)}): learned probe AUC by step {aucs}; over correctly answered graphs {aucs_ok}", flush=True)
    return res, weights


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=("symbol", "gpt2"), required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", choices=("train", "test"), default="train",
                   help="score on the last 500 training graphs (default) or on the 419 test graphs")
    args = p.parse_args()
    ckpt = require_checkpoint(args.run_name)
    if args.model == "symbol":
        from harness import Runner
        runner = Runner(ckpt, device=args.device, seed=args.seed)
    else:
        from gpt2 import GPT2Runner
        runner = GPT2Runner(ckpt, device=args.device, seed=args.seed)
    train = load_train()
    run_dir = ROOT / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_idx = list(range(*FIT))
    bases = load_bases(run_dir)
    print(f"bases present: {sorted(bases)}")
    if args.eval == "train":
        eval_idx, eval_data, eval_cache, out_name = list(range(len(train) - EVAL_LAST, len(train))), None, None, "winner_probe.json"
    else:
        test = load_test()
        eval_idx, eval_data, eval_cache, out_name = list(range(len(test))), test, run_dir / "thought_cache_test.pt", "winner_probe_test.json"
    res, weights = run(runner, args.model, train, fit_idx, eval_idx, bases, run_dir / "thought_cache.pt", args.seed,
                       eval_data=eval_data, eval_split=args.eval, eval_cache=eval_cache)
    res.update({"model": args.model, "run_name": args.run_name, "fit_slice": list(FIT),
                "eval_last": EVAL_LAST if args.eval == "train" else None,
                "bases_present": sorted(bases), "l2": L2, "lbfgs_iters": ITERS,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_json(run_dir / out_name, res)
    torch.save(weights, run_dir / WEIGHTS_FILE)
    print(f"written: {run_dir / out_name} and {run_dir / WEIGHTS_FILE}")


if __name__ == "__main__":
    main()
