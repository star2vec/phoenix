"""Per-step winner probe, one script for both model kinds (experiment 7).

The from-scratch numbers in NOTES.md (check 3 of experiment 6) were computed
ad hoc; this recomputes them so every number in the comparison table points
to a results file.

Held-out graphs: the last EVAL_LAST training graphs (the probe holdout). Per
pass k, three readouts and one learned probe:
  separation/<basis>   fraction of held-out graphs where <t_k, u_target> >
                       <t_k, u_decoy> in the input-embedding basis, the probe
                       basis and the Jacobian basis (results/<run>/)
  learned_probe        multinomial logistic regression from t_k to the
                       target's node token, fit on train[FIT], scored as the
                       AUC over the held-out candidate instances (target 1,
                       decoy 0, score = that candidate's logit); pairs whose
                       candidates share a node token are dropped and counted
Thoughts are cached in results/<run>/thought_cache.pt (gitignored).

    python src/phoenix/winner_probe.py --model symbol --run-name seed0 --device cpu
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
from sets import ROOT, load_train, require_checkpoint, train_pin, write_json  # noqa: E402
from stats import bootstrap  # noqa: E402

FIT = (500, 2500)
EVAL_LAST = 500
L2 = 1e-3
ITERS = 200


def prompt_for(model_kind, sample, gi, seed):
    if model_kind == "symbol":
        return Prompt.from_sample(sample, train_pin(gi, seed))
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
def thoughts_for(runner, model_kind, train, idx, seed, cache_path):
    """{gi: (K_gi, d) tensor} for the given graphs, cached on disk."""
    cache = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = torch.load(cache_path, weights_only=True)
    missing = [gi for gi in idx if gi not in cache]
    t0 = time.time()
    for n, gi in enumerate(missing):
        pr = prompt_for(model_kind, train[gi], gi, seed)
        cap = capture(runner, pr.ids(runner.tok))
        cache[gi] = torch.stack([cap[k].float().cpu() for k in range(pr.K)])
        if (n + 1) % 200 == 0:
            print(f"  thoughts {n + 1}/{len(missing)} ({time.time() - t0:.0f}s)", flush=True)
    if missing and cache_path is not None:
        torch.save(cache, cache_path)
    return {gi: cache[gi] for gi in idx}


def load_bases(run_dir):
    out = {}
    for name in ("probe", "jlens"):
        p = run_dir / f"{name}_basis.pt"
        if p.exists():
            out[name] = torch.load(p, map_location="cpu", weights_only=True)
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


def run(runner, model_kind, train, fit_idx, eval_idx, bases, cache, seed=0):
    idx = list(fit_idx) + list(eval_idx)
    th = thoughts_for(runner, model_kind, train, idx, seed, cache)
    info = {}
    for gi in idx:
        pr = prompt_for(model_kind, train[gi], gi, seed)
        t, d = node_tokens(model_kind, pr)
        names = getattr(pr, "names", None)
        info[gi] = {"t": t, "d": d, "K": pr.K,
                    "t_name": names[pr.target] if names else str(pr.target),
                    "d_name": names[pr.decoy] if names else str(pr.decoy)}
    k_max = max(info[gi]["K"] for gi in idx)
    wte = runner.wte.detach().float().cpu()
    classes = sorted({info[gi]["t"] for gi in fit_idx})
    cls_index = {c: i for i, c in enumerate(classes)}
    res = {"separation": {b: {} for b in ("input_embedding", "probe", "jlens")}, "learned_probe": {},
           "n_classes": len(classes), "k_max": k_max}
    for k in range(k_max):
        fit_k = [gi for gi in fit_idx if info[gi]["K"] > k]
        ev_k = [gi for gi in eval_idx if info[gi]["K"] > k]
        # readouts in the three bases
        for basis in ("input_embedding", "probe", "jlens"):
            wins, n_missing = [], 0
            for gi in ev_k:
                i = info[gi]
                if basis == "input_embedding":
                    ut, ud = wte[i["t"]], wte[i["d"]]
                    ut, ud = ut / ut.norm(), ud / ud.norm()
                else:
                    obj = bases.get(basis)
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
            res["separation"][basis][str(k + 1)] = {"frac_target_higher": bootstrap(wins, np.mean) if wins else None,
                                                    "n": len(wins), "n_without_direction": n_missing}
        # the learned winner probe
        X = torch.stack([th[gi][k] for gi in fit_k])
        y = torch.tensor([cls_index[info[gi]["t"]] for gi in fit_k])
        W, b = fit_softmax(X, y, len(classes))
        pos, neg, dropped = [], [], 0
        for gi in ev_k:
            i = info[gi]
            if i["t"] not in cls_index or i["d"] not in cls_index or i["t"] == i["d"]:
                dropped += 1
                continue
            logit = th[gi][k] @ W.T + b
            pos.append(logit[cls_index[i["t"]]])
            neg.append(logit[cls_index[i["d"]]])
        pos, neg = torch.stack(pos), torch.stack(neg)
        res["learned_probe"][str(k + 1)] = {
            "auc": auc(pos, neg), "frac_target_higher": bootstrap((pos > neg).float().tolist(), np.mean),
            "n_eval": int(len(pos)), "n_dropped": dropped, "n_fit": len(fit_k),
        }
        print(f"step {k + 1}: input-embedding separation "
              f"{res['separation']['input_embedding'][str(k + 1)]['frac_target_higher']['point']:.3f} "
              f"(n {len(ev_k)}), learned probe AUC {res['learned_probe'][str(k + 1)]['auc']:.3f}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=("symbol", "gpt2"), required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
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
    eval_idx = list(range(len(train) - EVAL_LAST, len(train)))
    bases = load_bases(run_dir)
    print(f"bases present: {sorted(bases)}")
    res = run(runner, args.model, train, fit_idx, eval_idx, bases, run_dir / "thought_cache.pt", args.seed)
    res.update({"model": args.model, "run_name": args.run_name, "fit_slice": list(FIT), "eval_last": EVAL_LAST,
                "bases_present": sorted(bases), "l2": L2, "lbfgs_iters": ITERS,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_json(run_dir / "winner_probe.json", res)
    print(f"written: {run_dir / 'winner_probe.json'}")


if __name__ == "__main__":
    main()
