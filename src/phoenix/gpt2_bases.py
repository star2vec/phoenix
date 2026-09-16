"""Refit the probe basis and the Jacobian basis for a GPT-2 run (experiment
7). They serve the winner probe's basis readouts (winner_probe.py).

probe   per concept name, logistic regression on step-1 thoughts (pass 0),
        label 1 iff the name is at depth 1 in the graph, over the graphs
        containing it; the same recipe as fit_probes.py (L2 1e-3, LBFGS 200,
        at least MIN_POS positives). Fit set train[:2500] and holdout
        train[-500:] from the thought cache winner_probe.py writes (the
        from-scratch fit used all training graphs; the smaller set is
        recorded). Output {"names", "ids", "vectors"(n, d)} and a report.
jlens   mean causal Jacobian of each node token's answer-position logit with
        respect to each pass's recycled thought (a detached leaf), over
        train[500:2500], the slice fit_jlens.py used; per token, not per
        name (colliding first tokens share a row). Output {"ids",
        "vectors"(K, n, d)} and a report. Resumable (partial sums saved).

    python src/phoenix/gpt2_bases.py --run-name gpt2_dilgren --device cpu --part probe
    python src/phoenix/gpt2_bases.py --run-name gpt2_dilgren --device cpu --part jlens
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from gpt2 import GPT2Runner  # noqa: E402
from nl import NLPrompt, is_person  # noqa: E402
from prompts import bfs_depths  # noqa: E402
from sets import ROOT, load_train, require_checkpoint, require_file, write_json  # noqa: E402

PROBE_FIT = 2500
HOLDOUT = 500
L2, ITERS, MIN_POS = 1e-3, 200, 50
JL_FIT = (500, 2500)
SAVE_EVERY = 50


def auc(pos, neg):
    s = torch.cat([pos, neg])
    ranks = s.argsort().argsort().float() + 1
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_probe_basis(runner, train, run_dir):
    cache = torch.load(require_file(run_dir / "thought_cache.pt", "winner_probe.py --model gpt2"), weights_only=True)
    fit_idx = [gi for gi in range(PROBE_FIT) if gi in cache]
    ho_idx = [gi for gi in range(len(train) - HOLDOUT, len(train)) if gi in cache]
    assert len(fit_idx) == PROBE_FIT and len(ho_idx) == HOLDOUT, (len(fit_idx), len(ho_idx))
    X = {gi: cache[gi][0] for gi in fit_idx + ho_idx}  # step-1 thought
    present, depth1, node_tok = {}, {}, {}
    for gi in fit_idx + ho_idx:
        s = train[gi]
        pr = NLPrompt.from_sample(s)
        names = s["idx_to_symbol"]
        d = bfs_depths(s["edges"], s["root"])
        present[gi] = {names[v] for v in pr.nodes() if not is_person(names[v])}
        depth1[gi] = {names[v] for v, dep in d.items() if dep == 1 and not is_person(names[v])}
        for nm, t in pr.readout()["node_by_name"].items():
            node_tok[nm] = t
    all_names = sorted(set().union(*present.values()))
    vectors, names_out, ids_out, report = [], [], [], {}
    for nm in all_names:
        f = [gi for gi in fit_idx if nm in present[gi]]
        h = [gi for gi in ho_idx if nm in present[gi]]
        y = torch.tensor([1.0 if nm in depth1[gi] else 0.0 for gi in f])
        if y.sum() < MIN_POS:
            report[nm] = {"skipped": True, "n_pos_fit": int(y.sum())}
            continue
        Xf = torch.stack([X[gi] for gi in f])
        w = torch.zeros(Xf.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=ITERS, line_search_fn="strong_wolfe")
        bce = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(Xf @ w + b, y) + L2 * (w @ w)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            sh = torch.stack([X[gi] for gi in h]) @ w + b
            yh = torch.tensor([1.0 if nm in depth1[gi] else 0.0 for gi in h])
            a = auc(sh[yh == 1], sh[yh == 0])
            vectors.append((w / w.norm()).detach())
            names_out.append(nm)
            ids_out.append(node_tok[nm])
            report[nm] = {"n_fit": len(f), "n_pos_fit": int(y.sum()), "holdout_auc": round(a, 4), "token": node_tok[nm]}
            print(f"{nm:10s} n_fit {len(f):5d} pos {int(y.sum()):4d} holdout AUC {a:.4f}", flush=True)
    aucs = [r["holdout_auc"] for r in report.values() if "holdout_auc" in r]
    torch.save({"names": names_out, "ids": ids_out, "vectors": torch.stack(vectors)}, run_dir / "probe_basis.pt")
    summary = {"n_names_probed": len(aucs), "median_holdout_auc": round(statistics.median(aucs), 4),
               "min_holdout_auc": round(min(aucs), 4),
               "recipe": {"step": 1, "label": "bfs_depth==1", "fit": f"train[:{PROBE_FIT}]", "holdout_last_n": HOLDOUT,
                          "l2": L2, "lbfgs_iters": ITERS, "min_pos": MIN_POS}, "per_name": report}
    write_json(run_dir / "probe_basis_report.json", summary)
    print(f"probed {len(aucs)} names; median holdout AUC {summary['median_holdout_auc']}, min {summary['min_holdout_auc']}")


def forward_with_leaves(runner, pr):
    ids = pr.ids(runner.tok)
    input_ids = torch.tensor([ids], device=runner.device)
    n = len(ids)
    leaves = {}

    def edit(k, v):
        leaf = v.detach().clone().requires_grad_(True)
        leaves[k] = leaf
        return leaf

    out = runner.model(input_ids, torch.ones_like(input_ids), input_ids.clone(),
                       torch.arange(n, device=runner.device).reshape(1, -1), thought_edit=edit)
    return out.logits[0, -1], leaves


def fit_jlens_basis(runner, train, run_dir):
    fit = list(range(*JL_FIT))
    all_ids = sorted({t for gi in fit for t in NLPrompt.from_sample(train[gi]).readout()["nodes"]})
    idx = {t: i for i, t in enumerate(all_ids)}
    K = NLPrompt.from_sample(train[fit[0]]).K
    d = runner.wte.shape[1]
    part = run_dir / "jlens_partial.pt"
    if part.exists():
        st = torch.load(part, weights_only=True)
        gsum, usum, kcount, done = st["gsum"], st["usum"], st["kcount"], st["done"]
        print(f"resuming after {done} graphs")
    else:
        gsum, usum, kcount, done = torch.zeros(K, len(all_ids), d), torch.zeros(K, len(all_ids), d), torch.zeros(K), 0
    t0 = time.time()
    for n_done, gi in enumerate(fit[done:], start=done):
        pr = NLPrompt.from_sample(train[gi])
        nodes = pr.readout()["nodes"]
        logits, leaves = forward_with_leaves(runner, pr)
        leaf_list = [leaves[k] for k in sorted(leaves)]
        grads = torch.autograd.grad(logits[nodes], leaf_list, grad_outputs=torch.eye(len(nodes), device=logits.device),
                                    is_grads_batched=True, retain_graph=False)
        for j, k in enumerate(sorted(leaves)):
            kcount[k] += 1
            for i, t in enumerate(nodes):
                g = grads[j][i].detach().float().cpu()
                gsum[k, idx[t]] += g
                usum[k, idx[t]] += g / g.norm().clamp_min(1e-12)
        if (n_done + 1) % SAVE_EVERY == 0 or n_done + 1 == len(fit):
            torch.save({"gsum": gsum, "usum": usum, "kcount": kcount, "done": n_done + 1}, part)
            print(f"  {n_done + 1}/{len(fit)} ({time.time() - t0:.0f}s)", flush=True)
    basis = torch.zeros(K, len(all_ids), d)
    wte = runner.wte.detach().float().cpu()
    report = {"per_step": {}, "per_token": {}}
    for k in range(K):
        conc, cos_w = [], []
        for t in all_ids:
            mg = gsum[k, idx[t]]
            if float(mg.norm()) == 0:
                continue
            j = mg / mg.norm()
            basis[k, idx[t]] = j
            c = float((usum[k, idx[t]] / max(int(kcount[k]), 1)).norm())
            w = wte[t]
            cw = float(j @ (w / w.norm()))
            conc.append(c)
            cos_w.append(cw)
            report["per_token"][f"t{t}_k{k}"] = {"concentration": round(c, 4), "cos_wte": round(cw, 4)}
        report["per_step"][str(k)] = {"n_graphs": int(kcount[k]), "median_concentration": round(statistics.median(conc), 4),
                                      "median_abs_cos_wte": round(statistics.median(abs(x) for x in cos_w), 4)}
    torch.save({"ids": all_ids, "vectors": basis}, run_dir / "jlens_basis.pt")
    write_json(run_dir / "jlens_basis_report.json", {"fit_slice": list(JL_FIT), "n_tokens": len(all_ids), "k_max": K,
                                                     "per_step": report["per_step"], "per_token": report["per_token"]})
    print(report["per_step"])
    print(f"written: {run_dir / 'jlens_basis.pt'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-name", default="gpt2_dilgren")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--part", choices=("probe", "jlens"), required=True)
    args = p.parse_args()
    ckpt = require_checkpoint(args.run_name)
    runner = GPT2Runner(ckpt, device=args.device, seed=args.seed)
    torch.manual_seed(args.seed)
    train = load_train()
    run_dir = ROOT / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.part == "probe":
        fit_probe_basis(runner, train, run_dir)
    else:
        fit_jlens_basis(runner, train, run_dir)


if __name__ == "__main__":
    main()
