"""Fit per-node frontier probes on step-1 thoughts.

For each node-token v: logistic regression on raw step-1 recycled thoughts,
label 1 iff BFS-depth_g(v) == 1, over training graphs containing v. The last
500 training graphs are held out for decode ROC-AUC (reported).

Writes results/<run>/probe_basis.pt  (tensor (vocab, 768); zero rows = no probe)
   and results/<run>/probe_basis_report.json.

Experiment 8: --score-test scores the saved basis on the 419 test graphs
(pinned test serialization, step-1 thoughts; the per-node AUC is rank based,
so the saved unit directions need no bias) and writes
results/<run>/probe_basis_test_report.json.

    python src/phoenix/fit_probes.py --run-name seed0 --device cpu
    python src/phoenix/fit_probes.py --run-name seed0 --device cpu --score-test
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "reasoning-by-superposition"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

import torch  # noqa: E402

from harness import Runner, bfs_depths  # noqa: E402

HOLDOUT = 500
L2 = 1e-3
ITERS = 200
MIN_POS = 50


def graph_nodes(sample):
    nodes = {sample["root"]}
    for s, t in sample["edges"]:
        nodes.add(s)
        nodes.add(t)
    return nodes


def auc(scores_pos, scores_neg):
    """Rank-based ROC-AUC without sklearn."""
    s = torch.cat([scores_pos, scores_neg])
    ranks = s.argsort().argsort().float() + 1
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def score_test(runner, basis, seed, n_graphs=None):
    """Score a probe basis on the test split: per node with a probe row, the
    ROC-AUC of <step-1 thought, direction> for depth-1 membership over the
    test graphs that contain the node. Pinned test serialization; the first
    n_graphs test graphs when given (tests), else all 419."""
    from measure import capture
    from prompts import Prompt
    from sets import load_test, test_pin
    test = load_test()
    idx = list(range(len(test)))[:n_graphs]
    X = torch.zeros(len(idx), basis.shape[1])
    present, depth1 = [], []
    for row, i in enumerate(idx):
        s = test[i]
        pr = Prompt.from_sample(s, test_pin(i, seed))
        X[row] = capture(runner, pr.ids(runner.tok))[0].float().cpu()
        present.append(graph_nodes(s))
        depth1.append({v for v, d in bfs_depths(s["edges"], s["root"]).items() if d == 1})
    report = {}
    for v in range(basis.shape[0]):
        if float(basis[v].norm()) == 0:
            continue
        rows = [r for r in range(len(idx)) if v in present[r]]
        if not rows:
            continue
        scores = X[rows] @ basis[v].float()
        y = torch.tensor([1.0 if v in depth1[r] else 0.0 for r in rows])
        a = auc(scores[y == 1], scores[y == 0])
        report[str(v)] = {"n": len(rows), "n_pos": int(y.sum()), "test_auc": None if a != a else round(a, 4)}
    aucs = [r["test_auc"] for r in report.values() if r["test_auc"] is not None]
    return {
        "n_graphs": len(idx), "split": "test", "serialization": "pinned test seeds", "step": 1, "label": "bfs_depth==1",
        "n_nodes_scored": len(aucs),
        "median_test_auc": round(sorted(aucs)[len(aucs) // 2], 4) if aucs else None,
        "min_test_auc": round(min(aucs), 4) if aucs else None,
        "per_node": report,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="seed0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-test", action="store_true",
                   help="experiment 8: score the saved probe_basis.pt on the test split and exit")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = ROOT / "results" / args.run_name
    runner = Runner(
        ROOT / "ckpts" / args.run_name / "best.pt", device=args.device
    )
    if args.score_test:
        basis = torch.load(out_dir / "probe_basis.pt", map_location="cpu", weights_only=True)
        rep = score_test(runner, basis, args.seed)
        with open(out_dir / "probe_basis_test_report.json", "w") as f:
            json.dump(rep, f, indent=2)
        print(f"test split: {rep['n_nodes_scored']} nodes, median AUC {rep['median_test_auc']}, min {rep['min_test_auc']}")
        print(f"written: {out_dir / 'probe_basis_test_report.json'}")
        return
    data = json.load(open(VENDOR / "data/prosqa_train_graph_4_coconut.json"))
    print(f"extracting step-1 thoughts from {len(data)} training graphs ...")

    thoughts = torch.zeros(len(data), 768)
    for i, sample in enumerate(data):
        cap = {}

        def grab(t, info, cap=cap):
            cap["t"] = t.detach().float().cpu()
            return t

        runner.measure(sample, 1, grab)
        thoughts[i] = cap["t"]
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(data)}")

    depth1 = [
        {v for v, d in bfs_depths(s["edges"], s["root"]).items() if d == 1}
        for s in data
    ]
    present = [graph_nodes(s) for s in data]
    all_nodes = sorted(set().union(*present))
    n_fit = len(data) - HOLDOUT
    dev = torch.device(args.device)
    X_all = thoughts.to(dev)

    vocab = runner.wte.shape[0]
    basis = torch.zeros(vocab, 768)
    report = {}
    for v in all_nodes:
        fit_idx = [i for i in range(n_fit) if v in present[i]]
        ho_idx = [i for i in range(n_fit, len(data)) if v in present[i]]
        y_fit = torch.tensor(
            [1.0 if v in depth1[i] else 0.0 for i in fit_idx], device=dev
        )
        if y_fit.sum() < MIN_POS:
            report[str(v)] = {"skipped": True, "n_pos_fit": int(y_fit.sum())}
            continue
        X_fit = X_all[fit_idx]

        w = torch.zeros(768, device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=ITERS, line_search_fn="strong_wolfe")
        bce = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(X_fit @ w + b, y_fit) + L2 * (w @ w)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            scores_ho = (X_all[ho_idx] @ w + b).cpu()
            y_ho = torch.tensor([1.0 if v in depth1[i] else 0.0 for i in ho_idx])
            a = auc(scores_ho[y_ho == 1], scores_ho[y_ho == 0])
            basis[v] = (w / w.norm()).cpu()
            report[str(v)] = {
                "n_fit": len(fit_idx),
                "n_pos_fit": int(y_fit.sum().item()),
                "holdout_auc": round(a, 4),
            }
            print(f"node {v:3d}: n_fit {len(fit_idx):6d} pos {int(y_fit.sum()):5d} "
                  f"holdout AUC {a:.4f}")

    aucs = [r["holdout_auc"] for r in report.values() if "holdout_auc" in r]
    summary = {
        "n_nodes_probed": len(aucs),
        "median_holdout_auc": round(sorted(aucs)[len(aucs) // 2], 4),
        "min_holdout_auc": round(min(aucs), 4),
        "recipe": {
            "step": 1,
            "label": "bfs_depth==1",
            "holdout_last_n": HOLDOUT,
            "l2": L2,
            "lbfgs_iters": ITERS,
            "min_pos": MIN_POS,
            "seed": args.seed,
        },
        "per_node": report,
    }
    torch.save(basis, out_dir / "probe_basis.pt")
    with open(out_dir / "probe_basis_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nprobed {len(aucs)} nodes; median holdout AUC "
          f"{summary['median_holdout_auc']}, min {summary['min_holdout_auc']}")
    print(f"written: {out_dir / 'probe_basis.pt'}")


if __name__ == "__main__":
    main()
