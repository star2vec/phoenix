"""Iterative Nullspace Projection (INLP) per node — the amnesic-probing arm's
fitter (after Ravfogel et al. 2004.07667 and Elazar et al. 2006.00995).

Extends fit_probes.py: instead of ONE logistic direction per node, fit a
SEQUENCE of directions by INLP — fit the same logistic probe on the residual, record its
unit direction, project it out, refit — until a freshly trained probe can no
longer beat chance on the held-out residual (holdout ROC-AUC <= AUC_STOP) or
K_MAX directions are removed. The collected directions (orthonormalized) span the
linearly-decodable "node-v-in-depth-1-frontier" subspace for node v.

Everything not INLP-specific is IDENTICAL to fit_probes.py: same step-1
recycled thoughts, same label (BFS-depth==1), same last-500 holdout, same logistic
recipe (LBFGS, L2 1e-3, 200 iters, fp32, seed 0), same MIN_POS cutoff.

Writes results/<run>/inlp_basis.pt      (dict {node:int -> (768, k) float tensor})
   and results/<run>/inlp_basis_report.json.
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
from fit_probes import auc, graph_nodes  # noqa: E402

# --- fit_probes.py recipe (identical) ---
HOLDOUT = 500
L2 = 1e-3
ITERS = 200
MIN_POS = 50
# --- INLP-specific ---
K_MAX = 8          # cap on directions removed per node; 8/768
AUC_STOP = 0.55    # stop when a fresh probe on the residual can no longer beat
                   # chance on holdout (ROC-AUC <= 0.55 ~ no reliable linear signal)


def fit_logistic(X, y, dev, iters=ITERS):
    """L2 logistic probe via LBFGS. Returns unit weight direction (768,)."""
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(X @ w + b, y) + L2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (w / w.norm()).detach(), b.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="seed0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-max", type=int, default=K_MAX)
    p.add_argument("--auc-stop", type=float, default=AUC_STOP)
    p.add_argument("--lbfgs-iters", type=int, default=ITERS)
    p.add_argument("--tag", default="", help="suffix for output artifacts")
    args = p.parse_args()
    k_max, auc_stop = args.k_max, args.auc_stop

    torch.manual_seed(args.seed)
    out_dir = ROOT / "results" / args.run_name
    runner = Runner(ROOT / "ckpts" / args.run_name / "best.pt", device=args.device)
    data = json.load(open(VENDOR / "data/prosqa_train_graph_4_coconut.json"))

    # thoughts depend only on best.pt + data (not on k_max), so cache them in
    # ckpts/ (git-ignored) to make full-null-out re-fits instant
    cache = ROOT / "ckpts" / args.run_name / "inlp_thoughts.pt"
    if cache.exists():
        thoughts = torch.load(cache, map_location="cpu", weights_only=True)
        print(f"loaded cached step-1 thoughts {tuple(thoughts.shape)} from {cache}")
    else:
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
        torch.save(thoughts, cache)

    depth1 = [
        {v for v, d in bfs_depths(s["edges"], s["root"]).items() if d == 1}
        for s in data
    ]
    present = [graph_nodes(s) for s in data]
    all_nodes = sorted(set().union(*present))
    n_fit = len(data) - HOLDOUT
    dev = torch.device(args.device)
    X_all = thoughts.to(dev)

    basis = {}       # node -> (768, k) orthonormal directions
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
        y_ho = torch.tensor([1.0 if v in depth1[i] else 0.0 for i in ho_idx])

        X_fit = X_all[fit_idx].clone()   # residual, projected as we go
        X_ho = X_all[ho_idx].clone()
        dirs = []           # accumulated unit directions (pre-orthonormalization)
        auc_traj = []
        for r in range(k_max):
            w, b = fit_logistic(X_fit, y_fit, dev, args.lbfgs_iters)
            with torch.no_grad():
                scores_ho = (X_ho @ w + b).cpu()
                a = auc(scores_ho[y_ho == 1], scores_ho[y_ho == 0])
            auc_traj.append(round(float(a), 4))
            if a <= auc_stop:
                break                      # fresh probe on residual can't beat chance
            dirs.append(w)
            # project this direction out of BOTH residuals (INLP step)
            with torch.no_grad():
                X_fit = X_fit - torch.outer(X_fit @ w, w)
                X_ho = X_ho - torch.outer(X_ho @ w, w)

        if not dirs:                       # first probe already <= AUC_STOP: 1 dir min
            dirs = [w]
        D = torch.stack(dirs, dim=1)       # (768, k)
        # orthonormalize (sequential residual projection keeps them ~orthogonal;
        # QR guarantees an exact orthonormal basis for the removed subspace)
        Q, _ = torch.linalg.qr(D)
        B = Q[:, : D.shape[1]].contiguous().cpu()
        basis[v] = B
        report[str(v)] = {
            "n_fit": len(fit_idx),
            "n_pos_fit": int(y_fit.sum().item()),
            "k": B.shape[1],
            "auc_trajectory": auc_traj,     # AUC of the r-th fresh probe on residual
        }
        print(f"node {v:3d}: k={B.shape[1]}  auc_traj={auc_traj}")

    ks = [r["k"] for r in report.values() if "k" in r]
    summary = {
        "n_nodes": len(ks),
        "median_k": sorted(ks)[len(ks) // 2] if ks else 0,
        "max_k": max(ks) if ks else 0,
        "k_hist": {str(k): ks.count(k) for k in sorted(set(ks))},
        "recipe": {
            "step": 1,
            "label": "bfs_depth==1",
            "holdout_last_n": HOLDOUT,
            "l2": L2,
            "lbfgs_iters": args.lbfgs_iters,
            "min_pos": MIN_POS,
            "k_max": k_max,
            "auc_stop": auc_stop,
            "seed": args.seed,
            "source": "INLP per Ravfogel 2004.07667",
        },
        "per_node": report,
    }
    torch.save(basis, out_dir / f"inlp_basis{args.tag}.pt")
    with open(out_dir / f"inlp_basis_report{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{len(ks)} nodes; median k={summary['median_k']}, max k={summary['max_k']}, "
          f"hist={summary['k_hist']}")
    print(f"written: {out_dir / f'inlp_basis{args.tag}.pt'}")


if __name__ == "__main__":
    main()
