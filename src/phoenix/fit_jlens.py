"""Fit the J-lens basis (mean causal Jacobian of each node logit w.r.t. each thought).

j_v(k) = normalize( mean over fit graphs [ d logit_v(answer position) / d t_k ] )
for every node token v and latent pass k, where t_k is the recycled thought
(treated as an independent variable: detached leaf, matching the J-lens's
"effect of the activation" semantics). Fit set: train[500:2500] — disjoint from
the measurement slice (train[:100]) and the probe holdout (last 500). One pinned
serialization per graph (see thoughts.py). Gradient sign kept.

Writes results/<run>/jlens_basis.pt  (tensor (K, vocab, 768); zero rows = none)
   and results/<run>/jlens_basis_report.json (concentration + cosine tables).
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

import torch  # noqa: E402

from harness import VENDOR, Runner  # noqa: E402  (inserts VENDOR on sys.path)
from dataset import expand_data  # noqa: E402

FIT_START, FIT_END = 500, 2500


def graph_nodes(sample):
    nodes = {sample["root"], sample["target"], sample["neg_target"]}
    for s, t in sample["edges"]:
        nodes.add(s)
        nodes.add(t)
    return nodes


def forward_answer_logits(runner, sample):
    """One grad-enabled forward; returns (answer-position logits, {k: leaf})
    where each leaf is the pass-k recycled thought as an independent variable."""
    max_steps = len(sample["steps"])
    question, _ = expand_data(sample, max_steps + 1, max_steps, neg_sampling=False)
    ids = runner.tok.encode(question)
    input_ids = torch.tensor([ids], device=runner.device)
    n = len(ids)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "position_ids": torch.arange(n, device=runner.device).reshape(1, -1),
    }
    leaves = {}

    def edit(k, v):
        leaf = v.detach().clone().requires_grad_(True)
        leaves[k] = leaf
        return leaf

    out = runner.model(**batch, thought_edit=edit)
    return out.logits[0, -1], leaves


def per_node_grads(logits, leaves, nodes, batched_ok):
    """{(v, k): grad tensor}; tries one batched-vmap call, falls back to loop."""
    leaf_list = [leaves[k] for k in sorted(leaves)]
    ks = sorted(leaves)
    out = {}
    if batched_ok:
        node_logits = logits[nodes]
        grads = torch.autograd.grad(
            node_logits,
            leaf_list,
            grad_outputs=torch.eye(len(nodes), device=logits.device),
            is_grads_batched=True,
            retain_graph=False,
        )
        for j, k in enumerate(ks):
            for i, v in enumerate(nodes):
                out[(v, k)] = grads[j][i].detach()
        return out
    for i, v in enumerate(nodes):
        last = i == len(nodes) - 1
        grads = torch.autograd.grad(logits[v], leaf_list, retain_graph=not last)
        for j, k in enumerate(ks):
            out[(v, k)] = grads[j].detach()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="seed0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = ROOT / "results" / args.run_name
    runner = Runner(
        ROOT / "ckpts" / args.run_name / "best.pt", device=args.device
    )
    runner.model.eval()

    data = json.load(open(VENDOR / "data/prosqa_train_graph_4_coconut.json"))
    fit = data[FIT_START:FIT_END]
    nodes = sorted(set().union(*(graph_nodes(s) for s in fit)))
    k_max = max(len(s["steps"]) for s in fit)
    vocab = runner.wte.shape[0]
    dev = runner.device
    print(f"fit graphs {FIT_START}:{FIT_END}, {len(nodes)} node tokens, K={k_max}")

    gsum = torch.zeros(k_max, vocab, 768, device=dev)
    usum = torch.zeros(k_max, vocab, 768, device=dev)
    kcount = torch.zeros(k_max, device=dev)

    batched_ok = True
    for i, sample in enumerate(fit):
        gi = FIT_START + i
        sample["edges"] = [list(e) for e in sample["edges"]]
        edges0 = [list(e) for e in sample["edges"]]
        sample["edges"] = edges0
        random.seed((gi << 16) ^ args.seed)

        logits, leaves = forward_answer_logits(runner, sample)
        try:
            grads = per_node_grads(logits, leaves, nodes, batched_ok)
        except RuntimeError as e:
            if not batched_ok:
                raise
            print(f"batched grads failed ({e}); falling back to per-node loop")
            batched_ok = False
            logits, leaves = forward_answer_logits(runner, sample)
            grads = per_node_grads(logits, leaves, nodes, batched_ok)

        for k in leaves:
            kcount[k] += 1
        for (v, k), g in grads.items():
            gsum[k, v] += g
            usum[k, v] += g / g.norm().clamp_min(1e-12)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(fit)}")

    basis = torch.zeros(k_max, vocab, 768)
    probe_file = out_dir / "probe_basis.pt"
    probes = (
        torch.load(probe_file, map_location="cpu", weights_only=True)
        if probe_file.exists()
        else None
    )
    report = {"per_node": {}, "per_step": {}}
    for k in range(k_max):
        n_k = int(kcount[k].item())
        cos_wte, cos_probe, conc = [], [], []
        for v in nodes:
            mean_g = gsum[k, v] / max(n_k, 1)
            if mean_g.norm() == 0:
                continue
            j = (mean_g / mean_g.norm()).cpu()
            basis[k, v] = j
            c = float((usum[k, v] / max(n_k, 1)).norm())
            w = runner.wte[v].detach().cpu()
            cw = float(j @ (w / w.norm()))
            cp = (
                float(j @ probes[v])
                if probes is not None and probes[v].norm() > 0
                else None
            )
            conc.append(c)
            cos_wte.append(cw)
            if cp is not None:
                cos_probe.append(cp)
            report["per_node"][f"v{v}_k{k}"] = {
                "concentration": round(c, 4),
                "cos_wte": round(cw, 4),
                "cos_probe": round(cp, 4) if cp is not None else None,
            }
        report["per_step"][str(k)] = {
            "n_graphs": n_k,
            "median_concentration": round(statistics.median(conc), 4),
            "median_abs_cos_wte": round(
                statistics.median(abs(x) for x in cos_wte), 4
            ),
            "median_abs_cos_probe": (
                round(statistics.median(abs(x) for x in cos_probe), 4)
                if cos_probe
                else None
            ),
        }

    summary = {
        "fit_slice": [FIT_START, FIT_END],
        "n_node_tokens": len(nodes),
        "k_max": k_max,
        "seed": args.seed,
        "grads_batched": batched_ok,
        "per_step": report["per_step"],
        "per_node": report["per_node"],
    }
    torch.save(basis, out_dir / "jlens_basis.pt")
    with open(out_dir / "jlens_basis_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(report["per_step"], indent=2))
    print(f"written: {out_dir / 'jlens_basis.pt'}")


if __name__ == "__main__":
    main()
