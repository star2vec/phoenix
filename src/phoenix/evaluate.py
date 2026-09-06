"""Held-out evaluation of a trained checkpoint.

(a) Held-out test accuracy on the paper's 419 test problems, computed
    with the vendor's own extraction (decode, split on "[A]") for fidelity.
(b) BFS-wave inner-product readout replicates: mean <thought_i, u_v> per node
    group per step, groups per the paper's Fig. 6 / Table 5:
      NotReachable: BFS depth(v) > i (incl. unreachable)
      Reachable:    depth(v) <= i
      Frontier:     depth(v) == i        (subset of Reachable)
      Optimal:      neighbor_k[str(i)]   (depth-i nodes on valid root->target
                                          paths; subset of Frontier)
    Expected ordering per step: Optimal > Frontier > Reachable-mean and
    NotReachable ~ 0.

Writes results/<run>/evaluation.json. The per-step ordering booleans are
reported alongside the group means.
"""

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "reasoning-by-superposition"
sys.path.insert(0, str(VENDOR))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from dataset import MyCollator, expand_data

from fast_coconut import FastCoconut, FastSTokenizer


def bfs_depths(edges, root):
    adj = {}
    for s, t in edges:
        adj.setdefault(s, []).append(t)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="seed0")
    p.add_argument("--checkpoint", default=None, help="default: ckpts/<run>/best.pt")
    p.add_argument("--device", default="mps")
    # the vendor builder draws each graph's edge order and candidate order from
    # the unseeded global RNG, so two runs never see the same prompts; this pins
    # graph gi to ((gi + 1000000) << 16) ^ seed, the namespace sets.test_pin uses
    p.add_argument("--serialization-seed", type=int, default=None)
    p.add_argument("--out-name", default=None,
                   help="default: evaluation.json, or evaluation_ser<seed>.json when seeded")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = args.checkpoint or ROOT / "ckpts" / args.run_name / "best.pt"
    results_dir = ROOT / "results" / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = FastSTokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    base = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(str(VENDOR / "configs/symbol-2layer-8head-768dim.json"))
    )
    model = FastCoconut(
        base,
        latent_id,
        tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        tokenizer.eos_token_id,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    test_data = json.load(open(VENDOR / "data/prosqa_test_graph_4_coconut.json"))
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    wte = base.transformer.wte.weight  # (40, 768); node tokens are ids 0..30

    # ---- (a) test accuracy, vendor-style extraction --------------------
    cor, total = 0, 0
    # ---- (b) readout accumulators: step -> group -> list of inner prods -
    groups = ["NotReachable", "Reachable", "Frontier", "Optimal"]
    acc_readout = {i: {g: [] for g in groups} for i in range(1, 5)}

    for gi, sample in enumerate(test_data):
        max_steps = len(sample["steps"])
        if args.serialization_seed is not None:
            random.seed(((gi + 1_000_000) << 16) ^ args.serialization_seed)
        question, _ = expand_data(sample, max_steps + 1, max_steps, neg_sampling=False)
        ids = tokenizer.encode(question, add_special_tokens=False)
        batch = collator(
            [{"input_ids": ids, "attention_mask": [1] * len(ids),
              "position_ids": list(range(len(ids))), "idx": 0}]
        )
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if v is not None and k not in ["idx", "position_ids"]
        }

        outputs = model.generate(**batch, max_new_tokens=1, output_embedding=True)
        tokens, inputs_embeds = outputs

        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        text = text.replace("<eos>", "").strip()
        pred = text.split("[A]")[-1].replace(",", "").strip()
        cor += pred == str(sample["target"])
        total += 1

        # thoughts: recycled embeddings at latent positions (in order)
        latent_pos = [j for j, t in enumerate(ids) if t == latent_id]
        depth = bfs_depths(sample["edges"], sample["root"])
        n_nodes = len(sample["idx_to_symbol"])
        node_embs = wte[:n_nodes]  # (n_nodes, 768)

        for i, pos in enumerate(latent_pos, start=1):
            thought = inputs_embeds[0, pos]  # (768,)
            sims = (node_embs @ thought).float().cpu().tolist()
            optimal = set(sample["neighbor_k"].get(str(i), []))
            for v in range(n_nodes):
                d = depth.get(v, None)
                if d is None or d > i:
                    acc_readout[i]["NotReachable"].append(sims[v])
                else:
                    acc_readout[i]["Reachable"].append(sims[v])
                    if d == i:
                        acc_readout[i]["Frontier"].append(sims[v])
                        if v in optimal:
                            acc_readout[i]["Optimal"].append(sims[v])

    accuracy = cor / total
    readout = {}
    for i in range(1, 5):
        readout[i] = {}
        for g in groups:
            vals = acc_readout[i][g]
            if vals:
                m = sum(vals) / len(vals)
                sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
                readout[i][g] = {"mean": round(m, 3), "std": round(sd, 3), "n": len(vals)}

    ordering_ok = {}
    for i, row in readout.items():
        if all(g in row for g in groups):
            ordering_ok[i] = (
                row["Optimal"]["mean"] > row["Frontier"]["mean"] >= row["Reachable"]["mean"]
                and row["Reachable"]["mean"] > row["NotReachable"]["mean"]
                and abs(row["NotReachable"]["mean"]) < 1.0
            )

    result = {
        "checkpoint": str(ckpt),
        "serialization_seed": args.serialization_seed,
        "test_accuracy": round(accuracy, 6),
        "test_cor": cor,
        "test_total": total,
        "readout_by_step": readout,
        "readout_ordering_ok_by_step": ordering_ok,
        "paper_reference": "Table 5 / Fig. 6 of arXiv:2505.12514v3",
    }
    out = results_dir / (args.out_name or (
        "evaluation.json" if args.serialization_seed is None
        else f"evaluation_ser{args.serialization_seed}.json"))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"test accuracy: {cor}/{total} = {accuracy:.4f}")
    for i, row in readout.items():
        line = " | ".join(f"{g} {row[g]['mean']:+.2f}" for g in groups if g in row)
        print(f"step {i}: {line}  ordering_ok={ordering_ok.get(i)}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
