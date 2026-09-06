"""Experiment 2: which heads look by position and which by content.

Two runs per graph: A = the pinned original; B = the edge list reordered with
the original thoughts fixed at all K passes (so every latent query is built
from the same thought). For each layer, head and query class the attention
over edge slots is compared between A and B:

  position score  correlation slot for slot (same slot, whatever edge is there)
  content score   correlation edge for edge (the same edge, wherever it moved)

Query classes: root (the token query that computes thought 1), intermediate
latents (thoughts 1..K-1 as queries), last latent (thought K as query), the
answer position. Edge-token and separator-token queries are summarised
separately by how much of their attention stays inside their own slot (Zhu
et al.'s layer-1 copy). For every head and class the mass on the first and
last slot says where a positional head looks.

    python src/phoenix/heads.py --run-name seed0 --device mps --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attn_hooks import AttnHooks  # noqa: E402
from common import (  # noqa: E402
    covariates, finish, graph_rng, header, load_runner, make_parser,
    recipient_prompts,
)
from measure import all_passes, capture, fixed, run_ids  # noqa: E402
from prompts import reorder  # noqa: E402
from stats import bootstrap  # noqa: E402

CLASSES = ("root", "intermediate_latent", "last_latent", "answer")


def slot_mass(w, slots):
    """(heads, E): attention mass per edge slot (sum over the slot's tokens)."""
    cols = []
    for ps, pt, sep in slots:
        idx = [ps, pt] + ([sep] if sep is not None else [])
        cols.append(w[:, idx].sum(-1))
    return torch.stack(cols, dim=1)


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def query_positions(L, K):
    return {
        "root": [L["root"]],
        "intermediate_latent": L["latents"][: K - 1],
        "last_latent": [L["latents"][K - 1]],
        "answer": [L["a"]],
    }


def record(runner, ids, thought_edit=None):
    with AttnHooks(runner.model.base_causallm) as h:
        h.record_weights = True
        logits = run_ids(runner, ids, thought_edit, attn_eager=True)
        weights = h.weights
    return logits, _Recorded(weights)


class _Recorded:
    def __init__(self, weights):
        self.weights = weights

    def attention(self, layer, q_abs):
        for rec in self.weights:
            if rec["layer"] != layer:
                continue
            q_len = rec["w"].shape[1]
            if rec["offset"] <= q_abs < rec["offset"] + q_len:
                return rec["w"][:, q_abs - rec["offset"], :]
        raise KeyError((layer, q_abs))


def run(runner, recips, base_seed=0):
    n_layers = len(runner.model.base_causallm.transformer.h)
    n_heads = runner.model.base_causallm.transformer.h[0].attn.num_heads
    rows = []
    for gi, sample, pr in recips:
        K, L = pr.K, pr.layout()
        E = len(pr.edges)
        ids = pr.ids(runner.tok)
        own = capture(runner, ids)
        rb, meta = reorder(pr, graph_rng(base_seed, gi))
        perm = meta["perm"]
        _, A = record(runner, ids)
        _, B = record(runner, rb.ids(runner.tok), fixed(own, all_passes(K)))
        qpos = query_positions(L, K)

        per = {}  # (layer, head, class) -> dict of lists
        for layer in range(n_layers):
            for cls, qs in qpos.items():
                for q in qs:
                    wa = slot_mass(A.attention(layer, q), L["slots"])  # (heads, E)
                    wb = slot_mass(B.attention(layer, q), L["slots"])
                    wb_by_edge = wb[:, perm]  # column j = mass on the slot where edge j went
                    for h in range(n_heads):
                        d = per.setdefault((layer, h, cls), {"pos": [], "con": [], "mass_a": [], "mass_b": []})
                        d["pos"].append(corr(wa[h], wb[h]))
                        d["con"].append(corr(wa[h], wb_by_edge[h]))
                        d["mass_a"].append(float(wa[h].sum()))
                        d["mass_b"].append(float(wb[h].sum()))
            # edge-token queries: attention kept inside the own slot (A vs B),
            # from the target token and from the separator token (which can
            # see both endpoints of its edge)
            for qname, qidx in (("edge_token", 1), ("edge_sep", 2)):
                own_a, own_b = [], []
                for j, slot in enumerate(L["slots"]):
                    q = slot[qidx]
                    if q is None:
                        continue
                    slot_b = L["slots"][perm[j]]
                    qb = slot_b[qidx]
                    if qb is None:
                        continue
                    wa = A.attention(layer, q)
                    wb = B.attention(layer, qb)
                    own_a.append(wa[:, [slot[0], slot[1]]].sum(-1))
                    own_b.append(wb[:, [slot_b[0], slot_b[1]]].sum(-1))
                own_a = torch.stack(own_a).mean(0)  # (heads,)
                own_b = torch.stack(own_b).mean(0)
                for h in range(n_heads):
                    per[(layer, h, qname)] = {"own_slot_a": float(own_a[h]), "own_slot_b": float(own_b[h])}
            # where a head looks by slot index: mass on the first and last slot
            for cls, qs in qpos.items():
                for q in qs:
                    wa = slot_mass(A.attention(layer, q), L["slots"])
                    for h in range(n_heads):
                        d = per[(layer, h, cls)]
                        d.setdefault("first_slot", []).append(float(wa[h, 0]))
                        d.setdefault("last_slot", []).append(float(wa[h, -1]))

        def mean_or_none(xs):
            xs = [x for x in xs if x is not None]
            return float(np.mean(xs)) if xs else None

        heads_row = {}
        for (layer, h, cls), d in per.items():
            key = f"L{layer + 1}H{h}/{cls}"
            if cls in ("edge_token", "edge_sep"):
                heads_row[key] = d
            else:
                heads_row[key] = {
                    "position_score": mean_or_none(d["pos"]),
                    "content_score": mean_or_none(d["con"]),
                    "slot_mass_a": mean_or_none(d["mass_a"]),
                    "slot_mass_b": mean_or_none(d["mass_b"]),
                    "first_slot": mean_or_none(d.get("first_slot", [])),
                    "last_slot": mean_or_none(d.get("last_slot", [])),
                }
        rows.append({"gi": gi, "K": K, "n_edges": E, "cov": covariates(pr), "moved": meta["moved"], "heads": heads_row})
        print(f"graph {gi}: done ({E} edges, K={K})")

    # summary: bootstrap over graphs of each per-graph mean
    summary = {}
    keys = sorted({k for r in rows for k in r["heads"]})
    for key in keys:
        entry = {}
        fields = ("own_slot_a", "own_slot_b") if key.endswith(("edge_token", "edge_sep")) else (
            "position_score", "content_score", "slot_mass_a", "slot_mass_b", "first_slot", "last_slot")
        for f in fields:
            vals = [r["heads"][key][f] for r in rows if r["heads"][key].get(f) is not None]
            entry[f] = bootstrap(vals, np.mean) if vals else None
        summary[key] = entry
    return {"rows": rows, "summary": summary, "classes": list(CLASSES) + ["edge_token", "edge_sep"],
            "n_layers": n_layers, "n_heads": n_heads}


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "heads")
    result.update(run(runner, recips, args.seed))
    finish(args, "heads", result)


if __name__ == "__main__":
    main()
