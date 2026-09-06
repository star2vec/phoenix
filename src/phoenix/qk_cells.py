"""Query-key subtraction cells for experiment 3 (predictions in NOTES.md).

The answer's edge is the edge (p, target) with p at depth K-1. It is read by
the layer-2 attention of the query built from thought K-1 (the last
intermediate thought; pass K-2; latent position K-1). For each layer-2 head
the direction in thought space that the head's query matrix maps onto that
edge's key is

    u_h = unit( center( gain_ln * (W_Q^h @ k_h) ) )

where k_h is the head's key averaged over the edge's tokens with the head's
own attention as weights, W_Q^h the head's query columns of the layer's
c_attn weight, and gain_ln the weight of the layer norm in front of the
attention. This is the first-order direction: it ignores the layer norm's
rescaling by the residual norm and layer 1's indirect response to a changed
thought.

Cells (all edits at pass K-2, norm preserved as in the paper's SUBTRACT):
  qk_subtract/answer_edge/head<h>      remove u_h
  qk_subtract/answer_edge/all_heads    remove the span of u_0..u_7
  qk_subtract/random_matched/head<h>   same coefficient along a random unit
  qk_subtract/random_matched/all_heads same coefficients along random
                                       orthonormal directions
  qk_subtract/nonanswer_edge/head<h>, /all_heads   the same construction for
                                       a control edge (the frontier edge with
                                       the most attention that does not lead
                                       to the target, else the most attended
                                       other edge)
Every cell records each head's attention from that query onto the answer's
edge and onto the control edge, before and after.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from attn_hooks import AttnHooks  # noqa: E402
from edits import rand_orthonormal  # noqa: E402
from measure import answer_split, at_passes, run_ids  # noqa: E402

LAYER = 1  # layer 2 (0-based block index)


def slot_tokens(slot):
    return [p for p in slot if p is not None]


def head_direction(WQ, gain, h, hd, key_vec):
    """Unit direction in pre-layer-norm residual space that raises head h's
    attention logit on key_vec (first order)."""
    d = WQ[:, h * hd:(h + 1) * hd] @ key_vec
    d = d * gain
    d = d - d.mean()
    return d / d.norm().clamp_min(1e-12)


def weighted_key(kv_key, attn_row, positions):
    """Key of an edge for one head: its tokens' keys averaged with the head's
    attention on them (uniform when the head ignores the edge)."""
    w = attn_row[positions].clone()
    w = w / w.sum() if float(w.sum()) > 1e-8 else torch.full_like(w, 1.0 / len(positions))
    return (kv_key[positions, :] * w[:, None]).sum(0)


def recorded_run(runner, ids, thought_edit=None):
    with AttnHooks(runner.model.base_causallm) as h:
        h.record_weights = True
        h.record_kv = True
        logits = run_ids(runner, ids, thought_edit, attn_eager=True)
        weights, kv = h.weights, dict(h.kv)
    rec = AttnHooks.__new__(AttnHooks)
    rec.weights = weights
    return logits, rec, kv


def attention_on(rec, q, slot, n_heads):
    """Per-head attention mass from query position q onto a slot's tokens."""
    w = rec.attention(LAYER, q)  # (heads, k_len)
    return [float(w[h, slot_tokens(slot)].sum()) for h in range(n_heads)]


def sub_dirs(units, rand=None):
    """Norm-preserving removal of the span of `units` (list of unit vectors);
    with `rand` (orthonormal (d, m)), the same coefficients are removed along
    those directions instead (the matched random control)."""
    Q, _ = torch.linalg.qr(torch.stack(units, dim=1))
    Q = Q[:, :len(units)]

    def f(k, t):
        c = Q.T @ t.to(Q)
        t2 = t - ((rand if rand is not None else Q) @ c).to(t)
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return f


def qk_cells(runner, pr, own, base_seed, gi, wte, gen):
    """Returns (cells, meta). cells: name -> answer split with dT against the
    eager baseline and the attention records."""
    base = runner.model.base_causallm
    blk = base.transformer.h[LAYER]
    n_heads, hd = blk.attn.num_heads, blk.attn.head_dim
    WQ = blk.attn.c_attn.weight[:, :blk.attn.embed_dim].detach()
    gain = blk.ln_1.weight.detach()
    K, L = pr.K, pr.layout()
    q = L["latents"][K - 2]        # latent K-1 holds thought K-1
    edit_pass = K - 2
    ids = pr.ids(runner.tok)
    d = pr.depths()

    logits0, rec0, kv0 = recorded_run(runner, ids)
    base_split = answer_split(logits0, pr.target, pr.decoy)
    key0 = kv0[LAYER][0]  # (heads, k_len, hd)

    # answer edge: the most attended parent edge
    parents = pr.parent_slots()
    if not parents:
        return {}, {"skipped": True, "reason": "no_parent_edge_at_depth_K-1"}
    tot = lambda j: sum(attention_on(rec0, q, L["slots"][j], n_heads))
    ans = max(parents, key=tot)
    # control edge: most attended frontier edge not into the target, else any other
    frontier = [j for j, (s_, t) in enumerate(pr.edges)
                if d.get(s_) == K - 1 and t != pr.target and j not in parents]
    others = [j for j in range(len(pr.edges)) if j not in parents]
    ctrl = max(frontier, key=tot) if frontier else max(others, key=tot)
    ctrl_is_frontier = bool(frontier)

    def directions(slot_j):
        slot = L["slots"][slot_j]
        pos = slot_tokens(slot)
        w = rec0.attention(LAYER, q)
        return [head_direction(WQ, gain, h, hd, weighted_key(key0[h], w[h], pos)) for h in range(n_heads)]

    u_ans = directions(ans)
    u_ctrl = directions(ctrl)
    t_edit = own[edit_pass].to(u_ans[0])
    coef = [float(t_edit @ u) / float(t_edit.norm()) for u in u_ans]
    U = torch.stack(u_ans)
    cos = U @ U.T
    off = cos[~torch.eye(n_heads, dtype=bool)]
    sv = torch.linalg.svdvals(U)
    p_node = pr.edges[ans][0]
    unit = lambda v: v / v.norm()
    meta = {
        "query_position": q, "edit_pass": edit_pass, "answer_slot": ans, "control_slot": ctrl,
        "control_is_frontier_edge": ctrl_is_frontier, "n_parent_edges": len(parents),
        "coef_frac_per_head": coef,
        "mean_abs_cos_between_heads": float(off.abs().mean()),
        "singular_values": [round(float(x), 3) for x in sv],
        "cos_to_source_embedding": [float(u @ unit(wte[p_node].to(u))) for u in u_ans],
        "cos_to_target_embedding": [float(u @ unit(wte[pr.target].to(u))) for u in u_ans],
        "baseline_eager": base_split,
        "attn_answer_before": attention_on(rec0, q, L["slots"][ans], n_heads),
        "attn_control_before": attention_on(rec0, q, L["slots"][ctrl], n_heads),
    }

    cells = {}

    def run_cell(name, edit, head=None):
        logits, rec, _ = recorded_run(runner, ids, at_passes(edit, [edit_pass]))
        sp = answer_split(logits, pr.target, pr.decoy)
        sp["dT"] = sp["T"] - base_split["T"]
        a_after = attention_on(rec, q, L["slots"][ans], n_heads)
        c_after = attention_on(rec, q, L["slots"][ctrl], n_heads)
        sp["attn_answer_total_before"] = sum(meta["attn_answer_before"])
        sp["attn_answer_total_after"] = sum(a_after)
        sp["attn_control_total_before"] = sum(meta["attn_control_before"])
        sp["attn_control_total_after"] = sum(c_after)
        if head is not None:
            sp["attn_answer_head_before"] = meta["attn_answer_before"][head]
            sp["attn_answer_head_after"] = a_after[head]
        sp["attn_answer_after_per_head"] = a_after
        cells[name] = sp

    rand_units = [rand_orthonormal(1, U.shape[1], gen, U.device)[:, 0] for _ in range(n_heads)]
    rand_span = rand_orthonormal(n_heads, U.shape[1], gen, U.device)
    for h in range(n_heads):
        run_cell(f"qk_subtract/answer_edge/head{h}", sub_dirs([u_ans[h]]), head=h)
        run_cell(f"qk_subtract/random_matched/head{h}", sub_dirs([u_ans[h]], rand=rand_units[h][:, None]), head=h)
        run_cell(f"qk_subtract/nonanswer_edge/head{h}", sub_dirs([u_ctrl[h]]), head=h)
    run_cell("qk_subtract/answer_edge/all_heads", sub_dirs(u_ans))
    run_cell("qk_subtract/random_matched/all_heads", sub_dirs(u_ans, rand=rand_span))
    run_cell("qk_subtract/nonanswer_edge/all_heads", sub_dirs(u_ctrl))
    return cells, meta


def qk_attention_summary(rows, cell_names):
    """Mean attention on the answer edge before and after, per qk cell."""
    import numpy as np
    from stats import bootstrap
    out = {}
    for name in cell_names:
        if not name.startswith("qk_subtract"):
            continue
        rs = [r["cells"][name] for r in rows if not r["cells"].get(name, {}).get("skipped")]
        if not rs:
            continue
        entry = {
            "n": len(rs),
            "attn_answer_total_before": bootstrap([r["attn_answer_total_before"] for r in rs], np.mean),
            "attn_answer_total_after": bootstrap([r["attn_answer_total_after"] for r in rs], np.mean),
            "attn_answer_total_drop": bootstrap([r["attn_answer_total_before"] - r["attn_answer_total_after"] for r in rs], np.mean),
            "attn_control_total_before": bootstrap([r["attn_control_total_before"] for r in rs], np.mean),
            "attn_control_total_after": bootstrap([r["attn_control_total_after"] for r in rs], np.mean),
        }
        if "attn_answer_head_before" in rs[0]:
            entry["attn_answer_head_drop"] = bootstrap(
                [r["attn_answer_head_before"] - r["attn_answer_head_after"] for r in rs], np.mean)
        out[name] = entry
    return out
