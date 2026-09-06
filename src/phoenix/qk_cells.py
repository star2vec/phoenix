"""Query-key subtraction cells for experiment 3 (predictions in NOTES.md).

For a query built from thought k+1 (latent position k+1; pass k), each
layer-2 head h has a direction in thought space that its query matrix maps
onto a chosen edge's key:

    u_h = unit( center( gain_ln * (W_Q^h @ k_h) ) )

where k_h is the head's key averaged over the edge's tokens with the head's
own attention as weights, W_Q^h the head's query columns of the layer's
c_attn weight, and gain_ln the weight of the layer norm in front of the
attention. First-order: it ignores the layer norm's rescaling by the residual
norm and layer 1's indirect response to a changed thought.

Last-step cells (edit at pass K-2, the thought that reads the parent edge):
  qk_subtract/answer_edge/head<h>, /all_heads
  qk_subtract/random_matched/head<h>, /all_heads
  qk_subtract/nonanswer_edge/head<h>, /all_heads
Every-step cells (edit at every pass k = 0..K-2; at step k the edge is the
most attended shortest-path edge from depth k+1 to depth k+2):
  qk_every_step/answer_path/head<h>, /all_heads
  qk_every_step/random_matched/all_heads
  qk_every_step/nonpath_edge/all_heads  (skipped when some step has no
                                         frontier edge off the answer path)
Every cell records each head's attention from the edited step's query onto
the answer edge and the control edge, before and after; every-step cells
record the per-step attention on the answer-path edge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from attn_hooks import AttnHooks  # noqa: E402
from edits import rand_orthonormal  # noqa: E402
from measure import answer_split, run_ids  # noqa: E402
from prompts import on_path_nodes  # noqa: E402

LAYER = 1  # layer 2 (0-based block index)


def slot_tokens(slot):
    return [p for p in slot if p is not None]


def head_direction(WQ, gain, h, hd, key_vec):
    d = WQ[:, h * hd:(h + 1) * hd] @ key_vec
    d = d * gain
    d = d - d.mean()
    return d / d.norm().clamp_min(1e-12)


def weighted_key(kv_key, attn_row, positions):
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
    w = rec.attention(LAYER, q)
    return [float(w[h, slot_tokens(slot)].sum()) for h in range(n_heads)]


def span_removal(units, rand=None):
    """Norm-preserving removal of the span of `units`; with `rand` the same
    coefficients are removed along those orthonormal directions instead."""
    Q, _ = torch.linalg.qr(torch.stack(units, dim=1))
    Q = Q[:, :len(units)]

    def f(t):
        c = Q.T @ t.to(Q)
        t2 = t - ((rand if rand is not None else Q) @ c).to(t)
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return f


def per_pass_edit(edits_by_pass):
    def f(k, t):
        e = edits_by_pass.get(k)
        return e(t) if e is not None else t
    return f


class _Geometry:
    """Shared baseline run and direction builder for one graph."""

    def __init__(self, runner, pr, ids):
        base = runner.model.base_causallm
        blk = base.transformer.h[LAYER]
        self.n_heads, self.hd = blk.attn.num_heads, blk.attn.head_dim
        self.WQ = blk.attn.c_attn.weight[:, :blk.attn.embed_dim].detach()
        self.gain = blk.ln_1.weight.detach()
        self.L = pr.layout()
        logits0, self.rec0, kv0 = recorded_run(runner, ids)
        self.base_split = answer_split(logits0, pr.target, pr.decoy)
        self.key0 = kv0[LAYER][0]

    def total_attention(self, q, slot_j):
        return sum(attention_on(self.rec0, q, self.L["slots"][slot_j], self.n_heads))

    def directions(self, q, slot_j):
        pos = slot_tokens(self.L["slots"][slot_j])
        w = self.rec0.attention(LAYER, q)
        return [head_direction(self.WQ, self.gain, h, self.hd, weighted_key(self.key0[h], w[h], pos))
                for h in range(self.n_heads)]


def qk_cells(runner, pr, own, base_seed, gi, wte, gen):
    """Last-step and every-step query-key cells. Returns (cells, meta)."""
    K, ids = pr.K, pr.ids(runner.tok)
    d = pr.depths()
    G = _Geometry(runner, pr, ids)
    n_heads, L = G.n_heads, G.L
    on = on_path_nodes(pr)
    cells, meta = {}, {"baseline_eager": G.base_split}

    # ---- last step: the parent edge, read by the query at latent K-1 ----
    q = L["latents"][K - 2]
    edit_pass = K - 2
    parents = pr.parent_slots()
    if not parents:
        return {}, {"skipped": True, "reason": "no_parent_edge_at_depth_K-1"}
    ans = max(parents, key=lambda j: G.total_attention(q, j))
    frontier = [j for j, (s_, t) in enumerate(pr.edges)
                if d.get(s_) == K - 1 and t != pr.target and j not in parents]
    others = [j for j in range(len(pr.edges)) if j not in parents]
    ctrl = max(frontier, key=lambda j: G.total_attention(q, j)) if frontier else max(others, key=lambda j: G.total_attention(q, j))
    u_ans, u_ctrl = G.directions(q, ans), G.directions(q, ctrl)
    t_edit = own[edit_pass].to(u_ans[0])
    U = torch.stack(u_ans)
    unit = lambda v: v / v.norm()
    p_node = pr.edges[ans][0]
    meta.update({
        "query_position": q, "edit_pass": edit_pass, "answer_slot": ans, "control_slot": ctrl,
        "control_is_frontier_edge": bool(frontier), "n_parent_edges": len(parents),
        "coef_frac_per_head": [float(t_edit @ u) / float(t_edit.norm()) for u in u_ans],
        "mean_abs_cos_between_heads": float((U @ U.T)[~torch.eye(n_heads, dtype=bool)].abs().mean()),
        "singular_values": [round(float(x), 3) for x in torch.linalg.svdvals(U)],
        "cos_to_source_embedding": [float(u @ unit(wte[p_node].to(u))) for u in u_ans],
        "cos_to_target_embedding": [float(u @ unit(wte[pr.target].to(u))) for u in u_ans],
        "attn_answer_before": attention_on(G.rec0, q, L["slots"][ans], n_heads),
        "attn_control_before": attention_on(G.rec0, q, L["slots"][ctrl], n_heads),
    })

    def run_cell(name, edits_by_pass, head=None):
        logits, rec, _ = recorded_run(runner, ids, per_pass_edit(edits_by_pass))
        sp = answer_split(logits, pr.target, pr.decoy)
        sp["dT"] = sp["T"] - G.base_split["T"]
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
        return rec

    rand_units = [rand_orthonormal(1, U.shape[1], gen, U.device)[:, 0] for _ in range(n_heads)]
    rand_span = rand_orthonormal(n_heads, U.shape[1], gen, U.device)
    for h in range(n_heads):
        run_cell(f"qk_subtract/answer_edge/head{h}", {edit_pass: span_removal([u_ans[h]])}, head=h)
        run_cell(f"qk_subtract/random_matched/head{h}", {edit_pass: span_removal([u_ans[h]], rand=rand_units[h][:, None])}, head=h)
        run_cell(f"qk_subtract/nonanswer_edge/head{h}", {edit_pass: span_removal([u_ctrl[h]])}, head=h)
    run_cell("qk_subtract/answer_edge/all_heads", {edit_pass: span_removal(u_ans)})
    run_cell("qk_subtract/random_matched/all_heads", {edit_pass: span_removal(u_ans, rand=rand_span)})
    run_cell("qk_subtract/nonanswer_edge/all_heads", {edit_pass: span_removal(u_ctrl)})

    # ---- every intermediate step: the answer-path edge read at each step ----
    steps = []  # per pass k: dict(query, slot, ctrl_slot or None, dirs, ctrl_dirs)
    for k in range(K - 1):
        qk = L["latents"][k]
        path_edges = [j for j, (s_, t) in enumerate(pr.edges)
                      if on.get(s_) == k + 1 and on.get(t) == k + 2]
        if not path_edges:
            steps = None
            break
        slot = max(path_edges, key=lambda j: G.total_attention(qk, j))
        off_path = [j for j, (s_, t) in enumerate(pr.edges)
                    if d.get(s_) == k + 1 and not (on.get(s_) == k + 1 and on.get(t) == k + 2)]
        cslot = max(off_path, key=lambda j: G.total_attention(qk, j)) if off_path else None
        steps.append({"pass": k, "query": qk, "slot": slot, "ctrl_slot": cslot,
                      "dirs": G.directions(qk, slot),
                      "ctrl_dirs": G.directions(qk, cslot) if cslot is not None else None,
                      "attn_before": attention_on(G.rec0, qk, L["slots"][slot], n_heads)})
    if steps is None:
        meta["every_step"] = {"skipped": True, "reason": "no_path_edge_at_some_step"}
        return cells, meta
    meta["every_step"] = {
        "slots": [st["slot"] for st in steps],
        "ctrl_slots": [st["ctrl_slot"] for st in steps],
        "attn_before_per_step": [sum(st["attn_before"]) for st in steps],
    }

    def run_every(name, edits_by_pass, head=None):
        rec = run_cell(name, edits_by_pass, head)
        after = [sum(attention_on(rec, st["query"], L["slots"][st["slot"]], n_heads)) for st in steps]
        cells[name]["attn_path_before_per_step"] = meta["every_step"]["attn_before_per_step"]
        cells[name]["attn_path_after_per_step"] = after
        cells[name]["attn_path_drop_mean"] = float(sum(b - a for b, a in zip(meta["every_step"]["attn_before_per_step"], after)) / len(steps))

    run_every("qk_every_step/answer_path/all_heads", {st["pass"]: span_removal(st["dirs"]) for st in steps})
    run_every("qk_every_step/random_matched/all_heads",
              {st["pass"]: span_removal(st["dirs"], rand=rand_orthonormal(n_heads, U.shape[1], gen, U.device)) for st in steps})
    if all(st["ctrl_dirs"] is not None for st in steps):
        run_every("qk_every_step/nonpath_edge/all_heads", {st["pass"]: span_removal(st["ctrl_dirs"]) for st in steps})
    else:
        cells["qk_every_step/nonpath_edge/all_heads"] = {"skipped": True, "reason": "no_off_path_edge_at_some_step"}
    for h in range(n_heads):
        run_every(f"qk_every_step/answer_path/head{h}", {st["pass"]: span_removal([st["dirs"][h]]) for st in steps}, head=h)
    return cells, meta


def qk_attention_summary(rows, cell_names):
    """Mean attention on the answer edge before and after, per qk cell; for
    every-step cells also the mean per-step drop on the answer-path edge."""
    import numpy as np
    from stats import bootstrap
    out = {}
    for name in cell_names:
        if not (name.startswith("qk_subtract") or name.startswith("qk_every_step")):
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
        if "attn_path_drop_mean" in rs[0]:
            entry["attn_path_drop_mean_over_steps"] = bootstrap([r["attn_path_drop_mean"] for r in rs], np.mean)
        out[name] = entry
    return out
