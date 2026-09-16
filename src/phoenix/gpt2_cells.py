"""Experiment 7: the from-scratch measurements on a fine-tuned GPT-2 COCONUT
model. Predictions (two lines) in NOTES.md.

Part `heads`: experiment 2's position-vs-content scoring (heads.run, all
layers) on natural-language prompts; writes results/<run>/heads_<mode>.json.

Part `cells`: reads that file, picks the latents route by the same cutoffs the
from-scratch runs used (a head is in the route if its mean edge-slot mass at
the search latents, passes 0..L-2, is at least EDGE_HEAD_CUT; fallback rungs
recorded), then per graph:
  standing controls   reserialized, self_transplant (exactly zero),
                      random_donor/intermediates, same_answer_donor/{intermediates,all}
  the removal         qk_removal: at each pass 0..L-2 the span of the route's
                      query-key directions onto the most attended answer-path
                      edge is removed from the recycled thought; qk_random:
                      the same number of random orthonormal directions
  calibration mask    latents_mask/{path,offpath}/alone: the route's heads
                      masked at all latent queries onto the path edges'
                      tokens (or a count-matched off-path set)
  carry-over          thoughtK/{same_answer,random}/{alone,plus_removal}:
                      the donor's final thought at pass K-1
K is 6 for every prompt (the model's fixed latent count); L is the solution
length. Cutoffs are named where used.

    python src/phoenix/gpt2_cells.py --run-name gpt2_dilgren --device cpu --mode pilot --part heads
    python src/phoenix/gpt2_cells.py --run-name gpt2_dilgren --device cpu --mode pilot --part cells
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import heads as heads_mod  # noqa: E402
from attn_hooks import AttnHooks  # noqa: E402
from common import covariates, finish, graph_gen, graph_rng, header, make_parser, summarize, with_delta  # noqa: E402
from edits import rand_orthonormal  # noqa: E402
from gpt2 import load_gpt2_runner, source_info  # noqa: E402
from measure import all_passes, answer_split, capture, fixed, intermediates, run_ids  # noqa: E402
from nl import donor_run_nl, random_donor_nl, recipient_prompts_nl, reserialized_nl, same_answer_donor_nl  # noqa: E402
from prompts import on_path_nodes  # noqa: E402
from qk_cells import head_direction, per_pass_edit, recorded_run, slot_tokens, span_removal, weighted_key  # noqa: E402
from recovery import fallback_line, DECOY_WEAKEN_CUT  # noqa: E402
from sets import ROOT, load_train, require_file  # noqa: E402
from stats import bootstrap  # noqa: E402

EDGE_HEAD_CUT = 0.5   # protects "this head reads edges" (experiment 5's cutoff)
CLASS_CUT = 0.2       # experiment 2's classification cutoff (second rung)
TOP_K = 8             # third rung, exploration: the from-scratch route's size
SEARCH_CLASS = "search_latent"


def head_set(heads_summary, n_layers, n_heads):
    """(route as [(layer, head)], rung, mass by head) by the recorded rungs."""
    mass = {}
    for l in range(n_layers):
        for h in range(n_heads):
            e = heads_summary.get(f"L{l + 1}H{h}/{SEARCH_CLASS}")
            mass[(l, h)] = e["slot_mass_a"]["point"] if e and e.get("slot_mass_a") else 0.0
    for rung, cut in (("edge_head_cut_0.5", EDGE_HEAD_CUT), ("class_cut_0.2", CLASS_CUT)):
        S = sorted(k for k, m in mass.items() if m >= cut)
        if S:
            return S, rung, mass
    return sorted(sorted(mass, key=lambda k: -mass[k])[:TOP_K]), "top8_exploration", mass


def head_name(l, h):
    return f"L{l + 1}H{h}"


class MultiGeometry:
    """One recorded baseline run; query-key directions for a route that may
    span several layers (the two-layer cell's construction, per layer)."""

    def __init__(self, runner, pr, ids, S):
        base = runner.model.base_causallm
        self.S = [tuple(x) for x in S]
        self.layers = sorted({l for l, _ in self.S})
        self.WQ, self.gain = {}, {}
        for l in self.layers:
            blk = base.transformer.h[l]
            self.WQ[l] = blk.attn.c_attn.weight[:, :blk.attn.embed_dim].detach()
            self.gain[l] = blk.ln_1.weight.detach()
        self.hd = base.transformer.h[0].attn.head_dim
        self.L = pr.layout()
        self.ro = pr.readout()
        logits0, self.rec0, kv0 = recorded_run(runner, ids)
        self.keys = {l: kv0[l][0] for l in self.layers}
        self.base_split = answer_split(logits0, self.ro["target"], self.ro["decoy"], node_ids=self.ro["nodes"])
        self.all_slot_tokens = [p for slot in self.L["slots"] for p in slot_tokens(slot)]

    def per_head(self, rec, q, positions):
        out, rows = {}, {}
        for l, h in self.S:
            if l not in rows:
                rows[l] = rec.attention(l, q)
            out[head_name(l, h)] = float(rows[l][h, positions].sum())
        return out

    def attention_on(self, q, slot_j, rec=None):
        return self.per_head(self.rec0 if rec is None else rec, q, slot_tokens(self.L["slots"][slot_j]))

    def total_attention(self, q, slot_j, rec=None):
        return sum(self.attention_on(q, slot_j, rec).values())

    def edge_mass(self, q, rec=None):
        return sum(self.per_head(self.rec0 if rec is None else rec, q, self.all_slot_tokens).values())

    def directions(self, q, slot_j):
        pos = slot_tokens(self.L["slots"][slot_j])
        dirs = []
        for l, h in self.S:
            w = self.rec0.attention(l, q)[h]
            dirs.append(head_direction(self.WQ[l], self.gain[l], h, self.hd, weighted_key(self.keys[l][h], w, pos)))
        return dirs


def path_and_offpath_slots(pr, G):
    """Answer-path edges (shortest-path edges from depth k+1 to k+2 at every
    search pass) and a count-matched set of the most attended off-path
    frontier edges."""
    on, d, L = on_path_nodes(pr), pr.depths(), G.L
    path, off = [], []
    for k in range(pr.L - 1):
        q = L["latents"][k]
        pe = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2]
        oe = [j for j, (s_, t) in enumerate(pr.edges) if d.get(s_) == k + 1 and j not in pe]
        oe = sorted(oe, key=lambda j: -G.total_attention(q, j))[:len(pe)]
        path += [j for j in pe if j not in path]
        off += [j for j in oe if j not in off and j not in path]
    return path, off


def route_mask(hooks, S, L, q_positions, slots):
    ks = [p for j in slots for p in slot_tokens(L["slots"][j])]
    by_layer = {}
    for l, h in S:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        hooks.add_mask([l], q_positions, ks, heads=hs)
    return ks


def run(runner, recips, train, S, rung, base_seed=0):
    S = [tuple(x) for x in S]
    rows, cell_names = [], []
    for gi, sample, pr in recips:
        K, L = pr.K, pr.layout()
        ids = pr.ids(runner.tok)
        G = MultiGeometry(runner, pr, ids, S)
        base, ro = G.base_split, G.ro
        own = capture(runner, ids, attn_eager=True)
        rng, gen = graph_rng(base_seed, gi), graph_gen(base_seed, gi)
        cells, meta = {}, {"readout": {k: v for k, v in ro.items() if k != "node_by_name"}}

        def split(logits):
            return answer_split(logits, ro["target"], ro["decoy"], node_ids=ro["nodes"])

        def cell(name, sp, **extra):
            cells[name] = dict(with_delta(sp, base["T"]), **extra)
            if name not in cell_names:
                cell_names.append(name)

        def skip(name, reason):
            cells[name] = {"skipped": True, "reason": reason}
            if name not in cell_names:
                cell_names.append(name)

        def measure(thought_edit=None, ids_=None):
            return split(run_ids(runner, ids if ids_ is None else ids_, thought_edit, attn_eager=True))

        # standing controls
        rs = reserialized_nl(pr, gi, base_seed)
        cell("reserialized", measure(ids_=rs.ids(runner.tok)))
        cell("self_transplant", measure(fixed(own, all_passes(K))))
        assert abs(cells["self_transplant"]["dT"]) < 1e-6, cells["self_transplant"]["dT"]
        d_gi, _ = random_donor_nl(train, pr.L, rng)
        _, rth = donor_run_nl(runner, train, d_gi, attn_eager=True)
        cell("random_donor/intermediates", measure(fixed(rth, intermediates(K))))
        sad = same_answer_donor_nl(train, pr)
        if sad is None:
            sth = None
            for nm in ("same_answer_donor/intermediates", "same_answer_donor/all"):
                skip(nm, "no_same_answer_donor")
        else:
            _, sth = donor_run_nl(runner, train, sad[0], attn_eager=True)
            cell("same_answer_donor/intermediates", measure(fixed(sth, intermediates(K))))
            cell("same_answer_donor/all", measure(fixed(sth, all_passes(K))))

        # the removal: directions from the unedited run, one answer-path edge per search pass
        on = on_path_nodes(pr)
        steps = []
        for k in range(pr.L - 1):
            qk = L["latents"][k]
            pe = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2]
            if not pe:
                steps = None
                break
            slot = max(pe, key=lambda j: G.total_attention(qk, j))
            steps.append((k, slot, G.directions(qk, slot)))
        removal = None
        if steps is None:
            for nm in ("qk_removal", "qk_random"):
                skip(nm, "no_path_edge_at_some_step")
        else:
            removal = per_pass_edit({k: span_removal(dirs) for k, _, dirs in steps})
            rand = per_pass_edit({k: span_removal(dirs, rand=rand_orthonormal(len(dirs), dirs[0].numel(), gen, dirs[0].device))
                                  for k, _, dirs in steps})
            for name, edit in (("qk_removal", removal), ("qk_random", rand)):
                logits, rec, _ = recorded_run(runner, ids, edit)
                before = [G.total_attention(L["latents"][k], slot) for k, slot, _ in steps]
                after = [G.total_attention(L["latents"][k], slot, rec) for k, slot, _ in steps]
                cell(name, split(logits), attn_path_before_per_step=before, attn_path_after_per_step=after,
                     attn_edges_before_per_step=[G.edge_mass(L["latents"][k]) for k, _, _ in steps],
                     attn_edges_after_per_step=[G.edge_mass(L["latents"][k], rec) for k, _, _ in steps],
                     n_directions=len(steps[0][2]))
            meta["removal_steps"] = [{"pass": k, "slot": slot} for k, slot, _ in steps]
            meta["coef_frac_per_step"] = [float(torch.stack(dirs, 1).T.matmul(own[k]).norm() / own[k].norm())
                                          for k, _, dirs in steps]

        # calibration mask: the route's heads at every latent query, path edges' tokens
        path, off = path_and_offpath_slots(pr, G)
        meta["path_slots"], meta["offpath_slots"] = path, off
        for tgt, slots in (("path", path), ("offpath", off)):
            if not slots:
                skip(f"latents_mask/{tgt}/alone", "no_slots")
                continue
            with AttnHooks(runner.model.base_causallm) as h:
                h.record_weights = True
                ks = route_mask(h, S, L, L["latents"], slots)
                logits = run_ids(runner, ids, attn_eager=True)
                leak = max(sum(G.per_head(h, q, ks).values()) for q in L["latents"])
            cell(f"latents_mask/{tgt}/alone", split(logits), masked_attention_max=leak)

        # carry-over: the donor's final thought at pass K-1, alone and after the removal
        def both(name, te):
            cell(f"{name}/alone", measure(te))
            if removal is None:
                skip(f"{name}/plus_removal", "no_path_edge_at_some_step")
            else:
                cell(f"{name}/plus_removal", measure(lambda k, t, te=te: te(k, removal(k, t))))

        if sth is not None:
            both("thoughtK/same_answer", fixed(sth, [K - 1]))
        else:
            skip("thoughtK/same_answer/alone", "no_same_answer_donor")
            skip("thoughtK/same_answer/plus_removal", "no_same_answer_donor")
        both("thoughtK/random", fixed(rth, [K - 1]))

        rows.append({"gi": gi, "K": K, "L": pr.L, "cov": covariates(pr), "baseline": base, "meta": meta,
                     "random_donor_gi": d_gi, "same_answer_donor_gi": sad[0] if sad else None, "cells": cells})
        show = ["reserialized", "same_answer_donor/intermediates", "qk_removal", "qk_random", "latents_mask/path/alone",
                "thoughtK/same_answer/plus_removal", "thoughtK/random/alone"]
        att = ""
        if steps is not None:
            c = cells["qk_removal"]
            att = f"  path attn {c['attn_path_before_per_step'][-1]:.2f}->{c['attn_path_after_per_step'][-1]:.2f}"
        print(f"graph {gi}: L={pr.L} base T {base['T']:.1f} e {base['e']:.2f}{att}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in show if n in cells and not cells[n].get("skipped")))

    summary = summarize(rows, cell_names)
    summary["fallback_line"] = fallback_line(rows, cell_names)
    summary["attention"] = attention_summary(rows)
    return {"rows": rows, "summary": summary, "cells": cell_names,
            "head_set": {"rung": rung, "heads": [head_name(l, h) for l, h in S], "n": len(S)},
            "cutoffs": {"edge_head_cut": EDGE_HEAD_CUT, "class_cut": CLASS_CUT, "top_k": TOP_K,
                        "decoy_weaken_cut": DECOY_WEAKEN_CUT}}


def attention_summary(rows):
    out = {}
    for name in ("qk_removal", "qk_random"):
        rs = [r["cells"][name] for r in rows if name in r["cells"] and not r["cells"][name].get("skipped")]
        if not rs:
            continue
        out[name] = {
            "last_step_before": bootstrap([c["attn_path_before_per_step"][-1] for c in rs], np.mean),
            "last_step_after": bootstrap([c["attn_path_after_per_step"][-1] for c in rs], np.mean),
            "mean_step_drop": bootstrap([float(np.mean([b - a for b, a in zip(c["attn_path_before_per_step"], c["attn_path_after_per_step"])])) for c in rs], np.mean),
            "edges_before_last": bootstrap([c["attn_edges_before_per_step"][-1] for c in rs], np.mean),
            "edges_after_last": bootstrap([c["attn_edges_after_per_step"][-1] for c in rs], np.mean),
        }
    for name in ("latents_mask/path/alone", "latents_mask/offpath/alone"):
        rs = [r["cells"][name] for r in rows if name in r["cells"] and not r["cells"][name].get("skipped")]
        if rs:
            out[name] = {"masked_attention_max": max(c["masked_attention_max"] for c in rs)}
    return out


def main():
    p = make_parser(__doc__.split("\n")[0])
    p.add_argument("--part", choices=("heads", "cells", "both"), default="both")
    p.set_defaults(run_name="gpt2_dilgren", device="cpu")
    args = p.parse_args()
    runner = load_gpt2_runner(args)
    print("load:", runner.load_report)
    recips = recipient_prompts_nl(args.mode)
    src = source_info(args.run_name)
    if args.part in ("heads", "both"):
        result = header(args, "heads", checkpoint_source=src)
        result.update(heads_mod.run(runner, recips, args.seed))
        finish(args, "heads", result)
    if args.part in ("cells", "both"):
        hp = require_file(ROOT / "results" / args.run_name / f"heads_{args.mode}.json", "gpt2_cells.py --part heads")
        H = json.load(open(hp))
        S, rung, mass = head_set(H["summary"], H["n_layers"], H["n_heads"])
        print(f"route ({rung}): {[head_name(l, h) for l, h in S]}")
        train = load_train()
        result = header(args, "cells", checkpoint_source=src, heads_file=str(hp.relative_to(ROOT)))
        result.update(run(runner, recips, train, S, rung, args.seed))
        result["head_set"]["search_latent_mass"] = {head_name(l, h): m for (l, h), m in mass.items()}
        finish(args, "cells", result)


if __name__ == "__main__":
    main()
