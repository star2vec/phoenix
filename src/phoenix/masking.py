"""Experiment 5: where the answer is recovered. Predictions in NOTES.md.

Blocks attention routes onto the answer-path edges (a mask added before the
softmax, `AttnHooks.add_mask`) alone and together with the query-key
every-step removal from experiment 3, on the existing checkpoints.

Head sets come from experiment 2's n=100 summary for the same run
(results/<run>/heads_n100.json): a head reads edges at a query class if its
mean attention onto edge slots there is at least EDGE_HEAD_CUT. Masks act on
the path edges (every shortest-path edge from depth k+1 to depth k+2, all
steps) or on a matched set of off-path edges (the most attended frontier
edges off the path, same count).

    python src/phoenix/masking.py --run-name seed0 --device cpu --mode pilot
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attn_hooks import AttnHooks  # noqa: E402
from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_gen, graph_rng, header,
    load_runner, load_train, make_parser, random_donor, recipient_prompts,
    same_answer_donor, summarize, with_delta,
)
from measure import all_passes, answer_split, capture, fixed, intermediates, run_ids  # noqa: E402
from prompts import on_path_nodes  # noqa: E402
from qk_cells import _Geometry, per_pass_edit, span_removal, slot_tokens  # noqa: E402
from edits import rand_orthonormal  # noqa: E402
from sets import ROOT, require_file, test_pin  # noqa: E402

EDGE_HEAD_CUT = 0.5  # protects "this head reads edges" (stricter than experiment 2's 0.2 scoring cutoff)


def head_sets(run_name, cut=EDGE_HEAD_CUT):
    """{'latents': {layer: [heads]}, 'answer': {layer: [heads]}} from heads_n100.json."""
    path = require_file(ROOT / "results" / run_name / "heads_n100.json", "heads.py --mode n100")
    S = json.load(open(path))["summary"]
    out = {"latents": {}, "answer": {}}
    for layer in (0, 1):
        for h in range(8):
            def mass(cls):
                e = S.get(f"L{layer + 1}H{h}/{cls}")
                return e["slot_mass_a"]["point"] if e and e.get("slot_mass_a") else 0.0
            lat = min(mass("intermediate_latent"), mass("last_latent"))
            if lat >= cut and layer == 0:
                # the latents route is layer 1 only: every layer-2 head reads
                # edges at the latents, and that route is the query-key removal
                # (and the l2_latents_mask calibration cell), not this one
                out["latents"].setdefault(layer, []).append(h)
            if mass("answer") >= cut:
                out["answer"].setdefault(layer, []).append(h)
    return out


def path_and_offpath_slots(pr, G):
    """Path edges (shortest-path edges from depth k+1 to k+2 at every step)
    and a matched count of the most attended off-path frontier edges."""
    d, on, K, L = pr.depths(), on_path_nodes(pr), pr.K, pr.layout()
    path, off = [], []
    for k in range(K):
        q = L["latents"][k] if k < K else L["a"]
        p_edges = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2] if k < K - 1 else \
                  [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == K - 1 and t == pr.target]
        o_edges = [j for j, (s_, t) in enumerate(pr.edges) if d.get(s_) == k + 1 and j not in p_edges]
        o_edges = sorted(o_edges, key=lambda j: -G.total_attention(q, j))[:len(p_edges)]
        path += [j for j in p_edges if j not in path]
        off += [j for j in o_edges if j not in off and j not in path]
    return path, off


def mask_positions(L, slots):
    return [p for j in slots for p in slot_tokens(L["slots"][j])]


def apply_masks(hooks, sets, which, L, K, slots):
    """which: subset of {'latents', 'answer'}; masks the heads of each layer
    at the given queries onto the slots' tokens. AttnHooks masks all heads of
    a layer, so head selection is done by masking only in layers where the
    set is non-empty and recording the selected heads; see note in run()."""
    ks = mask_positions(L, slots)
    if "latents" in which:
        for layer, heads in sets["latents"].items():
            hooks.add_mask([layer], L["latents"], ks, heads=heads)
    if "answer" in which:
        for layer, heads in sets["answer"].items():
            hooks.add_mask([layer], [L["a"]], ks, heads=heads)
    if "l2_latents" in which:
        hooks.add_mask([1], L["latents"], ks, heads=list(range(8)))


def run(runner, recips, train, sets, base_seed=0):
    rows, cell_names = [], []
    n_heads = 8
    for gi, sample, pr in recips:
        K, L = pr.K, pr.layout()
        ids = pr.ids(runner.tok)
        G = _Geometry(runner, pr, ids)
        base = G.base_split
        own = capture(runner, ids, attn_eager=True)
        rng, gen = graph_rng(base_seed, gi), graph_gen(base_seed, gi)
        path, off = path_and_offpath_slots(pr, G)
        cells, meta = {}, {"path_slots": path, "offpath_slots": off, "head_sets": sets}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in cell_names:
                cell_names.append(name)

        def skip(name, reason):
            cells[name] = {"skipped": True, "reason": reason}
            if name not in cell_names:
                cell_names.append(name)

        def measure_with(mask_which, slots, thought_edit=None, name=None):
            with AttnHooks(runner.model.base_causallm) as h:
                if mask_which:
                    apply_masks(h, sets, mask_which, L, K, slots)
                logits = run_ids(runner, ids, thought_edit, attn_eager=True)
            return answer_split(logits, pr.target, pr.decoy)

        # standing controls (eager path, same baseline)
        cell("reserialized", answer_split(run_ids(runner, Prompt.from_sample(sample, test_pin(gi, base_seed, True)).ids(runner.tok), attn_eager=True), pr.target, pr.decoy))
        cell("self_transplant", measure_with((), [], fixed(own, all_passes(K))))
        assert abs(cells["self_transplant"]["dT"]) < 1e-6
        d_gi, _ = random_donor(train, K, rng)
        donor = donor_run(runner, train, d_gi, base_seed, attn_eager=True)[1]
        cell("random_donor/intermediates", measure_with((), [], fixed(donor, intermediates(K))))
        sad = same_answer_donor(train, pr.target, pr.decoy, K)
        if sad is None:
            skip("same_answer_donor/intermediates", "no_same_answer_donor"); skip("same_answer_donor/all", "no_same_answer_donor")
        else:
            sth = donor_run(runner, train, sad[0], base_seed, attn_eager=True)[1]
            cell("same_answer_donor/intermediates", measure_with((), [], fixed(sth, intermediates(K))))
            cell("same_answer_donor/all", measure_with((), [], fixed(sth, all_passes(K))))

        # the query-key every-step removal, rebuilt here (directions from the unedited run)
        on = on_path_nodes(pr)
        steps = []
        for k in range(K - 1):
            qk = L["latents"][k]
            pe = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2]
            if not pe:
                steps = None; break
            slot = max(pe, key=lambda j: G.total_attention(qk, j))
            steps.append((k, G.directions(qk, slot)))
        if steps is None:
            for nm in ("qk_removal", "qk_random", "l1_latents/path/plus_removal", "answer_heads/path/plus_removal",
                       "all_routes/path/plus_removal", "l1_latents/offpath/plus_removal", "answer_heads/offpath/plus_removal",
                       "all_routes/offpath/plus_removal"):
                skip(nm, "no_path_edge_at_some_step")
            removal = None
        else:
            removal = per_pass_edit({k: span_removal(dirs) for k, dirs in steps})
            rand = per_pass_edit({k: span_removal(dirs, rand=rand_orthonormal(n_heads, 768, gen, dirs[0].device)) for k, dirs in steps})
            cell("qk_removal", measure_with((), [], removal))
            cell("qk_random", measure_with((), [], rand))

        # masks alone and plus removal, on path and off-path slots
        for route, which in (("l1_latents", ("latents",)), ("answer_heads", ("answer",)), ("all_routes", ("latents", "answer"))):
            if all(not sets[w] for w in which):
                for tgt in ("path", "offpath"):
                    skip(f"{route}/{tgt}/alone", "no_heads_selected"); skip(f"{route}/{tgt}/plus_removal", "no_heads_selected")
                continue
            for tgt, slots in (("path", path), ("offpath", off)):
                cell(f"{route}/{tgt}/alone", measure_with(which, slots))
                if removal is not None:
                    cell(f"{route}/{tgt}/plus_removal", measure_with(which, slots, removal))
        cell("l2_latents_mask/path/alone", measure_with(("l2_latents",), path))
        cell("l2_latents_mask/offpath/alone", measure_with(("l2_latents",), off))

        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base, "meta": meta,
                     "random_donor_gi": d_gi, "same_answer_donor_gi": sad[0] if sad else None, "cells": cells})
        show = ["qk_removal", "l1_latents/path/alone", "l1_latents/path/plus_removal", "answer_heads/path/alone",
                "answer_heads/path/plus_removal", "all_routes/path/plus_removal", "all_routes/offpath/plus_removal", "l2_latents_mask/path/alone"]
        print(f"graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in show if n in cells and not cells[n].get("skipped")))
    return {"rows": rows, "summary": summarize(rows, cell_names), "cells": cell_names, "head_sets": sets,
            "edge_head_cut": EDGE_HEAD_CUT}


def main():
    args = make_parser(__doc__.split("\n")[0]).parse_args()
    runner = load_runner(args)
    sets = head_sets(args.run_name)
    print("head sets:", sets)
    train = load_train()
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "masking")
    result.update(run(runner, recips, train, sets, args.seed))
    finish(args, "masking", result)


if __name__ == "__main__":
    main()
