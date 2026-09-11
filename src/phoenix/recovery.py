"""Experiment 6: where the answer is recovered, part two. Predictions in
NOTES.md (written before the pilot).

Final positions = the final latent (holding thought K) and the answer
position. Every cell runs alone and on top of the query-key every-step
removal from experiment 3 (directions from the unedited run).

  cand_tokens/{path,ctrl}      mask attention from the final positions onto
                               the two candidate tokens / onto [Q] and [R]
  root_token/{path,ctrl}       onto the root token / onto [Q] and [R]
  mlp/{answer,latent}/{both,L1,L2}/{mean,same_answer,random}
                               replace the MLP output at a final position by
                               its mean over training graphs at that role /
                               by the same-answer donor's output there / by
                               the random donor's (on-manifold controls)
  thoughtK/{same_answer,random}
                               thought K substituted by the donor's thought K
  decoy_edges/{path,ctrl}      final positions' attention masked onto every
                               edge into the decoy / onto a matched count of
                               the most attended edges into unreachable
                               non-candidates
  both_cand_edges/path         decoy edges plus the target's parent edges
  cand_tokens_plus_decoy_edges/path
  allq_decoy_edges/{path,ctrl}, allq_both_cand_edges/path
                               the same edge masks at every latent query and
                               the answer position (experiment 6b)
  thoughtKm1/{same_answer,random}, thoughtKm2/...
                               the donor's thought at step K-1 or K-2 only
Standing controls: reserialized, self-transplant (exactly zero), random
donor and same-answer donor at intermediates and at all K, the removal's
matched random directions.

    python src/phoenix/recovery.py --run-name seed0 --device cpu --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from attn_hooks import AttnHooks, MLPHooks  # noqa: E402
from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_gen, graph_rng, header,
    load_runner, load_train, make_parser, random_donor, recipient_prompts,
    same_answer_donor, summarize, with_delta,
)
from edits import rand_orthonormal  # noqa: E402
from measure import all_passes, answer_split, capture, fixed, intermediates, run_ids  # noqa: E402
from prompts import on_path_nodes  # noqa: E402
from qk_cells import _Geometry, per_pass_edit, span_removal, slot_tokens  # noqa: E402
from sets import ROOT, test_pin, train_pin  # noqa: E402

N_MEAN = 2000
DECOY_WEAKEN_CUT = 0.25  # protects "the switch to the decoy lost its support" (fallback line)


def mlp_means(runner, train, n=N_MEAN, base_seed=0, cache=None):
    """Mean MLP output per layer at the final latent and the answer position
    over the first n training graphs, plus the mean deviation norm."""
    if cache is not None and Path(cache).exists():
        return torch.load(cache, map_location=runner.device, weights_only=True)
    sums, sq, cnt = {}, {}, 0
    for gi in range(min(n, len(train))):
        pr = Prompt.from_sample(train[gi], train_pin(gi, base_seed))
        L = pr.layout()
        with MLPHooks(runner.model.base_causallm) as mh:
            mh.record = True
            run_ids(runner, pr.ids(runner.tok))
            for li in mh.store:
                for role, pos in (("latent", L["latents"][pr.K - 1]), ("answer", L["a"])):
                    v = mh.store[li][pos]
                    sums[(li, role)] = sums.get((li, role), 0) + v
                    sq[(li, role)] = sq.get((li, role), 0) + float(v.norm()) ** 2
        cnt += 1
    out = {}
    for key in sums:
        mean = sums[key] / cnt
        dev = (sq[key] / cnt - float(mean.norm()) ** 2) ** 0.5  # rms deviation norm
        out[key] = (mean, dev)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
    return out


def donor_mlp_outputs(runner, pr_donor):
    """{(layer, role): MLP output} captured from the donor's own run at its
    final latent and answer position (on-manifold replacement vectors)."""
    L = pr_donor.layout()
    with MLPHooks(runner.model.base_causallm) as mh:
        mh.record = True
        run_ids(runner, pr_donor.ids(runner.tok))
        out = {}
        for li in mh.store:
            out[(li, "latent")] = mh.store[li][L["latents"][pr_donor.K - 1]]
            out[(li, "answer")] = mh.store[li][L["a"]]
    return out


def run(runner, recips, train, means, base_seed=0):
    rows, cell_names = [], []
    for gi, sample, pr in recips:
        K, L = pr.K, pr.layout()
        ids = pr.ids(runner.tok)
        G = _Geometry(runner, pr, ids)
        base = G.base_split
        own = capture(runner, ids, attn_eager=True)
        rng, gen = graph_rng(base_seed, gi), graph_gen(base_seed, gi)
        finals = [L["latents"][K - 1], L["a"]]
        d = pr.depths()
        cells, meta = {}, {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in cell_names:
                cell_names.append(name)

        def skip(name, reason):
            cells[name] = {"skipped": True, "reason": reason}
            if name not in cell_names:
                cell_names.append(name)

        def measure(mask_keys=None, mlp_patches=None, thought_edit=None):
            with AttnHooks(runner.model.base_causallm) as h, MLPHooks(runner.model.base_causallm) as mh:
                if mask_keys:
                    h.add_mask([0, 1], finals, mask_keys)
                if mlp_patches:
                    mh.patches.update(mlp_patches)
                logits = run_ids(runner, ids, thought_edit, attn_eager=True)
            return answer_split(logits, pr.target, pr.decoy)

        # standing controls
        cell("reserialized", answer_split(run_ids(runner, Prompt.from_sample(sample, test_pin(gi, base_seed, True)).ids(runner.tok), attn_eager=True), pr.target, pr.decoy))
        cell("self_transplant", measure(thought_edit=fixed(own, all_passes(K))))
        assert abs(cells["self_transplant"]["dT"]) < 1e-6
        d_gi, _ = random_donor(train, K, rng)
        rpr, rth = donor_run(runner, train, d_gi, base_seed, attn_eager=True)
        cell("random_donor/intermediates", measure(thought_edit=fixed(rth, intermediates(K))))
        sad = same_answer_donor(train, pr.target, pr.decoy, K)
        spr, sth = donor_run(runner, train, sad[0], base_seed, attn_eager=True) if sad else (None, None)
        rmlp = donor_mlp_outputs(runner, rpr)
        smlp = donor_mlp_outputs(runner, spr) if spr is not None else None
        if sth is None:
            skip("same_answer_donor/intermediates", "no_same_answer_donor"); skip("same_answer_donor/all", "no_same_answer_donor")
        else:
            cell("same_answer_donor/intermediates", measure(thought_edit=fixed(sth, intermediates(K))))
            cell("same_answer_donor/all", measure(thought_edit=fixed(sth, all_passes(K))))

        # the query-key every-step removal (experiment 3), directions from the unedited run
        on = on_path_nodes(pr)
        steps = []
        for k in range(K - 1):
            qk = L["latents"][k]
            pe = [j for j, (s_, t) in enumerate(pr.edges) if on.get(s_) == k + 1 and on.get(t) == k + 2]
            if not pe:
                steps = None; break
            steps.append((k, G.directions(qk, max(pe, key=lambda j: G.total_attention(qk, j)))))
        removal = None
        if steps is None:
            skip("qk_removal", "no_path_edge_at_some_step"); skip("qk_random", "no_path_edge_at_some_step")
        else:
            removal = per_pass_edit({k: span_removal(dirs) for k, dirs in steps})
            cell("qk_removal", measure(thought_edit=removal))
            cell("qk_random", measure(thought_edit=per_pass_edit(
                {k: span_removal(dirs, rand=rand_orthonormal(8, 768, gen, dirs[0].device)) for k, dirs in steps})))

        def both(name, **kw):
            """alone and plus removal"""
            cell(f"{name}/alone", measure(**kw))
            if removal is None:
                skip(f"{name}/plus_removal", "no_path_edge_at_some_step")
            else:
                te = kw.pop("thought_edit", None)
                if te is None:
                    cell(f"{name}/plus_removal", measure(thought_edit=removal, **kw))
                else:
                    # compose: removal at passes 0..K-2, then the cell's own edit
                    cell(f"{name}/plus_removal", measure(thought_edit=lambda k, t, te=te: te(k, removal(k, t)), **kw))

        # 1, 2: candidate tokens, root token; control: the [Q] and [R] markers
        both("cand_tokens/path", mask_keys=[L["c1"], L["c2"]])
        both("cand_tokens/ctrl", mask_keys=[L["q"], L["r"]])
        both("root_token/path", mask_keys=[L["root"]])
        both("root_token/ctrl", mask_keys=[L["q"], L["r"]])

        # 3: final-position MLPs: training mean, same-answer donor's output,
        # random donor's output (the last two are on-manifold controls)
        for role, pos in (("answer", L["a"]), ("latent", L["latents"][K - 1])):
            for lname, layers in (("both", (0, 1)), ("L1", (0,)), ("L2", (1,))):
                if role == "latent" and lname != "both":
                    continue
                both(f"mlp/{role}/{lname}/mean", mlp_patches={(li, pos): means[(li, role)][0] for li in layers})
                if smlp is not None:
                    both(f"mlp/{role}/{lname}/same_answer", mlp_patches={(li, pos): smlp[(li, role)] for li in layers})
                else:
                    skip(f"mlp/{role}/{lname}/same_answer/alone", "no_same_answer_donor"); skip(f"mlp/{role}/{lname}/same_answer/plus_removal", "no_same_answer_donor")
                both(f"mlp/{role}/{lname}/random", mlp_patches={(li, pos): rmlp[(li, role)] for li in layers})

        # 4: thought K substituted by a donor's thought K
        if sth is not None:
            both("thoughtK/same_answer", thought_edit=fixed(sth, [K - 1]))
        else:
            skip("thoughtK/same_answer/alone", "no_same_answer_donor"); skip("thoughtK/same_answer/plus_removal", "no_same_answer_donor")
        both("thoughtK/random", thought_edit=fixed(rth, [K - 1]))

        # 5, 6, 7: decoy's edges, both candidates' edges, candidate tokens plus decoy edges
        decoy_slots = [j for j, (s_, t) in enumerate(pr.edges) if t == pr.decoy]
        parent_slots = pr.parent_slots()
        others = [j for j, (s_, t) in enumerate(pr.edges) if t not in d and t not in (pr.target, pr.decoy)]
        others = sorted(others, key=lambda j: -sum(G.total_attention(q, j) for q in finals))[:len(decoy_slots)]
        meta.update({"decoy_slots": decoy_slots, "parent_slots": parent_slots, "ctrl_slots": others})
        tok = lambda slots: [p for j in slots for p in slot_tokens(L["slots"][j])]
        if decoy_slots:
            both("decoy_edges/path", mask_keys=tok(decoy_slots))
            both("decoy_edges/ctrl", mask_keys=tok(others) if others else tok(decoy_slots))
            both("both_cand_edges/path", mask_keys=tok(decoy_slots + parent_slots))
            both("cand_tokens_plus_decoy_edges/path", mask_keys=[L["c1"], L["c2"]] + tok(decoy_slots))
        else:
            for nm in ("decoy_edges/path", "decoy_edges/ctrl", "both_cand_edges/path", "cand_tokens_plus_decoy_edges/path"):
                skip(f"{nm}/alone", "decoy_has_no_in_edge"); skip(f"{nm}/plus_removal", "decoy_has_no_in_edge")

        # 6b: candidate-edge masks at every latent query and the answer position
        allq = list(L["latents"]) + [L["a"]]

        def measure_allq(mask_keys, thought_edit=None):
            with AttnHooks(runner.model.base_causallm) as h:
                h.add_mask([0, 1], allq, mask_keys)
                logits = run_ids(runner, ids, thought_edit, attn_eager=True)
            return answer_split(logits, pr.target, pr.decoy)

        def both_allq(name, keys):
            cell(f"{name}/alone", measure_allq(keys))
            if removal is None:
                skip(f"{name}/plus_removal", "no_path_edge_at_some_step")
            else:
                cell(f"{name}/plus_removal", measure_allq(keys, removal))
        if decoy_slots:
            both_allq("allq_decoy_edges/path", tok(decoy_slots))
            both_allq("allq_decoy_edges/ctrl", tok(others) if others else tok(decoy_slots))
            both_allq("allq_both_cand_edges/path", tok(decoy_slots + parent_slots))
        else:
            for nm in ("allq_decoy_edges/path", "allq_decoy_edges/ctrl", "allq_both_cand_edges/path"):
                skip(f"{nm}/alone", "decoy_has_no_in_edge"); skip(f"{nm}/plus_removal", "decoy_has_no_in_edge")

        # 6b: per-step carry-over, the donor's thought at step j only (thought K left to the model)
        for j, tag in ((K - 2, "Km1"), (K - 3, "Km2")):
            if j < 0:
                for src in ("same_answer", "random"):
                    skip(f"thought{tag}/{src}/alone", "no_such_step"); skip(f"thought{tag}/{src}/plus_removal", "no_such_step")
                continue
            if sth is not None:
                both(f"thought{tag}/same_answer", thought_edit=fixed(sth, [j]))
            else:
                skip(f"thought{tag}/same_answer/alone", "no_same_answer_donor"); skip(f"thought{tag}/same_answer/plus_removal", "no_same_answer_donor")
            both(f"thought{tag}/random", thought_edit=fixed(rth, [j]))

        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base, "meta": meta,
                     "random_donor_gi": d_gi, "same_answer_donor_gi": sad[0] if sad else None, "cells": cells})
        show = ["qk_removal", "mlp/answer/L1/mean/alone", "mlp/answer/L1/same_answer/alone", "mlp/answer/L1/random/alone", "thoughtK/same_answer/plus_removal",
                "thoughtK/random/plus_removal", "decoy_edges/path/plus_removal", "cand_tokens/path/alone"]
        print(f"graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in show if n in cells and not cells[n].get("skipped")))

    summary = summarize(rows, cell_names)
    summary["fallback_line"] = fallback_line(rows, cell_names)
    return {"rows": rows, "summary": summary, "cells": cell_names, "decoy_weaken_cut": DECOY_WEAKEN_CUT}


def fallback_line(rows, cell_names):
    """On graphs that flip under the removal alone: mean p_decoy under each
    plus-removal cell, against the removal's own; 'weakened' when it falls by
    at least DECOY_WEAKEN_CUT."""
    import numpy as np
    from stats import bootstrap
    fl = [r for r in rows if not r["cells"].get("qk_removal", {}).get("skipped") and r["cells"]["qk_removal"]["dT"] <= -50]
    out = {"n_removal_flipped": len(fl)}
    if not fl:
        return out
    base = np.array([r["cells"]["qk_removal"]["p_decoy"] for r in fl])
    out["removal_p_decoy"] = bootstrap(base, np.mean)
    for c in cell_names:
        if not c.endswith("/plus_removal"):
            continue
        vals = [(r["cells"][c]["p_decoy"], r["cells"][c]["p_target"], r["cells"][c]["e"]) for r in fl if not r["cells"].get(c, {}).get("skipped")]
        if not vals:
            continue
        pd_ = np.array([v[0] for v in vals])
        out[c] = {"p_decoy": bootstrap(pd_, np.mean), "p_target": bootstrap([v[1] for v in vals], np.mean),
                  "e": bootstrap([v[2] for v in vals], np.mean),
                  "frac_weakened": bootstrap([float(b - p >= DECOY_WEAKEN_CUT) for b, p in zip(base[:len(pd_)], pd_)], np.mean)}
    return out


def main():
    p = make_parser(__doc__.split("\n")[0])
    p.add_argument("--n-mean", type=int, default=N_MEAN)
    args = p.parse_args()
    runner = load_runner(args)
    train = load_train()
    means = mlp_means(runner, train, args.n_mean, args.seed, cache=ROOT / "ckpts" / args.run_name / "mlp_means.pt")
    recips = recipient_prompts(args.mode, args.seed)
    result = header(args, "recovery", n_mean=args.n_mean)
    result.update(run(runner, recips, train, means, args.seed))
    finish(args, "recovery", result)


if __name__ == "__main__":
    main()
