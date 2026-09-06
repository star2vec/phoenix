"""Experiment 0: the paper's core measurements, rerun on a retrained model.

Three parts, on the paper's own sets so the numbers are comparable:

  subtraction  eval_graphs.json: at step 1, subtract the answer-branch root's
               direction (input-embedding basis; probe basis) with the
               matched random-direction control, and the sibling-branch
               subtraction, norm-preserving as in the paper.
  transplant   training graphs 0-99: matched donor (same two candidates and
               solution length, different graph, correct answer = the
               recipient's decoy) at intermediate steps, first step only,
               final step only, all steps; same-answer donor (control: can
               only break, not redirect); label-swap donor;
               placebo swap; interior swap; random donor; self-transplant;
               reserialized baseline.
  swap         training graphs 0-99: swap the target and decoy directions of
               the causal-Jacobian basis at the final step, and at each
               intermediate step.

Needs results/<run>/probe_basis.pt (fit_probes.py) and jlens_basis.pt
(fit_jlens.py). Accuracy and readout ordering come from evaluate.py.
pilot = first 10 graphs of each set; n100 = first 100.

    python src/phoenix/baseline.py --run-name seed0 --device mps --mode pilot [--part all]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_gen, graph_rng, header,
    load_runner, load_train, make_parser, random_donor, summarize, with_delta,
)
from edits import swap_edit  # noqa: E402
from measure import all_passes, at_passes, capture, fixed, intermediates, measure  # noqa: E402
from prompts import bfs_depths, candidate_swap, pin_seed, swap_labels  # noqa: E402
from sets import ROOT, load_eval_graphs, require_file, train_pin  # noqa: E402
from thoughts import find_donor  # noqa: E402

EVAL_OFFSET = 2_000_000
PARTS = ("subtraction", "transplant", "swap")


def unit(v):
    return v / v.norm()


def subtract_edit(u, r=None):
    """Norm-preserving subtraction of unit direction u (or of u's coefficient
    along random unit r, the matched control), at pass 0."""
    def f(k, t):
        u_ = u.to(t)
        d = r.to(t) if r is not None else u_
        t2 = t - (t @ u_) * d
        return t2 / t2.norm().clamp_min(1e-12) * t.norm()
    return at_passes(f, [0])


def run_subtraction(runner, graphs, probe_basis, base_seed=0):
    wte = runner.wte.detach()
    rows, names = [], []
    for gi, s in graphs:
        pr = Prompt.from_sample(s, pin_seed(gi + EVAL_OFFSET, base_seed))
        meta = s["meta"]
        v, sib = meta["answer_branch_root"], meta["sibling_branch_roots"][0]
        base = measure(runner, pr)
        gen = graph_gen(base_seed, gi)
        r = unit(torch.randn(wte.shape[1], generator=gen)).to(runner.device)
        cells = {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in names:
                names.append(name)

        cell("subtract_answer/input_embedding", measure(runner, pr, subtract_edit(unit(wte[v]))))
        cell("subtract_answer/random_matched", measure(runner, pr, subtract_edit(unit(wte[v]), r)))
        cell("subtract_sibling/input_embedding", measure(runner, pr, subtract_edit(unit(wte[sib]))))
        if probe_basis is not None and probe_basis[v].norm() > 0:
            cell("subtract_answer/probe", measure(runner, pr, subtract_edit(unit(probe_basis[v]))))
        else:
            cells["subtract_answer/probe"] = {"skipped": True, "reason": "no_probe_direction"}
        rows.append({"gi": gi, "K": pr.K, "cov": covariates(pr), "baseline": base, "cells": cells})
        print(f"eval graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.2f}" for n in names if not cells[n].get("skipped")))
    return {"rows": rows, "summary": summarize(rows, names), "cells": names}


def placebo_pair(pr):
    d = pr.depths()
    pool = sorted(v for v in pr.nodes() if v not in d and v not in (pr.target, pr.decoy, pr.root))
    return (pool[0], pool[1]) if len(pool) >= 2 else None


def interior_pair(pr, sample):
    d = pr.depths()
    interior = sorted(v for k in range(1, pr.K) for v in sample["neighbor_k"].get(str(k), [])
                      if v not in (pr.target, pr.decoy, pr.root))
    outsiders = sorted(v for v in pr.nodes() if v not in d and v not in (pr.target, pr.decoy, pr.root))
    return (interior[0], outsiders[0]) if interior and outsiders else None


def run_transplant(runner, graphs, train, base_seed=0):
    rows, names = [], []
    for gi, s in graphs:
        pr = Prompt.from_sample(s, train_pin(gi, base_seed))
        K = pr.K
        own = capture(runner, pr.ids(runner.tok))
        base = measure(runner, pr)
        rng = graph_rng(base_seed, gi)
        cells, info = {}, {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in names:
                names.append(name)

        cell("reserialized", measure(runner, Prompt.from_sample(s, train_pin(gi, base_seed, reserial=True))))
        cell("self_transplant", measure(runner, pr, fixed(own, all_passes(K))))
        assert cells["self_transplant"]["dT"] == 0.0

        # matched donor (the paper's): a different graph with the same two
        # candidates and solution length whose correct answer is the
        # recipient's decoy, so a transplant that carries the donor's search
        # switches the answer. The same-answer donor (correct answer = the
        # recipient's target) is kept as a control: it can only move the
        # answer by breaking the search, never by redirecting it.
        pool = [d for j, d in enumerate(train) if j != gi]
        for name, tgt, dec in (("matched_donor", s["neg_target"], s["target"]),
                               ("same_answer_donor", s["target"], s["neg_target"])):
            donor = find_donor(pool, tgt, dec, K)
            variants = ("intermediates", "first", "final", "all") if name == "matched_donor" else ("intermediates",)
            if donor is None:
                for v in variants:
                    cells[f"{name}/{v}"] = {"skipped": True, "reason": f"no_{name}"}
                continue
            d_gi = train.index(donor)
            _, dth = donor_run(runner, train, d_gi, base_seed)
            info[f"{name}_gi"] = d_gi
            passes = {"intermediates": intermediates(K), "first": [0], "final": [K - 1], "all": all_passes(K)}
            for v in variants:
                cell(f"{name}/{v}", measure(runner, pr, fixed(dth, passes[v])))

        # label-surgery donors under the same pinned serialization
        for name, pair_fn in (("label_swap", lambda: (pr.target, pr.decoy)),
                              ("placebo_swap", lambda: placebo_pair(pr)),
                              ("interior_swap", lambda: interior_pair(pr, s))):
            pair = pair_fn()
            cf, meta = (candidate_swap(pr) if name == "label_swap"
                        else (swap_labels(pr, *pair) if pair else (None, {"reason": "no_pair"})))
            if cf is None:
                cells[f"{name}/intermediates"] = {"skipped": True, "reason": meta.get("reason")}
                continue
            dth = capture(runner, cf.ids(runner.tok))
            info[name] = {"pair": list(pair), "donor_T_on_own_prompt": measure(runner, cf)["T"]}
            cell(f"{name}/intermediates", measure(runner, pr, fixed(dth, intermediates(K))))

        r_gi, _ = random_donor(train, K, rng, exclude={gi})
        _, rth = donor_run(runner, train, r_gi, base_seed)
        info["random_donor_gi"] = r_gi
        cell("random_donor/intermediates", measure(runner, pr, fixed(rth, intermediates(K))))

        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base, "info": info, "cells": cells})
        print(f"train graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in
            ("matched_donor/intermediates", "matched_donor/final", "same_answer_donor/intermediates",
             "label_swap/intermediates", "random_donor/intermediates")
            if not cells.get(n, {}).get("skipped")))
    return {"rows": rows, "summary": summarize(rows, names), "cells": names}


def run_swap(runner, graphs, jbasis, base_seed=0):
    """jbasis: tensor (K_max, vocab, 768) from fit_jlens.py."""
    rows, names = [], []
    for gi, s in graphs:
        pr = Prompt.from_sample(s, train_pin(gi, base_seed))
        K = pr.K
        base = measure(runner, pr)
        cells = {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in names:
                names.append(name)

        def swap_at(passes):
            fns = {}
            for k in passes:
                ua, ub = jbasis[k, pr.target], jbasis[k, pr.decoy]
                if ua.norm() == 0 or ub.norm() == 0:
                    return None
                f = swap_edit(unit(ua).to(runner.device), unit(ub).to(runner.device))
                if f is None:
                    return None
                fns[k] = f
            return lambda k, t: fns[k](t) if k in fns else t

        for name, passes in [("swap_final", [K - 1]), ("swap_intermediates", intermediates(K))] + [
                (f"swap_pass_{k}", [k]) for k in range(K - 1)]:
            e = swap_at(passes)
            if e is None:
                cells[name] = {"skipped": True, "reason": "no_jacobian_direction"}
                continue
            cell(name, measure(runner, pr, e))
        rows.append({"gi": gi, "K": K, "cov": covariates(pr), "baseline": base, "cells": cells})
        print(f"train graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}" for n in ("swap_final", "swap_intermediates") if not cells[n].get("skipped")))
    return {"rows": rows, "summary": summarize(rows, names), "cells": names}


def main():
    p = make_parser(__doc__.split("\n")[0])
    p.add_argument("--part", choices=PARTS + ("all",), default="all")
    args = p.parse_args()
    runner = load_runner(args)
    n = 10 if args.mode == "pilot" else 100
    parts = PARTS if args.part == "all" else (args.part,)
    rdir = ROOT / "results" / args.run_name
    for part in parts:
        result = header(args, f"baseline_{part}")
        if part == "subtraction":
            probe = torch.load(require_file(rdir / "probe_basis.pt", "fit_probes.py"), map_location="cpu", weights_only=True)
            graphs = list(enumerate(load_eval_graphs()))[:n]
            result.update(run_subtraction(runner, graphs, probe, args.seed))
        elif part == "transplant":
            train = load_train()
            result.update(run_transplant(runner, list(enumerate(train))[:n], train, args.seed))
        else:
            jb = torch.load(require_file(rdir / "jlens_basis.pt", "fit_jlens.py"), map_location="cpu", weights_only=True)
            train = load_train()
            result.update(run_swap(runner, list(enumerate(train))[:n], jb, args.seed))
        finish(args, f"baseline_{part}", result)


if __name__ == "__main__":
    main()
