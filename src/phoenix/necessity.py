"""Experiment 1: necessity, done properly (Kshirsagar's battery plus the
random donor).

Conditions on each pinned recipient, each at all K passes and at the
intermediate passes only:
  average            the mean recycled thought over training graphs at the
                     same pass (present but uninformative)
  noise              Gaussian direction at the graph's own thought norm
  zero               the paper's condition (off-manifold)
  random_donor       a random training graph's thoughts, same K (on-manifold
                     uninformative; the standing matched-random control)
and once each:
  removed            latent tokens deleted; the prompt ends `[R] root [A]`
  removed_length_kept  latent tokens replaced by the pad token (attended)
  self_transplant    own thoughts back into own run (must be exactly zero)
  reserialized       a second fixed edge order, nothing fixed (noise floor)

Every cell reports T, the change in T, e, and where the probability went.

    python src/phoenix/necessity.py --run-name seed0 --device mps --mode pilot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (  # noqa: E402
    Prompt, covariates, donor_run, finish, graph_gen, graph_rng, header,
    load_runner, load_train, make_parser, random_donor, recipient_prompts,
    summarize, with_delta,
)
from measure import (  # noqa: E402
    all_passes, at_passes, capture, fixed, intermediates, measure,
)
from sets import ROOT, test_pin, train_pin  # noqa: E402

N_MEAN = 2000
PAD = "<eos>"


def thought_means(runner, train, n=N_MEAN, base_seed=0, cache=None):
    """{pass_idx: mean recycled thought over the first n training graphs}."""
    if cache is not None and Path(cache).exists():
        return torch.load(cache, map_location=runner.device, weights_only=True)
    sums, counts = {}, {}
    for gi in range(min(n, len(train))):
        pr = Prompt.from_sample(train[gi], train_pin(gi, base_seed))
        for k, v in capture(runner, pr.ids(runner.tok)).items():
            sums[k] = sums.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1
    means = {k: sums[k] / counts[k] for k in sums}
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(means, cache)
    return means


def run(runner, recips, train, means, base_seed=0):
    rows = []
    cell_names = []
    for gi, sample, pr in recips:
        K = pr.K
        ids = pr.ids(runner.tok)
        own = capture(runner, ids)
        base = measure(runner, pr)
        rng = graph_rng(base_seed, gi)
        gen = graph_gen(base_seed, gi)
        d_gi, _ = random_donor(train, K, rng)
        _, donor = donor_run(runner, train, d_gi, base_seed)

        cells = {}

        def cell(name, split):
            cells[name] = with_delta(split, base["T"])
            if name not in cell_names:
                cell_names.append(name)

        for vname, passes in (("all", all_passes(K)), ("intermediates", intermediates(K))):
            cell(f"average/{vname}", measure(runner, pr, at_passes(lambda k, t: means[k].to(t), passes)))

            def noise(k, t, gen=gen):
                r = torch.randn(t.shape, generator=gen).to(t)
                return r / r.norm() * t.norm()
            cell(f"noise/{vname}", measure(runner, pr, at_passes(noise, passes)))
            cell(f"zero/{vname}", measure(runner, pr, at_passes(lambda k, t: t * 0.0, passes)))
            cell(f"random_donor/{vname}", measure(runner, pr, fixed(donor, passes)))
        cell("removed", measure(runner, pr, K=0))
        cell("removed_length_kept", measure(runner, pr, latent=PAD))
        cell("self_transplant", measure(runner, pr, fixed(own, all_passes(K))))
        assert cells["self_transplant"]["dT"] == 0.0, "self-transplant is not exactly zero"
        cell("reserialized", measure(runner, Prompt.from_sample(sample, test_pin(gi, base_seed, reserial=True))))

        rows.append({
            "gi": gi, "K": K, "cov": covariates(pr), "baseline": base,
            "random_donor_gi": d_gi, "thought_norms": {k: float(v.norm()) for k, v in own.items()},
            "cells": cells,
        })
        print(f"graph {gi}: base T {base['T']:.1f}  " + "  ".join(
            f"{n} {cells[n]['dT']:+.1f}/e{cells[n]['e']:.2f}" for n in
            ("average/all", "noise/all", "zero/all", "random_donor/all", "removed")))
    return {"rows": rows, "summary": summarize(rows, cell_names), "cells": cell_names}


def main():
    p = make_parser(__doc__.split("\n")[0])
    p.add_argument("--n-mean", type=int, default=N_MEAN)
    args = p.parse_args()
    runner = load_runner(args)
    train = load_train()
    recips = recipient_prompts(args.mode, args.seed)
    means = thought_means(runner, train, args.n_mean, args.seed,
                          cache=ROOT / "ckpts" / args.run_name / "thought_means.pt")  # gitignored cache
    result = header(args, "necessity", n_mean=args.n_mean)
    result.update(run(runner, recips, train, means, args.seed))
    finish(args, "necessity", result)


if __name__ == "__main__":
    main()
