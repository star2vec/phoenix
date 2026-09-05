"""Validation of minimal-pair construction (pure graph/string checks).

1. Availability on train[:300] matches the read-only scan (300 / >=299 / 300).
2. Under an identical pinned serialization seed, the candidate-swap donor
   prompt differs from the recipient prompt ONLY at transposed token
   occurrences; question and root lines identical.
3. BFS invariants hold for every constructed donor (already enforced inside
   the constructors; re-checked independently here).
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

from harness import VENDOR, bfs_depths  # noqa: E402
from minimal_pairs import candidate_swap, interior_swap, placebo_swap  # noqa: E402
from dataset import expand_data  # noqa: E402


def pinned_question(sample, gi, seed=0):
    sample = dict(sample)
    sample["edges"] = [list(e) for e in sample["edges"]]
    random.seed((gi << 16) ^ seed)
    max_steps = len(sample["steps"])
    q, _ = expand_data(sample, max_steps + 1, max_steps, neg_sampling=False)
    return q


def main():
    data = json.load(open(VENDOR / "data/prosqa_train_graph_4_coconut.json"))
    slice300 = data[:300]

    counts = {"candidate": 0, "placebo": 0, "interior": 0}
    reasons = {}
    for s in slice300:
        for name, fn in (
            ("candidate", candidate_swap),
            ("placebo", placebo_swap),
            ("interior", interior_swap),
        ):
            donor, reason = fn(s)
            if donor is not None:
                counts[name] += 1
            else:
                reasons.setdefault(f"{name}:{reason}", 0)
                reasons[f"{name}:{reason}"] += 1
    print("availability on train[:300]:", counts, "skips:", reasons)
    assert counts["candidate"] == 300, counts
    assert counts["placebo"] >= 299, counts
    assert counts["interior"] == 300, counts

    n_diff_checked = 0
    for gi, s in enumerate(slice300[:20]):
        donor, _ = candidate_swap(s)
        qr = pinned_question(s, gi).split()
        qd = pinned_question(donor, gi).split()
        assert len(qr) == len(qd), (gi, len(qr), len(qd))
        T, D = str(s["target"]), str(s["neg_target"])
        swap = {T: D, D: T}
        n_swapped = 0
        in_question_tail = False
        for a, b in zip(qr, qd):
            if a == "[Q]":
                in_question_tail = True
            if a == b:
                continue
            assert not in_question_tail, (
                f"graph {gi}: question/root section differs: {a} vs {b}"
            )
            assert swap.get(a) == b, f"graph {gi}: non-swap diff {a} vs {b}"
            n_swapped += 1
        assert n_swapped >= 1, f"graph {gi}: no token actually swapped"
        n_diff_checked += 1
    print(f"prompt-diff check: {n_diff_checked}/20 pairs differ only at "
          f"transposed tokens; question+root lines identical")

    for gi, s in enumerate(slice300):
        d0 = bfs_depths(s["edges"], s["root"])
        for fn, kind in ((candidate_swap, "candidate"), (placebo_swap, "placebo"),
                         (interior_swap, "interior")):
            donor, _ = fn(s)
            if donor is None:
                continue
            d1 = bfs_depths(donor["edges"], donor["root"])
            if kind == "candidate":
                assert s["target"] not in d1
                assert d1.get(s["neg_target"]) == d0.get(s["target"])
            else:
                assert d1.get(s["target"]) == d0.get(s["target"])
                assert (s["neg_target"] in d1) == (s["neg_target"] in d0)
    print("BFS invariants: OK on all constructed donors (train[:300])")
    print("MINIMAL_PAIRS: PASS")


if __name__ == "__main__":
    main()
