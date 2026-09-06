"""Amendment 1 relabeler: injective labels in 0..30, structure preserved,
fields remapped, and the label-depth cue gone.

    .venv/bin/python tests/test_relabel.py     # must print RELABEL: PASS
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

from prompts import bfs_depths  # noqa: E402
from relabel import depth_by_label, relabel_graph, VENDOR_DATA, FILE  # noqa: E402


def main():
    data = json.load(open(VENDOR_DATA / FILE.format(split="train")))
    rng = random.Random(0)
    new = [relabel_graph(s, rng) for s in data[:2000]]
    for s, t in zip(data[:2000], new):
        perm = t["relabel_perm"]
        n = len(s["idx_to_symbol"])
        assert len(perm) == n and len(set(perm)) == n and all(0 <= v <= 30 for v in perm)
        f = lambda v: perm[v]
        assert t["edges"] == [[f(a), f(b)] for a, b in s["edges"]]
        assert (t["root"], t["target"], t["neg_target"]) == (f(s["root"]), f(s["target"]), f(s["neg_target"]))
        d0, d1 = bfs_depths(s["edges"], s["root"]), bfs_depths(t["edges"], t["root"])
        assert d1 == {f(v): dep for v, dep in d0.items()}
        assert t["neighbor_k"] == {k: [f(v) for v in vs] for k, vs in s["neighbor_k"].items()}
        assert len(t["idx_to_symbol"]) == 31 and all(t["idx_to_symbol"][f(i)] == sym for i, sym in enumerate(s["idx_to_symbol"]))
        assert t["steps"] == s["steps"]
    print("structure and fields: OK (2000 graphs)")
    md_new, md_old = depth_by_label(new), depth_by_label(data[:2000])
    rng_new = max(md_new.values()) - min(md_new.values())
    rng_old = max(md_old.values()) - min(md_old.values())
    print(f"mean depth by label: original range {rng_old:.2f}, relabeled range {rng_new:.2f}")
    assert rng_old > 1.5, "the original cue should be visible"
    assert rng_new < 0.4, "the relabeled cue should be gone"
    roots = {t["root"] for t in new}
    assert len(roots) > 20, "roots should spread over the label range"
    # the full file is reproduced exactly by the seed (manifest reproducibility)
    a = [relabel_graph(s, random.Random(7)) for s in data[:50]]
    b = [relabel_graph(s, random.Random(7)) for s in data[:50]]
    assert a == b
    print("RELABEL: PASS")


if __name__ == "__main__":
    main()
