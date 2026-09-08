"""Amendment 1: relabel ProsQA graphs with uniformly random node labels.

ProsQA assigns node ids in breadth-first order, so a label is a depth cue
and the two name tokens 0 and 1 mark the root and the unreachable source.
This script rewrites each graph with an independent uniformly random
injection of its n nodes into the 31 node tokens 0..30, remapping edges,
root, target, decoy and the depth-k node lists. Graph structure, candidate
pairs, solution lengths and file order are unchanged. `idx_to_symbol`
becomes a 31-long list with each symbol at its new label and "<unused>"
elsewhere (the pipeline only reads its length and never its content).

Output: data/relabel/prosqa_{train,valid,test}_graph_4_coconut.json (the
vendor file names, so --data-dir swaps them in) and data/relabel/
manifest.json with the seed, source hashes, and the mean depth by label on
the relabeled training set (flat if the cue is gone).

    python src/phoenix/relabel.py --seed 20260906                                   # Amendment 1
    python src/phoenix/relabel.py --seed 20260908 --keep-names --out data/relabel_names  # Amendment 2
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import N_NODE_TOKENS, bfs_depths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VENDOR_DATA = ROOT / "vendor" / "reasoning-by-superposition" / "data"
OUT_DIR = ROOT / "data" / "relabel"
SPLITS = ("train", "valid", "test")
FILE = "prosqa_{split}_graph_4_coconut.json"


NAME_TOKENS = (0, 1)  # ProsQA: the two names; one is the root, the other the unreachable source


def relabel_graph(sample, rng, keep_names=False):
    """One graph with a fresh uniformly random injective labeling. With
    keep_names (Amendment 2) the two name nodes keep tokens 0 and 1 and only
    the concept nodes are relabeled, uniformly at random over 2..30."""
    n = len(sample["idx_to_symbol"])
    if keep_names:
        assert n >= 2 and sample["root"] in NAME_TOKENS
        perm = [0, 1] + rng.sample(range(2, N_NODE_TOKENS), n - 2)  # perm[old] = new
    else:
        perm = rng.sample(range(N_NODE_TOKENS), n)  # perm[old] = new
    f = lambda v: perm[v]
    symbols = ["<unused>"] * N_NODE_TOKENS
    for old, sym in enumerate(sample["idx_to_symbol"]):
        symbols[perm[old]] = sym
    out = dict(sample)
    out["edges"] = [[f(a), f(b)] for a, b in sample["edges"]]
    out["root"] = f(sample["root"])
    out["target"] = f(sample["target"])
    out["neg_target"] = f(sample["neg_target"])
    out["neighbor_k"] = {k: [f(v) for v in vs] for k, vs in sample["neighbor_k"].items()}
    out["idx_to_symbol"] = symbols
    out["relabel_perm"] = perm
    return out


def depth_by_label(graphs):
    acc = defaultdict(list)
    for s in graphs:
        d = bfs_depths(s["edges"], s["root"])
        for v in {v for e in s["edges"] for v in e} | {s["root"], s["target"], s["neg_target"]}:
            if v in d:
                acc[v].append(d[v])
    return {str(v): round(sum(x) / len(x), 3) for v, x in sorted(acc.items())}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--keep-names", action="store_true",
                   help="Amendment 2: names keep tokens 0 and 1; concepts random over 2..30")
    args = p.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "node_tokens": N_NODE_TOKENS,
                "mode": "names kept, concepts random" if args.keep_names else "all labels random", "splits": {}}
    for split in SPLITS:  # fixed order, so the seed reproduces every file
        src = VENDOR_DATA / FILE.format(split=split)
        data = json.load(open(src))
        new = [relabel_graph(s, rng, args.keep_names) for s in data]
        dst = out_dir / FILE.format(split=split)
        with open(dst, "w") as f:
            json.dump(new, f)
        manifest["splits"][split] = {
            "source": str(src.relative_to(ROOT)), "source_sha256": sha256(src),
            "output": str(dst.relative_to(ROOT)), "output_sha256": sha256(dst), "n_graphs": len(new),
        }
        if split == "train":
            manifest["mean_depth_by_label_relabeled_train"] = depth_by_label(new)
            manifest["mean_depth_by_label_original_train"] = depth_by_label(data)
        print(f"{split}: {len(new)} graphs -> {dst}")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    md = manifest["mean_depth_by_label_relabeled_train"]
    vals = list(md.values())
    print(f"relabeled train, mean depth by label: min {min(vals):.2f} max {max(vals):.2f} "
          f"(original: {min(manifest['mean_depth_by_label_original_train'].values()):.2f} to "
          f"{max(manifest['mean_depth_by_label_original_train'].values()):.2f})")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
