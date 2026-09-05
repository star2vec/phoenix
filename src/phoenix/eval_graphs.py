"""Evaluation-graph generator.

Constraints per graph:
- At intervention step t=1 the reachable frontier is exactly the root's 3
  children {A, B, C} (branching 3; the MODE of the training distribution —
  31% of training graphs have root out-degree 3).
- Branch subtrees are node-disjoint (unique endpoint->branch attribution);
  extra DAG edges only connect consecutive depth layers WITHIN a branch, so
  BFS depths are exact and no shortcut changes the solution length.
- Answer candidate (target) sits at depth L in exactly one branch (L in
  {3,4}, ~50/50 like training); sibling branches contain no candidate.
- Decoy (neg_target) lives in a self-contained unreachable component with
  1-3 incoming edges, mirroring training (decoys average ~2 in-edges from
  unreachable nodes; ~9.6 unreachable nodes/graph).
- Node ids are a shuffled assignment of 0..n-1 (vocab node tokens 0..30),
  n <= 26 to leave slack under the 31-node-token vocabulary.

Output samples carry the vendor schema fields (root/target/neg_target/edges/
steps/idx_to_symbol/neighbor_k) so vendor expand_data() renders prompts with
byte-identical formatting, plus a "meta" dict describing the branch structure.
"""

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TARGET_NODES = 23          # match training mean 22.8
TARGET_EDGES = 36          # match training mean 36.6
MAX_NODES = 26             # vocab holds node tokens 0..30


def bfs_depths(edges, root):
    adj = {}
    for s, t in edges:
        adj.setdefault(s, []).append(t)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


def gen_graph(rng: random.Random):
    """Returns a sample dict in vendor schema + meta. Node ids are abstract
    ints 0..n-1 assigned by shuffle at the end."""
    L = rng.choice([3, 4])

    nodes = []  # abstract handles; index in this list is the pre-shuffle id
    def new_node():
        nodes.append(len(nodes))
        return nodes[-1]

    root = new_node()
    branch_roots = [new_node() for _ in range(3)]
    answer_branch = rng.randrange(3)
    edges = [(root, br) for br in branch_roots]

    # per-branch depth layers: layers[b][d] = nodes of branch b at depth d+1
    layers = [[[br]] for br in branch_roots]
    branch_of = {br: b for b, br in enumerate(branch_roots)}

    # answer path: branch_roots[answer_branch] -> ... -> target at depth L
    prev = branch_roots[answer_branch]
    for d in range(2, L + 1):
        nxt = new_node()
        branch_of[nxt] = answer_branch
        edges.append((prev, nxt))
        layers[answer_branch].append([nxt])
        prev = nxt
    target = prev

    # sibling branches: random chains of depth 2..L (never containing a candidate)
    for b in range(3):
        if b == answer_branch:
            continue
        depth_b = rng.randint(2, L)
        prev = branch_roots[b]
        for d in range(2, depth_b + 1):
            nxt = new_node()
            branch_of[nxt] = b
            edges.append((prev, nxt))
            layers[b].append([nxt])
            prev = nxt

    # widen branches with extra nodes until reachable part ~ 13-14 nodes
    reachable_budget = TARGET_NODES - 9 - len(nodes)  # ~9 unreachable later
    for _ in range(max(0, reachable_budget)):
        if len(nodes) >= MAX_NODES - 9:
            break
        b = rng.randrange(3)
        d = rng.randrange(len(layers[b]))  # attach below layer d
        parent = rng.choice(layers[b][d])
        child = new_node()
        branch_of[child] = b
        edges.append((parent, child))
        if d + 1 < len(layers[b]):
            layers[b][d + 1].append(child)
        else:
            layers[b].append([child])

    n_reachable = len(nodes)

    # unreachable component around the decoy
    n_unreach = min(rng.randint(8, 10), MAX_NODES - n_reachable)
    unreach = [new_node() for _ in range(n_unreach)]
    decoy = rng.choice(unreach)
    for u in unreach:
        if u == decoy:
            continue
        # internal DAG edges (higher id -> may receive from lower); density
        # tuned to training's ~14 unreach-internal edges per graph
        for v in unreach:
            if v >= u or rng.random() > 0.45:
                continue
            edges.append((v, u))
    for _ in range(rng.randint(1, 3)):  # decoy in-edges, like training
        src = rng.choice([u for u in unreach if u != decoy])
        if (src, decoy) not in edges:
            edges.append((src, decoy))
    # unreach -> reach edges (~2/graph in training). Direction can't create a
    # root->decoy path and leaves BFS depths and branch attribution untouched.
    reachable_nonroot = [v for v in range(n_reachable) if v != root]
    for _ in range(rng.randint(1, 3)):
        e = (rng.choice(unreach), rng.choice(reachable_nonroot))
        if e not in edges:
            edges.append(e)

    # densify: extra edges between consecutive layers within a branch
    # (depth-preserving, subtree-disjoint, no shortcut to target)
    tries = 0
    while len(edges) < TARGET_EDGES and tries < 500:
        tries += 1
        b = rng.randrange(3)
        if len(layers[b]) < 2:
            continue
        d = rng.randrange(len(layers[b]) - 1)
        u, v = rng.choice(layers[b][d]), rng.choice(layers[b][d + 1])
        if (u, v) not in edges:
            edges.append((u, v))

    # shuffle node ids
    n = len(nodes)
    perm = list(range(n))
    rng.shuffle(perm)
    remap = {old: perm[old] for old in range(n)}
    edges = [[remap[s], remap[t]] for s, t in edges]
    root, target, decoy = remap[root], remap[target], remap[decoy]
    branch_roots = [remap[br] for br in branch_roots]
    branch_of = {remap[k]: v for k, v in branch_of.items()}

    depth = bfs_depths(edges, root)
    # vendor-schema fields: neighbor_k = depth-k nodes on valid root->target
    # paths (used only for prompt length / optimal bucket); steps only via len()
    on_path = {target}
    changed = True
    parents = {}
    for s, t in edges:
        parents.setdefault(t, []).append(s)
    while changed:
        changed = False
        for v in list(on_path):
            for p in parents.get(v, []):
                if p in depth and depth[p] == depth[v] - 1 and p not in on_path:
                    on_path.add(p)
                    changed = True
    neighbor_k = {}
    for k in range(1, L + 1):
        neighbor_k[str(k)] = sorted(
            v for v in on_path if depth.get(v) == k
        )

    sample = {
        "question": "",
        "answer": str(target),
        "steps": [f"step{i}" for i in range(L)],
        "idx_to_symbol": [f"n{i}" for i in range(n)],
        "edges": edges,
        "root": root,
        "target": target,
        "neg_target": decoy,
        "neighbor_k": neighbor_k,
        "meta": {
            "L": L,
            "t_intervene": 1,
            "branch_roots": branch_roots,
            "answer_branch_root": branch_roots[answer_branch],
            "sibling_branch_roots": [
                br for i, br in enumerate(branch_roots) if i != answer_branch
            ],
            "branch_of": {str(k): v for k, v in branch_of.items()},
            "depth": {str(k): v for k, v in depth.items()},
        },
    }
    return sample


def validate(sample):
    edges = [tuple(e) for e in sample["edges"]]
    root, target, decoy = sample["root"], sample["target"], sample["neg_target"]
    meta = sample["meta"]
    depth = bfs_depths(edges, root)
    # frontier at depth 1 is exactly the 3 branch roots
    f1 = {v for v, d in depth.items() if d == 1}
    assert f1 == set(meta["branch_roots"]), "frontier-1 mismatch"
    # decoy unreachable, target at depth L
    assert decoy not in depth, "decoy reachable"
    assert depth[target] == meta["L"], "target depth != L"
    # branch subtrees disjoint: every reachable non-root node belongs to
    # exactly one branch, and edges never cross branches
    bo = {int(k): v for k, v in meta["branch_of"].items()}
    for s, t in edges:
        if s == root:
            continue
        if s in bo and t in bo:
            assert bo[s] == bo[t], "cross-branch edge"
        else:
            # remaining edges must originate in the unreachable set
            # (unreach->unreach or unreach->reach; never reach->unreach)
            assert s not in depth, "reach->unreach edge"
    # candidates: target in answer branch; no candidate in sibling branches
    assert bo[target] == bo[meta["answer_branch_root"]]
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--out", default=str(ROOT / "data" / "eval_graphs.json"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    graphs = []
    while len(graphs) < args.n:
        g = gen_graph(rng)
        validate(g)
        graphs.append(g)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(graphs, f)

    import statistics as st

    ns = [len(g["idx_to_symbol"]) for g in graphs]
    es = [len(g["edges"]) for g in graphs]
    Ls = [g["meta"]["L"] for g in graphs]
    print(
        f"wrote {len(graphs)} graphs -> {args.out}\n"
        f"nodes {st.mean(ns):.1f} (train 22.8) | edges {st.mean(es):.1f} "
        f"(train 36.6) | sol len {st.mean(Ls):.2f} (train 3.53)"
    )


if __name__ == "__main__":
    main()
