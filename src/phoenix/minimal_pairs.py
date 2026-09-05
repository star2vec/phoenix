"""Label-surgery minimal-pair donors.

Each constructor returns (donor_dict, None) or (None, skip_reason). Donor dicts
carry the RECIPIENT's target/neg_target fields so that, under the same pinned
serialization seed, the donor prompt's question and root lines are identical to
the recipient's — the only textual difference is the transposed token
occurrences inside edge tuples. `steps` matters only through len() (the
full-question branch of vendor expand_data never reads its content).

Validation is built in and hard-fails closed: a donor is returned only if its
BFS invariants hold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import bfs_depths


def _swap_tokens(edges, a, b):
    return [[b if x == a else a if x == b else x for x in e] for e in edges]


def _swap_neighbor_k(nk, a, b):
    return {
        d: [b if v == a else a if v == b else v for v in vs]
        for d, vs in nk.items()
    }


def _edge_nodes(edges):
    nodes = set()
    for s, t in edges:
        nodes.add(s)
        nodes.add(t)
    return nodes


def _transposed_depths_ok(sample, donor, a, b):
    """The donor's full depth map must equal the recipient's
    under the a<->b relabeling (guaranteed by isomorphism; tripwire for bugs)."""
    d0 = bfs_depths(sample["edges"], sample["root"])
    d1 = bfs_depths(donor["edges"], donor["root"])
    sw = {a: b, b: a}
    return d1 == {sw.get(v, v): d for v, d in d0.items()}


def _donor(sample, a, b):
    return {
        "edges": _swap_tokens(sample["edges"], a, b),
        "root": sample["root"],
        "target": sample["target"],
        "neg_target": sample["neg_target"],
        "steps": list(sample["steps"]),
        "neighbor_k": _swap_neighbor_k(sample["neighbor_k"], a, b),
        "idx_to_symbol": sample.get("idx_to_symbol"),
    }


def candidate_swap(sample):
    """Transpose target and decoy tokens: the decoy becomes reachable exactly
    where the target was; the answer flips by construction."""
    T, D, root = sample["target"], sample["neg_target"], sample["root"]
    if root in (T, D):
        return None, "root_is_candidate"
    d_orig = bfs_depths(sample["edges"], root)
    donor = _donor(sample, T, D)
    d_new = bfs_depths(donor["edges"], root)
    if T in d_new:
        return None, "target_still_reachable"
    if d_new.get(D) != d_orig.get(T):
        return None, "decoy_depth_mismatch"
    if not _transposed_depths_ok(sample, donor, T, D):
        return None, "depth_map_not_transposed"
    return donor, None


def placebo_swap(sample):
    """Transpose the two lowest-id nodes that are unreachable from root and are
    not candidates or root: same edit type, answer unchanged."""
    T, D, root = sample["target"], sample["neg_target"], sample["root"]
    d_orig = bfs_depths(sample["edges"], root)
    pool = sorted(
        v
        for v in _edge_nodes(sample["edges"])
        if v not in d_orig and v not in (T, D, root)
    )
    if len(pool) < 2:
        return None, "fewer_than_two_unreachable_noncandidates"
    a, b = pool[0], pool[1]
    donor = _donor(sample, a, b)
    d_new = bfs_depths(donor["edges"], root)
    if d_new.get(T) != d_orig.get(T) or (D in d_new) != (D in d_orig):
        return None, "placebo_changed_candidate_reachability"
    if not _transposed_depths_ok(sample, donor, a, b):
        return None, "depth_map_not_transposed"
    return donor, None


def interior_swap(sample):
    """Transpose the lowest-id valid-path interior node (depth 1..K-1) with the
    lowest-id unreachable non-candidate: same search shape, different interior
    identity."""
    T, D, root = sample["target"], sample["neg_target"], sample["root"]
    K = len(sample["steps"])
    interior = sorted(
        v
        for d in range(1, K)
        for v in sample["neighbor_k"].get(str(d), [])
        if v not in (T, D, root)
    )
    d_orig = bfs_depths(sample["edges"], root)
    outsiders = sorted(
        v
        for v in _edge_nodes(sample["edges"])
        if v not in d_orig and v not in (T, D, root)
    )
    if not interior:
        return None, "no_interior_node"
    if not outsiders:
        return None, "no_unreachable_noncandidate"
    donor = _donor(sample, interior[0], outsiders[0])
    d_new = bfs_depths(donor["edges"], root)
    if d_new.get(T) != d_orig.get(T) or (D in d_new) != (D in d_orig):
        return None, "interior_swap_changed_candidates"
    if not _transposed_depths_ok(sample, donor, interior[0], outsiders[0]):
        return None, "depth_map_not_transposed"
    return donor, None
