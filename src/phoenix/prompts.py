"""Explicit prompt rendering and prompt counterfactuals.

The vendor builder (dataset.expand_data) draws the edge order and the
candidate order from the global random module. Here both are explicit, so a
counterfactual can change exactly one thing about the prompt while every
other token stays where it was. render() reproduces the vendor string byte
for byte; tests/test_prompts.py checks this under pinned seeds.

Layout of a prompt with E edges and K latent tokens (absolute positions):

  0                <eos>
  1+3j, 2+3j       source and target of edge j (the edge's "slot")
  3+3j             the "|" separator after edge j (none after the last edge)
  3E               [Q]
  3E+1, 3E+2       the two candidates, in display order
  3E+3             [R]
  3E+4             root
  3E+5 .. 3E+4+K   the K latent tokens
  3E+5+K           [A]   (the answer is read from the logits here)

Reordering edges moves content between slots and leaves every other position
fixed. That is what makes the counterfactuals clean.

Conventions for the fields of a Prompt:
  target, decoy    the recipient's two candidates, under THIS prompt's
                   labeling. For a renamed prompt they are the renamed
                   labels. For a rewritten prompt (where the decoy becomes
                   the right answer) they stay as they were, so "following
                   the edit" shows as T falling to 0.
"""

import random
from collections import deque
from dataclasses import dataclass, replace

N_NODE_TOKENS = 31  # node labels are the tokens "0".."30"
LATENT = "<|latent|>"


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


def reverse_distances(edges, sink):
    """dist(v, sink) for every v with a path to sink (BFS on reversed edges)."""
    radj = {}
    for s, t in edges:
        radj.setdefault(t, []).append(s)
    dist = {sink: 0}
    q = deque([sink])
    while q:
        u = q.popleft()
        for v in radj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def layout(E, K):
    """Absolute token positions for a prompt with E edges and K latents."""
    slots = [(1 + 3 * j, 2 + 3 * j, (3 + 3 * j) if j < E - 1 else None) for j in range(E)]
    q = 3 * E
    return {
        "slots": slots,
        "q": q,
        "c1": q + 1,
        "c2": q + 2,
        "r": q + 3,
        "root": q + 4,
        "latents": [q + 5 + i for i in range(K)],
        "a": q + 5 + K,
        "n": q + 6 + K,
    }


def render(edges, cands, root, K, latent=LATENT):
    """Byte-identical to vendor expand_data for the full-task question."""
    body = "|".join(f" {s} {t} " for s, t in edges).strip()
    return f"<eos> {body} [Q] {cands[0]} {cands[1]} [R] {root}" + f" {latent}" * K + " [A] "


def pin_seed(gi, base_seed, reserial=False):
    """Same formula as thoughts.pin_serialization."""
    return ((gi << 16) ^ base_seed) + (0xABCDEF if reserial else 0)


def vendor_draws(sample, seed):
    """Replay the vendor's two RNG draws under `seed`: the edge shuffle and
    the candidate-order coin. Does not mutate the sample."""
    edges = [list(e) for e in sample["edges"]]
    random.seed(seed)
    random.shuffle(edges)
    if random.random() < 0.5:
        cands = (sample["target"], sample["neg_target"])
    else:
        cands = (sample["neg_target"], sample["target"])
    return edges, cands


@dataclass
class Prompt:
    edges: list       # ordered [[source, target], ...]
    cands: tuple      # (first, second) as displayed after [Q]
    root: int
    K: int            # number of latent tokens
    target: int
    decoy: int

    @classmethod
    def from_sample(cls, sample, seed):
        edges, cands = vendor_draws(sample, seed)
        return cls(edges, cands, sample["root"], len(sample["steps"]),
                   sample["target"], sample["neg_target"])

    def text(self, K=None, latent=LATENT):
        return render(self.edges, self.cands, self.root,
                      self.K if K is None else K, latent)

    def ids(self, tok, **kw):
        return tok.encode(self.text(**kw))

    def layout(self):
        return layout(len(self.edges), self.K)

    def depths(self):
        return bfs_depths(self.edges, self.root)

    def nodes(self):
        s = {self.root, self.target, self.decoy}
        for a, b in self.edges:
            s.add(a)
            s.add(b)
        return s

    def reachable_candidate(self):
        d = self.depths()
        r = [c for c in (self.target, self.decoy) if c in d]
        return r[0] if len(r) == 1 else None

    def parent_slots(self):
        """Slots of edges (p, target) with depth(p) == K-1: the edges the
        last expansion reads under either story."""
        d = self.depths()
        return [j for j, (s, t) in enumerate(self.edges)
                if t == self.target and d.get(s) == self.K - 1]

    def with_edges(self, edges, **kw):
        return replace(self, edges=[list(e) for e in edges], **kw)


# --- counterfactuals: each returns (Prompt or None, meta) --------------------

def reorder(prompt, rng):
    """Random permutation of edge slots. meta['perm'][j] is the new slot of
    original edge j."""
    E = len(prompt.edges)
    perm = list(range(E))
    for _ in range(100):
        rng.shuffle(perm)
        if any(perm[j] != j for j in range(E)):
            break
    new = [None] * E
    for j, e in enumerate(prompt.edges):
        new[perm[j]] = list(e)
    moved = sum(perm[j] != j for j in range(E))
    return prompt.with_edges(new), {"perm": perm, "moved": moved, "n_edges": E}


def reorder_unreachable(prompt, rng):
    """Permute only the edges whose source is unreachable from the root, among
    themselves. No edge a frontier could ever expand changes slot."""
    d = prompt.depths()
    idx = [j for j, (s, t) in enumerate(prompt.edges) if s not in d]
    if len(idx) < 2:
        return None, {"reason": "fewer_than_two_unreachable_edges"}
    sub = list(idx)
    for _ in range(100):
        rng.shuffle(sub)
        if any(a != b for a, b in zip(idx, sub)):
            break
    new = [list(e) for e in prompt.edges]
    perm = list(range(len(prompt.edges)))
    for a, b in zip(idx, sub):
        new[b] = list(prompt.edges[a])
        perm[a] = b
    moved = sum(a != b for a, b in zip(idx, sub))
    out = prompt.with_edges(new)
    assert out.depths() == d
    return out, {"perm": perm, "moved": moved, "n_unreachable_edges": len(idx)}


def rename(prompt, rng):
    """One consistent relabeling of every node token, edge order kept. No node
    present in the graph keeps its label. meta['sigma'][old] = new."""
    present = prompt.nodes()
    sigma = list(range(N_NODE_TOKENS))
    for _ in range(1000):
        rng.shuffle(sigma)
        if all(sigma[v] != v for v in present):
            break
    else:
        return None, {"reason": "no_derangement_found"}
    new = Prompt(
        edges=[[sigma[s], sigma[t]] for s, t in prompt.edges],
        cands=(sigma[prompt.cands[0]], sigma[prompt.cands[1]]),
        root=sigma[prompt.root],
        K=prompt.K,
        target=sigma[prompt.target],
        decoy=sigma[prompt.decoy],
    )
    return new, {"sigma": sigma}


def decoy_swap(prompt):
    """The edge(s) into the target from depth K-1 and edge(s) into the decoy
    trade slots. The graph is unchanged; only where those edges sit moves."""
    d = prompt.depths()
    parents = prompt.parent_slots()
    into_decoy = [j for j, (s, t) in enumerate(prompt.edges) if t == prompt.decoy]
    if not parents:
        return None, {"reason": "no_parent_edge_at_depth_K-1"}
    if not into_decoy:
        return None, {"reason": "decoy_has_no_in_edge"}
    n = min(len(parents), len(into_decoy))
    pairs = list(zip(parents[:n], into_decoy[:n]))
    new = [list(e) for e in prompt.edges]
    for j, m in pairs:
        new[j], new[m] = list(prompt.edges[m]), list(prompt.edges[j])
    out = prompt.with_edges(new)
    assert out.depths() == d, "decoy swap changed the graph"
    return out, {
        "pairs": pairs,
        "n_parent_edges": len(parents),
        "n_decoy_in_edges": len(into_decoy),
        "n_swapped": n,
        "complete": n == len(parents),
    }


def on_path_nodes(prompt):
    """Nodes on some shortest root->target path, with their depths."""
    d = prompt.depths()
    if prompt.target not in d:
        return {}
    parents = {}
    for s, t in prompt.edges:
        parents.setdefault(t, []).append(s)
    on = {prompt.target: d[prompt.target]}
    frontier = [prompt.target]
    while frontier:
        nxt = []
        for v in frontier:
            for p in parents.get(v, []):
                if d.get(p) == d[v] - 1 and p not in on:
                    on[p] = d[p]
                    nxt.append(p)
        frontier = nxt
    return on


def rewrite_at_depth(prompt, depth, rng):
    """Redirect one cut edge on the root->target paths whose head sits at
    `depth` so that the decoy becomes reachable at depth exactly K and the
    target becomes unreachable. depth == K is the last-hop rewrite (the edge
    into the target now points at the decoy, same slot). Returns None with a
    reason if no such edge exists for this graph."""
    d = prompt.depths()
    K = prompt.K
    if not 1 <= depth <= K:
        return None, {"reason": "depth_out_of_range"}
    on = on_path_nodes(prompt)
    cands = [j for j, (s, t) in enumerate(prompt.edges)
             if d.get(s) == depth - 1 and d.get(t) == depth and t in on]
    rng.shuffle(cands)
    need = K - depth
    dist_to_decoy = reverse_distances(prompt.edges, prompt.decoy)
    for j in cands:
        s, t = prompt.edges[j]
        without = prompt.edges[:j] + prompt.edges[j + 1:]
        if prompt.target in bfs_depths(without, prompt.root):
            continue  # not a cut edge for the target
        if need == 0:
            heads = [prompt.decoy]
        else:
            heads = [v for v, dv in dist_to_decoy.items() if dv == need and v not in d]
            rng.shuffle(heads)
        for bp in heads:
            if bp == s or [s, bp] in prompt.edges:
                continue
            new = [list(e) for e in prompt.edges]
            new[j] = [s, bp]
            dn = bfs_depths(new, prompt.root)
            if prompt.target in dn or dn.get(prompt.decoy) != K:
                continue
            return prompt.with_edges(new), {
                "slot": j, "depth": depth, "old_edge": [s, t], "new_edge": [s, bp],
            }
    return None, {"reason": "no_valid_rewrite_at_depth", "depth": depth}


def swap_labels(prompt, a, b):
    """Transpose two node labels in the edge list only. The question line
    (candidates) and the root are left as they are, as in the paper's label
    surgery donors."""
    if prompt.root in (a, b):
        return None, {"reason": "root_is_swapped_label"}
    new = [[b if x == a else a if x == b else x for x in e] for e in prompt.edges]
    return prompt.with_edges(new), {"a": a, "b": b}


def candidate_swap(prompt):
    """Target and decoy tokens transposed in the edge list: the decoy becomes
    reachable exactly where the target was; the answer flips by construction."""
    d0 = prompt.depths()
    out, meta = swap_labels(prompt, prompt.target, prompt.decoy)
    if out is None:
        return None, meta
    d1 = out.depths()
    if prompt.target in d1:
        return None, {"reason": "target_still_reachable"}
    if d1.get(prompt.decoy) != d0.get(prompt.target):
        return None, {"reason": "decoy_depth_mismatch"}
    return out, meta


def covariates(prompt):
    """Per-graph descriptors for the split analysis."""
    d = prompt.depths()
    E = len(prompt.edges)
    parents = prompt.parent_slots()
    return {
        "K": prompt.K,
        "n_edges": E,
        "n_nodes": len(prompt.nodes()),
        "n_branches": sum(1 for s, t in prompt.edges if s == prompt.root),
        "n_reachable": len(d),
        "n_parent_edges": len(parents),
        "first_parent_slot": parents[0] if parents else None,
        "first_parent_slot_frac": (parents[0] / E) if parents else None,
        "target_first": prompt.cands[0] == prompt.target,
    }
