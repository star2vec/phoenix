"""prompts.py: byte-identity with the vendor builder, layout, and the BFS
invariants of every counterfactual. Pure graph/string checks; no model.

    .venv/bin/python tests/test_prompts.py      # must print PROMPTS: PASS
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

from harness import VENDOR, bfs_depths  # noqa: E402
from dataset import expand_data  # noqa: E402
from fast_coconut import FastSTokenizer  # noqa: E402
from minimal_pairs import candidate_swap as mp_candidate_swap  # noqa: E402
from thoughts import pin_serialization  # noqa: E402
import prompts as P  # noqa: E402


def main():
    data = json.load(open(VENDOR / "data/prosqa_train_graph_4_coconut.json"))
    test = json.load(open(VENDOR / "data/prosqa_test_graph_4_coconut.json"))
    tok = FastSTokenizer()

    # 1. byte-identity with expand_data under the project's pinned seeds
    n_checked = 0
    for gi, s in enumerate(data[:300]):
        for base_seed in (0, 1):
            for reserial in (False, True):
                s_copy = dict(s)
                s_copy["edges"] = [list(e) for e in s["edges"]]
                K = len(s["steps"])
                pin_serialization(gi, base_seed, reserial)
                q, _ = expand_data(s_copy, K + 1, K, neg_sampling=False)
                pr = P.Prompt.from_sample(s, P.pin_seed(gi, base_seed, reserial))
                assert pr.text() == q, (gi, base_seed, reserial, pr.text(), q)
                assert pr.ids(tok) == tok.encode(q)
                n_checked += 1
    print(f"render: OK ({n_checked} prompts byte-identical to expand_data)")

    # 2. layout: every named position holds the token it names
    for gi, s in enumerate(test[:100]):
        pr = P.Prompt.from_sample(s, P.pin_seed(gi + 1_000_000, 0))
        ids = pr.ids(tok)
        L = pr.layout()
        assert len(ids) == L["n"], (len(ids), L["n"])
        assert ids[0] == tok.vocab["<eos>"]
        for j, (ps, pt, sep) in enumerate(L["slots"]):
            assert ids[ps] == pr.edges[j][0] and ids[pt] == pr.edges[j][1]
            if sep is not None:
                assert ids[sep] == tok.vocab["|"]
        assert ids[L["q"]] == tok.vocab["[Q]"]
        assert (ids[L["c1"]], ids[L["c2"]]) == pr.cands
        assert ids[L["r"]] == tok.vocab["[R]"] and ids[L["root"]] == pr.root
        assert all(ids[p] == tok.vocab["<|latent|>"] for p in L["latents"])
        assert ids[L["a"]] == tok.vocab["[A]"]
        # removal variants
        assert len(pr.ids(tok, K=0)) == L["n"] - pr.K
        padded = pr.ids(tok, latent="<eos>")
        assert len(padded) == L["n"] and all(padded[p] == tok.vocab["<eos>"] for p in L["latents"])
    print("layout: OK (100 test prompts)")

    # 3. counterfactual invariants on 300 training graphs
    rng = random.Random(0)
    avail = Counter()
    for gi, s in enumerate(data[:300]):
        pr = P.Prompt.from_sample(s, P.pin_seed(gi, 0))
        d0 = pr.depths()
        E = len(pr.edges)
        assert pr.reachable_candidate() == pr.target

        # reorder: same edge multiset, same graph, perm consistent
        r, m = P.reorder(pr, rng)
        assert sorted(map(tuple, r.edges)) == sorted(map(tuple, pr.edges))
        assert r.depths() == d0 and m["moved"] > 0
        for j in range(E):
            assert r.edges[m["perm"][j]] == pr.edges[j]

        # unreachable-only reorder: reachable-source edges never move
        u, m = P.reorder_unreachable(pr, rng)
        if u is not None:
            avail["reorder_unreachable"] += 1
            assert u.depths() == d0 and m["moved"] > 0
            for j, (src, _) in enumerate(pr.edges):
                if src in d0:
                    assert u.edges[j] == pr.edges[j], "reachable edge moved"
                    assert m["perm"][j] == j

        # rename: depth map transposed, candidates and root renamed, no fixed point
        n, m = P.rename(pr, rng)
        sig = m["sigma"]
        assert all(sig[v] != v for v in pr.nodes())
        assert n.depths() == {sig[v]: dep for v, dep in d0.items()}
        assert n.cands == (sig[pr.cands[0]], sig[pr.cands[1]]) and n.root == sig[pr.root]
        assert n.target == sig[pr.target] and n.reachable_candidate() == n.target
        assert n.layout() == pr.layout()
        assert n.root in (0, 1) and n.root != pr.root, "renamed root must be the other name token"
        assert all(2 <= sig[v] <= 30 for v in pr.nodes() if v >= 2), "concepts stay concepts"

        # decoy swap: graph unchanged, swapped slots hold each other's edges
        ds, m = P.decoy_swap(pr)
        if ds is not None:
            avail["decoy_swap"] += 1
            assert ds.depths() == d0 and m["n_swapped"] >= 1
            for j, k in m["pairs"]:
                assert ds.edges[j] == pr.edges[k] and ds.edges[k] == pr.edges[j]
                assert pr.edges[j][1] == pr.target and pr.edges[k][1] == pr.decoy
            for j in range(E):
                if j not in {x for pair in m["pairs"] for x in pair}:
                    assert ds.edges[j] == pr.edges[j]
        else:
            avail["decoy_swap:" + m["reason"]] += 1

        # non-candidate swap: graph unchanged; watched nodes unreachable non-candidates
        ns, m = P.noncandidate_swap(pr)
        if ns is not None:
            avail["noncandidate_swap"] += 1
            assert ns.depths() == d0 and m["n_swapped"] >= 1
            for j, k in m["pairs"]:
                assert ns.edges[j] == pr.edges[k] and ns.edges[k] == pr.edges[j]
                assert pr.edges[j][1] == pr.target
                assert pr.edges[k][1] in m["watch"]
            for w in m["watch"]:
                assert w not in d0 and w not in (pr.target, pr.decoy)
        else:
            avail["noncandidate_swap:" + m["reason"]] += 1

        # parent swaps: two labels move, target depth kept, decoy unreachable
        for off, key in ((-1, "parent_swap_km2"), (0, "parent_swap_same_depth")):
            ps_, m = P.parent_swap(pr, rng, off)
            if ps_ is None:
                avail[key + ":" + m["reason"]] += 1
                continue
            avail[key] += 1
            dn = ps_.depths()
            assert dn[pr.target] == pr.K and pr.decoy not in dn
            assert d0[m["parent"]] == pr.K - 1 and d0[m["other"]] == m["depth_other"]
            changed = [j for j in range(E) if ps_.edges[j] != pr.edges[j]]
            assert all(m["parent"] in pr.edges[j] or m["other"] in pr.edges[j] for j in changed)

        # unique answer branch (subtraction set): counted, sibling is another root child
        ab = P.unique_answer_branch(pr)
        if ab is not None:
            avail["unique_answer_branch"] += 1
            v, sib = ab
            assert d0[v] == 1 and [pr.root, v] in pr.edges
            assert sib is None or (d0[sib] == 1 and sib != v)

        # rewrite at each depth: target unreachable, decoy at depth K, one slot changed
        for depth in range(1, pr.K + 1):
            rw, m = P.rewrite_at_depth(pr, depth, rng)
            key = f"rewrite_d{depth}" + ("" if depth < pr.K else "_last")
            if rw is None:
                avail[key + ":" + m["reason"]] += 1
                continue
            avail[key] += 1
            dn = rw.depths()
            assert pr.target not in dn and dn.get(pr.decoy) == pr.K
            assert rw.reachable_candidate() == pr.decoy
            diff = [j for j in range(E) if rw.edges[j] != pr.edges[j]]
            assert diff == [m["slot"]] and rw.edges[m["slot"]] == m["new_edge"]
            assert d0[m["old_edge"][1]] == depth
            if depth == pr.K:
                assert m["new_edge"][1] == pr.decoy and m["old_edge"][1] == pr.target

        # candidate swap: matches minimal_pairs.candidate_swap edge set
        cs, m = P.candidate_swap(pr)
        donor, reason = mp_candidate_swap({**s, "edges": [list(e) for e in s["edges"]]})
        assert (cs is None) == (donor is None), (gi, m, reason)
        if cs is not None:
            avail["candidate_swap"] += 1
            assert sorted(map(tuple, cs.edges)) == sorted(map(tuple, donor["edges"]))
            assert cs.cands == pr.cands and cs.reachable_candidate() == pr.decoy
            assert bfs_depths(cs.edges, cs.root).get(pr.decoy) == d0[pr.target]

        cov = P.covariates(pr)
        assert cov["K"] == pr.K and cov["n_branches"] >= 1

    print("counterfactuals: OK on 300 training graphs; availability:")
    for k in sorted(avail):
        print(f"  {k:40s} {avail[k]}")
    assert avail["decoy_swap"] >= 250, "decoy swap should be available on most graphs"
    assert avail["noncandidate_swap"] >= 250
    assert avail["rewrite_d%d_last" % 3] + avail["rewrite_d%d_last" % 4] >= 150
    print("PROMPTS: PASS")


if __name__ == "__main__":
    main()
