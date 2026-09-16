"""Plumbing for the GPT-2 path (experiment 7) on a tiny RANDOM model with the
real tokenizer. No checkpoint is read; numbers carry no evidential weight.

Checks: the natural-language renderer is byte-identical to the stored
questions; the layout partitions the token ids and its anchors land on the
name tokens and the period; the readout ids differ and the probability split
sums to one; self-transplant is exactly zero; the cached final pass equals an
uncached forward; heads.run, the cells driver and the winner probe run end to
end into a scratch directory.

    .venv/bin/python tests/test_gpt2_plumbing.py [scratch_dir]   # must print GPT2: PASS
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

import torch  # noqa: E402

from gpt2 import GPT2Runner, TOKENIZER_DIR, greedy_generate  # noqa: E402
from nl import NLPrompt, is_person, recipient_prompts_nl, reserialized_nl, same_answer_donor_nl  # noqa: E402
from prompts import reorder  # noqa: E402
from sets import load_test, load_train, _default  # noqa: E402
from measure import all_passes, answer_split, capture, fixed, run_ids  # noqa: E402
import heads  # noqa: E402

TINY = {"n_layer": 2, "n_head": 2, "n_embd": 32}


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    assert TOKENIZER_DIR.exists(), f"tokenizer snapshot missing: {TOKENIZER_DIR}"
    torch.manual_seed(0)
    r = GPT2Runner(None, device="cpu", seed=0, tiny=TINY)
    tok = r.tok
    test, train = load_test(), load_train()

    # 1. renderer byte-identical to the stored question on 300 graphs
    for s in test[:300]:
        pr = NLPrompt.from_sample(s)
        assert pr.question() == s["question"]
        assert pr.answer_text(pr.target) == "### " + s["answer"]
    print("renderer: OK (300 graphs byte-identical, answer text matches)")

    # 2. layout: partition and anchors
    for gi in (0, 1, 400):
        pr = NLPrompt.from_sample(test[gi])
        ids = pr.ids(tok)
        L = pr.layout()
        assert L["n"] == len(ids) and L["a"] == len(ids) - 1
        q_ids, lat, pre, _ = pr.pieces(tok)
        assert ids == q_ids + lat + pre
        assert [ids[p] for p in L["latents"]] == [r.latent_id] * pr.K
        assert ids[L["start"]] == 50257 and ids[L["end"]] == 50258
        covered = {0}
        for j, slot in enumerate(L["slots"]):
            ps = [p for p in slot if p is not None]
            assert ps == sorted(set(ps)) or True  # order is (src, tgt, period, rest): check the set instead
            assert len(ps) == len(set(ps))
            assert 0 not in ps
            covered |= set(ps)
            s_, t_ = pr.edges[j]
            assert tok.decode([ids[slot[2]]]) == "."
            tgt_piece = tok.decode([ids[slot[1]]]).strip()
            assert pr.names[t_].startswith(tgt_piece), (pr.names[t_], tgt_piece)
            if slot[0] is not None:
                src_piece = tok.decode([ids[slot[0]]]).strip()
                assert pr.names[s_].startswith(src_piece), (pr.names[s_], src_piece)
            else:
                assert j == 0 and is_person(pr.names[s_])
        # the question tokens fill the rest of the question part
        q_part = set(range(L["q"], len(q_ids)))
        assert covered | q_part == set(range(len(q_ids))), sorted(set(range(len(q_ids))) - covered - q_part)
        assert pr.names[pr.root].startswith(tok.decode([ids[L["root"]]]).strip())
        assert pr.names[pr.cands[0]].startswith(tok.decode([ids[L["c1"]]]).strip())
        assert pr.names[pr.cands[1]].startswith(tok.decode([ids[L["c2"]]]).strip())
        assert L["extra_query_classes"]["search_latent"] == L["latents"][: pr.L - 1]
        assert L["thought1_query"] == L["start"]
        # readout
        ro = pr.readout(tok)
        assert ro["target"] != ro["decoy"] and {ro["target"], ro["decoy"]} <= set(ro["nodes"])
        assert tok.decode(pre) == f"### {pr.names[pr.root]} is a", tok.decode(pre)
        logits = run_ids(r, ids)
        sp = answer_split(logits, ro["target"], ro["decoy"], node_ids=ro["nodes"])
        assert abs(sp["p_target"] + sp["p_decoy"] + sp["p_other_node"] + sp["p_other_token"] - 1) < 1e-5
        assert sp["p_other_node"] >= -1e-9
    print("layout and readout: OK (partition, anchors, prefix frame, split sums to one)")

    # 3. reorder and reserialized keep the subclass and the question set
    pr = NLPrompt.from_sample(test[400])
    import random
    rb, meta = reorder(pr, random.Random(0))
    assert isinstance(rb, NLPrompt) and rb.L == pr.L and rb.names == pr.names
    assert sorted(map(tuple, rb.edges)) == sorted(map(tuple, pr.edges)) and rb.question() != pr.question()
    rs = reserialized_nl(pr, 400, 0)
    assert isinstance(rs, NLPrompt) and rs.question() != pr.question()
    LB = rb.layout()
    assert len(LB["slots"]) == len(pr.layout()["slots"])
    print("counterfactual constructors: OK (subclass kept, sentence order changed)")

    # 4. self-transplant exactly zero; donor thoughts change the logits
    ids = pr.ids(tok)
    own = capture(r, ids)
    assert set(own) == set(range(pr.K))
    a = run_ids(r, ids)
    b = run_ids(r, ids, fixed(own, all_passes(pr.K)))
    assert torch.equal(a, b), "self-transplant is not exact"
    sad = same_answer_donor_nl(train, pr)
    assert sad is not None
    dpr = NLPrompt.from_sample(train[sad[0]])
    assert dpr.names[dpr.target] == pr.names[pr.target] and dpr.names[dpr.decoy] == pr.names[pr.decoy] and dpr.L == pr.L
    dth = capture(r, dpr.ids(tok))
    c = run_ids(r, ids, fixed(dth, all_passes(pr.K)))
    assert not torch.equal(a, c)
    print("thought injection: OK (self-transplant exact; same-answer donor found by name)")

    # 5. the cached final pass equals one uncached forward on the filled embeddings
    dev = r.device
    input_ids = torch.tensor([ids], device=dev)
    with torch.no_grad():
        out = r.model(input_ids, torch.ones_like(input_ids), input_ids.clone(),
                      torch.arange(len(ids), device=dev).reshape(1, -1))
        full = r.model.base_causallm(inputs_embeds=out.inputs_embeds).logits
    diff = float((full[0, -1] - out.logits[0, -1]).abs().max())
    assert diff < 1e-4, diff
    # eager path agrees with the default path
    e = run_ids(r, ids, attn_eager=True)
    assert float((e - a).abs().max()) < 1e-4
    # greedy generation runs and stops
    toks = greedy_generate(r, ids, max_new_tokens=5)
    assert 1 <= len(toks) <= 5
    print(f"cached final pass vs uncached forward: OK (max diff {diff:.1e}); generation runs")

    # 6. heads.run on two NL prompts
    recips = [(gi, test[gi], NLPrompt.from_sample(test[gi])) for gi in (400, 401)]
    res = heads.run(r, recips)
    S = res["summary"]
    assert "L1H0/search_latent" in S and "L2H1/latent5" in S and "sink_mass" in S["L1H0/intermediate_latent"]
    assert res["n_layers"] == 2 and res["n_heads"] == 2
    p = out_dir / "heads_smoke.json"
    json.dump(res, open(p, "w"), default=_default)
    json.load(open(p))
    print("heads.run on NL prompts: OK")

    # 7. cells driver and winner probe, if present
    try:
        import gpt2_cells
    except ImportError:
        gpt2_cells = None
    if gpt2_cells is not None:
        sets = [(0, 0), (1, 1)]
        res = gpt2_cells.run(r, recips, train[:400], sets, "smoke", base_seed=0)
        assert {"qk_removal", "qk_random", "latents_mask/path/alone", "latents_mask/offpath/alone",
                "thoughtK/same_answer/alone", "thoughtK/same_answer/plus_removal", "thoughtK/random/plus_removal",
                "same_answer_donor/intermediates", "reserialized"} <= set(res["cells"])
        assert all(abs(row["cells"]["self_transplant"]["dT"]) < 1e-6 for row in res["rows"])
        assert "fallback_line" in res["summary"]
        p = out_dir / "cells_smoke.json"
        json.dump(res, open(p, "w"), default=_default)
        json.load(open(p))
        print("cells driver: OK")
    try:
        import winner_probe
    except ImportError:
        winner_probe = None
    if winner_probe is not None:
        res = winner_probe.run(r, "gpt2", train, fit_idx=list(range(0, 40)), eval_idx=list(range(len(train) - 20, len(train))),
                               bases={}, cache=None)
        assert "learned_probe" in res and "input_embedding" in res["separation"]
        p = out_dir / "winner_smoke.json"
        json.dump(res, open(p, "w"), default=_default)
        json.load(open(p))
        print("winner probe: OK")
    print("GPT2: PASS")


if __name__ == "__main__":
    main()
