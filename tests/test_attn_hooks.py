"""attn_hooks.py and measure.py plumbing on a randomly initialised model.

No trained weights are needed: these checks are about indexing and
bit-exactness, not about what the model computes.

    .venv/bin/python tests/test_attn_hooks.py     # must print HOOKS: PASS
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from harness import VENDOR  # noqa: E402
from utils import set_seed  # noqa: E402
from fast_coconut import FastCoconut, FastSTokenizer  # noqa: E402
from attn_hooks import AttnHooks, ResidualHooks  # noqa: E402
import measure as M  # noqa: E402
import prompts as P  # noqa: E402


def make_runner():
    set_seed(0)
    tok = FastSTokenizer()
    base = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(str(VENDOR / "configs/symbol-2layer-8head-768dim.json"))
    )
    model = FastCoconut(
        base,
        tok.convert_tokens_to_ids("<|latent|>"),
        tok.convert_tokens_to_ids("<|start-latent|>"),
        tok.convert_tokens_to_ids("<|end-latent|>"),
        tok.eos_token_id,
    )
    model.eval()
    return SimpleNamespace(model=model, tok=tok, device=torch.device("cpu"), base=base)


def main():
    r = make_runner()
    test = json.load(open(VENDOR / "data/prosqa_test_graph_4_coconut.json"))
    prs = [P.Prompt.from_sample(s, P.pin_seed(gi + 1_000_000, 0)) for gi, s in enumerate(test[:4])]

    # 1. eager path vs default path: rounding-level difference only
    worst = 0.0
    for pr in prs:
        ids = pr.ids(r.tok)
        a = M.run_ids(r, ids)
        b = M.run_ids(r, ids, attn_eager=True)
        worst = max(worst, float((a - b).abs().max()))
    assert worst < 1e-4, worst
    print(f"eager vs default attention: max logit diff {worst:.2e} (rounding only)")

    # 2. hooks installed but inactive: identical to the eager path
    for pr in prs:
        ids = pr.ids(r.tok)
        b = M.run_ids(r, ids, attn_eager=True)
        with AttnHooks(r.base):
            c = M.run_ids(r, ids, attn_eager=True)
        assert torch.equal(b, c), "inactive hooks changed the logits"
        d = M.run_ids(r, ids)  # hooks removed: default path restored
        assert torch.equal(d, M.run_ids(r, ids))
    print("inactive hooks: OK (bit-exact with the eager path; removal restores default)")

    pr = prs[0]
    ids = pr.ids(r.tok)
    L = pr.layout()
    n = L["n"]

    # 3. recorded attention: one row per position, rows sum to 1, causal
    with AttnHooks(r.base) as h:
        h.record_weights = True
        h.record_kv = True
        ref = M.run_ids(r, ids, attn_eager=True)
        for layer in range(h.n_layers):
            for q in range(n):
                w = h.attention(layer, q)  # (heads, k_len)
                assert abs(float(w.sum(-1).mean()) - 1.0) < 1e-5
                assert float(w[:, q + 1:].abs().sum()) == 0.0, "attention beyond the query"
            k, v = h.kv[layer]
            assert k.shape[-2] == n and v.shape[-2] == n
        kv_saved = {layer: h.kv[layer] for layer in range(h.n_layers)}
    print("recording: OK (attention rows and full key/value cache for every position)")

    # 4. self patch of the whole cache: identical logits
    with AttnHooks(r.base) as h:
        pos = list(range(n))
        for layer in range(h.n_layers):
            h.add_kv_patch(layer, "k", pos, kv_saved[layer][0][:, pos, :])
            h.add_kv_patch(layer, "v", pos, kv_saved[layer][1][:, pos, :])
        out = M.run_ids(r, ids, attn_eager=True)
    assert float((out - ref).abs().max()) < 1e-5, "self cache patch changed the answer"
    # a foreign patch (another graph's keys at the edge slots) changes them
    other = prs[1].ids(r.tok)
    with AttnHooks(r.base) as h:
        h.record_kv = True
        M.run_ids(r, other, attn_eager=True)
        okv = dict(h.kv)
    with AttnHooks(r.base) as h:
        slots = [p for s in L["slots"] for p in s if p is not None]
        slots = [p for p in slots if p < min(n, len(other))]
        h.add_kv_patch(1, "k", slots, okv[1][0][:, slots, :])
        out2 = M.run_ids(r, ids, attn_eager=True)
    assert float((out2 - ref).abs().max()) > 1e-6, "foreign key patch had no effect"
    print("cache patch: OK (self patch exact; foreign patch acts)")

    # 5. mask: the answer position cannot attend to the candidates
    with AttnHooks(r.base) as h:
        h.record_weights = True
        h.add_mask(range(h.n_layers), [L["a"]], [L["c1"], L["c2"]])
        M.run_ids(r, ids, attn_eager=True)
        for layer in range(h.n_layers):
            w = h.attention(layer, L["a"])
            assert float(w[:, [L["c1"], L["c2"]]].abs().max()) < 1e-12
            w2 = h.attention(layer, L["root"])
            assert float(w2[:, [L["c1"], L["c2"]]].sum()) > 0, "mask leaked to other queries"
    with AttnHooks(r.base) as h:
        h.record_weights = True
        h.add_mask([0], [L["a"]], [L["c1"], L["c2"]], heads=[2, 5])
        M.run_ids(r, ids, attn_eager=True)
        w = h.attention(0, L["a"])
        assert float(w[[2, 5]][:, [L["c1"], L["c2"]]].abs().max()) < 1e-12
        others = [hh for hh in range(w.shape[0]) if hh not in (2, 5)]
        assert float(w[others][:, [L["c1"], L["c2"]]].sum()) > 0, "per-head mask leaked to other heads"
    print("mask: OK (whole-layer and per-head)")

    # 6. residual hooks: record then self-patch is exact; a foreign vector acts
    with ResidualHooks(r.base) as rh:
        rh.record = True
        ref2 = M.run_ids(r, ids)
        store = {lvl: dict(d) for lvl, d in rh.store.items()}
    assert set(store) == {0, 1, 2} and all(len(store[l]) == n for l in store)
    with ResidualHooks(r.base) as rh:
        for lvl in store:
            for pos, vec in store[lvl].items():
                rh.patches[(lvl, pos)] = vec
        out3 = M.run_ids(r, ids)
    assert float((out3 - ref2).abs().max()) < 1e-5, "self residual patch changed the answer"
    with ResidualHooks(r.base) as rh:
        rh.patches[(1, L["root"])] = torch.randn(768) * 5
        out4 = M.run_ids(r, ids)
    assert float((out4 - ref2).abs().max()) > 1e-6
    # level 0 at a latent position equals the recycled thought plus its position embedding
    cap = M.capture(r, ids)
    wpe = r.base.transformer.wpe.weight
    lat1 = L["latents"][0]
    assert torch.allclose(store[0][lat1], cap[0] + wpe[lat1], atol=1e-5)
    print("residual hooks: OK (exact self patch; level 0 at latent = thought + position embedding)")

    # 6b. MLP hooks: record, self-patch exact, foreign vector acts, inactive is exact
    from attn_hooks import MLPHooks
    with MLPHooks(r.base):
        assert torch.equal(M.run_ids(r, ids), ref2), "inactive MLP hooks changed the logits"
    with MLPHooks(r.base) as mh:
        mh.record = True
        M.run_ids(r, ids)
        mstore = {li: dict(d) for li, d in mh.store.items()}
    assert set(mstore) == {0, 1} and all(len(mstore[li]) == n for li in mstore)
    with MLPHooks(r.base) as mh:
        for li in mstore:
            for pos, vec in mstore[li].items():
                mh.patches[(li, pos)] = vec
        out7 = M.run_ids(r, ids)
    assert float((out7 - ref2).abs().max()) < 1e-5, "self MLP patch changed the answer"
    with MLPHooks(r.base) as mh:
        mh.patches[(1, L["a"])] = torch.randn(768) * 5
        out8 = M.run_ids(r, ids)
    assert float((out8 - ref2).abs().max()) > 1e-6
    print("MLP hooks: OK (inactive exact; self patch exact; foreign vector acts)")

    # 7. thought injection: self-transplant is exactly zero; donor changes it
    own = M.capture(r, ids)
    out5 = M.run_ids(r, ids, M.fixed(own, M.all_passes(pr.K)))
    assert torch.equal(out5, M.run_ids(r, ids)), "self-transplant is not exact"
    donor = M.capture(r, prs[2].ids(r.tok)) if prs[2].K == pr.K else M.capture(r, prs[3].ids(r.tok))
    out6 = M.run_ids(r, ids, M.fixed(donor, M.intermediates(pr.K)))
    assert not torch.equal(out6, out5)
    split = M.answer_split(out5, pr.target, pr.decoy)
    assert abs(split["p_target"] + split["p_decoy"] + split["p_other_node"] + split["p_other_token"] - 1) < 1e-5
    print("thought injection and answer split: OK")
    print("HOOKS: PASS")


if __name__ == "__main__":
    main()
