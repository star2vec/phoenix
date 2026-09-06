"""Plumbing test for every driver on RANDOM weights and two graphs.

This is not a pilot: no checkpoint is read, results go to a scratch
directory, and the numbers carry no evidential weight. It checks that each
driver runs end to end, that the standing controls behave (self-transplant
and self cache patch exactly zero), that every cell is present, and that the
output is JSON-serialisable.

    .venv/bin/python tests/test_drivers_smoke.py [scratch_dir]   # must print SMOKE: PASS
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phoenix"))

import torch  # noqa: E402

from harness import Runner  # noqa: E402
from sets import load_train, load_test, test_pin, _default  # noqa: E402
from prompts import Prompt  # noqa: E402
import necessity, heads, counterfactuals, tracing, cache_patch, baseline  # noqa: E402


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    runner = Runner(None, device="cpu", seed=0)  # random init: plumbing only
    train = load_train()[:400]
    test = load_test()
    recips = [(gi, test[gi], Prompt.from_sample(test[gi], test_pin(gi, 0))) for gi in (400, 401)]

    def dump(name, result):
        p = out_dir / f"{name}_smoke.json"
        json.dump(result, open(p, "w"), default=_default)
        json.load(open(p))
        print(f"  wrote {p}")

    print("necessity")
    means = necessity.thought_means(runner, train, n=20)
    res = necessity.run(runner, recips, train, means)
    assert all(res["rows"][i]["cells"]["self_transplant"]["dT"] == 0.0 for i in range(2))
    assert {"average/all", "noise/intermediates", "zero/all", "random_donor/all", "removed",
            "removed_length_kept", "reserialized", "same_answer_donor/intermediates",
            "same_answer_donor/all"} <= set(res["cells"])
    assert res["summary"]["zero/all"]["n"] == 2
    dump("necessity", res)

    print("heads")
    res = heads.run(runner, recips)
    key = "L2H0/intermediate_latent"
    assert key in res["summary"] and "position_score" in res["summary"][key]
    assert "L1H0/edge_token" in res["summary"]
    dump("heads", res)

    print("counterfactuals")
    res = counterfactuals.run(runner, recips, train)
    for v in ("free", "intermediates", "all"):
        assert f"reordered/{v}" in res["cells"] and f"renamed/{v}" in res["cells"]
        assert f"noncandidate_swap/{v}" in res["cells"]
    assert "same_answer_donor/intermediates" in res["cells"]
    r0 = res["rows"][0]["cells"]
    assert r0["self_transplant"]["dT"] == 0.0
    assert "qk_subtract/answer_edge/all_heads" in res["cells"] and "qk_subtract/random_matched/head0" in res["cells"]
    qk = res["rows"][0]["cells"]["qk_subtract/answer_edge/all_heads"]
    assert "attn_answer_total_before" in qk and "dT" in qk
    assert "qk_subtract/answer_edge/all_heads" in res["summary"]["qk_attention"]
    m = res["rows"][0]["meta"]["qk"]
    assert len(m["coef_frac_per_head"]) == 8 and isinstance(m["query_position"], int)
    assert m["edit_pass"] == res["rows"][0]["K"] - 2
    nc = [r["cells"]["noncandidate_swap/all"] for r in res["rows"] if not r["cells"]["noncandidate_swap/all"].get("skipped")]
    assert nc and "p_watch" in nc[0] and 0.0 <= nc[0]["p_watch"] <= 1.0
    if res["summary"]["reordered/intermediates"].get("redirection"):
        assert "frac_redirected" in res["summary"]["reordered/intermediates"]["redirection"]
    # a renamed prompt with thoughts free is a valid new problem: its own target field is used
    assert "T" in r0["renamed/free"]
    dump("counterfactuals", res)

    print("tracing (slot granularity, one corruption, for speed)")
    res = tracing.run(runner, recips, granularity="slot", corruptions=("candidate_swap",))
    t = res["rows"][0]["traces"]["candidate_swap"]
    assert not t.get("skipped")
    # restoring an unchanged token at level 0 is a no-op: T equals the corrupted T
    unchanged = [lab for lab in t["restored_T"] if lab.startswith("slot_") and
                 int(lab.split("_")[1]) not in t["differing_slots"]]
    assert unchanged, "no unchanged slot"
    assert abs(t["restored_T"][unchanged[0]][0] - t["corrupted"]["T"]) < 1e-6
    dump("tracing", res)

    print("cache_patch")
    res = cache_patch.run(runner, recips, train)
    assert abs(res["rows"][0]["cells"]["self_patch/edges/kv/both"]["dT"]) < 1e-6
    assert "donor/edges/k/L2" in res["cells"] and "random_graph/latents_intermediate/kv/both" in res["cells"]
    dump("cache_patch", res)

    print("baseline (random bases)")
    probe = torch.randn(40, 768)
    res = baseline.run_subtraction(runner, recips, train, probe)
    assert "subtract_answer/input_embedding" in res["cells"] or res["skipped_no_unique_ancestor"]
    assert "same_answer_donor/intermediates" in res["cells"] or res["skipped_no_unique_ancestor"]
    res = baseline.run_transplant(runner, list(enumerate(train))[:2], train)
    assert "matched_donor/intermediates" in res["cells"] and "label_swap/intermediates" in res["cells"]
    assert "same_answer_donor/intermediates" in res["cells"]
    jb = torch.randn(4, 40, 768)
    res = baseline.run_swap(runner, list(enumerate(train))[:2], jb)
    assert "swap_final" in res["cells"]
    dump("baseline", res)
    print("SMOKE: PASS")


if __name__ == "__main__":
    main()
