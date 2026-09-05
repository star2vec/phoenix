"""Graph sets, pinned seeds, checkpoint check, result paths.

Recipients for every new experiment come from the paper's held-out test
split (419 graphs, never trained on). Pilot = test graphs 400-409; n=100 =
test graphs 0-99. Donors, averages and fits use training graphs.

Pinned serialization: test graph gi uses pin_seed(gi + TEST_OFFSET, base);
training graph gi uses pin_seed(gi, base). The offset keeps the two
namespaces apart (as in the paper's held-out replication).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "reasoning-by-superposition"
TRAIN_PATH = VENDOR / "data" / "prosqa_train_graph_4_coconut.json"
TEST_PATH = VENDOR / "data" / "prosqa_test_graph_4_coconut.json"
EVAL_GRAPHS = ROOT / "data" / "eval_graphs.json"

TEST_OFFSET = 1_000_000
PILOT = list(range(400, 410))
N100 = list(range(0, 100))
MODES = ("pilot", "n100")


def load_train():
    return json.load(open(TRAIN_PATH))


def load_test():
    return json.load(open(TEST_PATH))


def load_eval_graphs():
    return json.load(open(EVAL_GRAPHS))


def recipients(mode):
    """[(gi, sample)] from the test split for the given mode."""
    assert mode in MODES, mode
    data = load_test()
    idx = PILOT if mode == "pilot" else N100
    return [(gi, data[gi]) for gi in idx]


def test_pin(gi, base_seed, reserial=False):
    from prompts import pin_seed
    return pin_seed(gi + TEST_OFFSET, base_seed, reserial)


def train_pin(gi, base_seed, reserial=False):
    from prompts import pin_seed
    return pin_seed(gi, base_seed, reserial)


def require_checkpoint(run_name, name="best.pt"):
    """Return the checkpoint path, or stop with a message. Checkpoints come
    from the laptop; nothing is trained on this machine."""
    p = ROOT / "ckpts" / run_name / name
    if not p.exists():
        print(
            f"[stop] checkpoint not found: {p}\n"
            f"       Copy {name} for run '{run_name}' from the laptop into ckpts/{run_name}/ "
            f"and rerun. No training is launched from here."
        )
        sys.exit(2)
    return p


def require_file(path, produced_by):
    p = Path(path)
    if not p.exists():
        print(f"[stop] missing {p}; run `{produced_by}` first.")
        sys.exit(2)
    return p


def results_dir(run_name):
    d = ROOT / "results" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_file(run_name, name, mode):
    return results_dir(run_name) / f"{name}_{mode}.json"


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    print(f"written: {path}")


def _default(o):
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
