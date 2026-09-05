# phoenix

What does a continuous thought encode? COCONUT-style models reason in latent
chain-of-thought: the last hidden state is fed back as the next input embedding
instead of a token. On graph reachability, a two-layer model trained from
scratch this way is known to produce thoughts whose inner products with node
embeddings trace a breadth-first wave over the graph. This project studies what
those intermediate thoughts actually carry, by reading them out, editing them,
and transplanting them between problems, and measuring what the answer does.

## Setup

Python 3.12 with pinned packages, and the reference implementation cloned into
`vendor/` (gitignored). On the Mac:

```bash
git clone --depth 1 https://github.com/Ber666/reasoning-by-superposition.git vendor/reasoning-by-superposition
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1 numpy==2.1.3 transformers==4.46.2 datasets==3.1.0 tqdm==4.67.0 pyyaml
```

For the CUDA laptop see `WINDOWS_SETUP.md`. Checkpoints go to `ckpts/<run>/`
(gitignored); metrics and result files go to `results/<run>/`.

## Train

```bash
.venv/bin/python -u src/phoenix/train.py --seed 0 --run-name seed0 \
  --device cuda --fixed-early-stages --full-task-patience 15
```

Resumable: rerun the same command after an interruption. Then evaluate the
best checkpoint (held-out accuracy and the readout ordering):

```bash
.venv/bin/python src/phoenix/evaluate.py --run-name seed0 --device cuda
```

## Run a test

The fast forward pass must match the reference code bit for bit. Run this on
CPU after any change to `src/phoenix/fast_coconut.py` and before training:

```bash
.venv/bin/python tests/test_fast_equivalence.py     # must print EQUIVALENCE: PASS
.venv/bin/python tests/test_minimal_pairs.py        # must print MINIMAL_PAIRS: PASS
.venv/bin/python tests/test_prompts.py              # must print PROMPTS: PASS
.venv/bin/python tests/test_attn_hooks.py           # must print HOOKS: PASS
.venv/bin/python tests/test_drivers_smoke.py        # must print SMOKE: PASS (random weights)
```

The experiment drivers (`necessity.py`, `heads.py`, `counterfactuals.py`,
`tracing.py`, `cache_patch.py`, `baseline.py` in `src/phoenix/`) take
`--run-name`, `--device`, and `--mode pilot|n100`; each stops with a message
if `ckpts/<run>/best.pt` is missing. Plan and predictions live in `NOTES.md`.
