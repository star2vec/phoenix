# Latent Reasoning Knows Where It Is, Not How It Got There

Models can sidestep natural language and reason through latent chain of thought by bypassing the token-generation phase and feeding the last-layer hidden states back as the next input. Because the thought does not commit to a single token per pass, it can encode multiple candidate tokens at once. The superposition hypothesis treats it as a weighted sum of search branches, supported so far by theoretical construction and evidence that the branches are decodable. Readable does not mean used, so we causally test the theory in its own setting: a two-layer model trained from scratch on a graph reachability task, across four seeds. The thought is readable and necessary: we can decode almost perfectly which nodes the search has reached, and replacing it with an uninformative vector makes the model lose the correct answer. Branch arithmetic is not possible: subtracting a node that lies on a path leading to the winning candidate does not change the outcome, regardless of whether the node is defined by its embedding, by a probe, by the direction with the most influence on the answer, or by the direction used by the model’s own attention heads to identify the corresponding edge. In the last case, the attention on the edge is measurably gone, yet the answer survives. The thought encodes the nodes the search reached so far, not the path leading to them, and at the step before the final one it knows the winner, which the final feed-forward layer can decode. The answer can rely on two types of evidence: whether the winner is reachable or the other candidate is not. The COCONUT models fine-tuned from GPT-2 show none of this: thoughts can be zeroed with no change, no attention head follows edges by their content, and the winner is decodable before the first latent reasoning step. The ProsQA benchmark numbers its nodes in breadth-first order, and even though the model does not use the numbering to determine the winner, it cannot learn the task without it. Continuous thoughts encode their current position, not the path that led there.

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

On the Mac use `--device cpu` for all analysis runs: batch-one forwards on
this model take about 25 ms on CPU and 180 ms on MPS, and the MPS allocator
grows by hundreds of MB per few dozen forwards (a long run was killed for
memory). MPS pays off only for batched training.

## Experiment 7: fine-tuned GPT-2 COCONUT

The four headline measurements on released GPT-2 checkpoints (no training
here). Put a checkpoint at `ckpts/<run>/best.pt` (`gpt2_dilgren`:
connordilgren/gpt2-prosqa-coconut `checkpoint_40`; `gpt2_aswal`:
darpanaswal/coconut-gpt2-prosqa `checkpoint_best`) with its `SOURCE.json`,
and the original ProsQA files from facebookresearch/coconut under
`data/prosqa_original/` (sha256 in its manifest). The GPT-2 tokenizer with the
three latent tokens is snapshotted to `ckpts/gpt2_tokenizer/` on first use.
Drivers: `gpt2_eval.py` (regime check), `gpt2_cells.py --part heads|cells`,
`winner_probe.py --model gpt2|symbol`, `gpt2_bases.py --part probe|jlens`;
all default to `--device cpu`. Test: `tests/test_gpt2_plumbing.py` (tiny
random model, real tokenizer; must print `GPT2: PASS`).
