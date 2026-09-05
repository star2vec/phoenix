# Notes

Running notes for the position-vs-identity study. A fresh session should be
able to pick up from this file alone. Literature is in `lit/NOTES.md`; the
preliminary paper is `paper/v1.pdf`. Last updated 2026-09-06.

## Status

- Code for experiments 0-4 is written and tested (`tests/`: bit-exact
  equivalence with hooks idle, prompt renderer byte-identical to the vendor
  builder, hook plumbing, and every driver end to end on random weights; the
  smoke test writes to a scratch directory, not to `results/`). No
  experiment has run. `scripts/make_numbers.py` macros get registered once
  result files exist. No trained checkpoint is on this
  machine: `ckpts/seed0/best.pt` and `ckpts/seed1/best.pt` come from the
  laptop. Every driver checks for its checkpoint and stops with a message if
  it is missing.
- Retraining with periodic checkpoints (`train.py --save-every`) happens on
  the laptop when the user says so. Claude never launches training or a
  pilot.

## Established (from paper/v1.pdf, treated as data)

- A two-layer model trained from scratch on graph reachability with K
  continuous thoughts reaches 90-96 held-out accuracy depending on seed, and
  the inner-product readout orders nodes optimal > frontier > reachable >
  not reachable at every latent step.
- The search frontier is linearly readable from the thought (probe AUC
  0.998). The thought is necessary (zeroing at every step: -42.9 median
  change in T).
- Per-branch linear edits (subtract, collapse, inject) do nothing in the
  input-embedding, probe, or causal-Jacobian basis, even with the full
  linearly readable subspace removed.
- Whole-thought transplants at every intermediate step redirect the answer
  when the donor is a different graph with the same candidates (-91.7) and do
  nothing when the donor is the same graph with the candidate labels swapped
  (-0.0). A random donor destroys rather than redirects (e = 0.98).
- Swapping the target and decoy directions in the causal-Jacobian basis at
  the final step flips the answer (-99.97); the same swap at intermediate
  steps does nothing.

One reframing, written 2026-09-06 before any new run: under both stories
below, the candidates are absent from every intermediate frontier (they sit
at depth K; the step K-1 frontier is their parents). So both stories predict
the paper's label-swap null. That result did not separate them.

## Open: what does a mid-search thought encode?

The position story: the thought remembers where to look in the prompt (which
edge slots are consumed, which to read next); node names are read off the
prompt at the last step. The identity story: the thought remembers which
nodes are in the frontier, in a form the paper's linear edits could not
reach. A hybrid is possible: positional mid-search, identity at the final
step.

Two terms the paper does not use. An edge slot is the three-token span one
edge occupies in the prompt (`source target |`). The memory cache is the set
of keys and values the model stores at each position, which later positions
read through attention.

## Working rules

- Predict only what a story forces; one line per story per cell; "no
  prediction" cells run as exploration. Never fill a table for completeness.
- Pilot on the 10 pilot graphs; write here what the pilot changed; then run
  n=100 once on both seeds.
- Effect sizes with 95% bootstrap intervals (2,000 resamples of graphs). Two
  counts with named cutoffs: flipped (change in T at or below -50; protects
  "the answer switched to the other candidate on this graph") and escaped
  (e at or above 0.5; protects "the model stopped answering the question").
- Standing controls on every run: self-transplant (must be exactly zero;
  catches pass misalignment), reserialized baseline (a second fixed edge
  order, nothing fixed: the noise floor for "the prompt changed and the
  thought recomputed"), random donor (a random training graph's thoughts of
  the same step count: what breaking looks like on a vector the model
  actually produced). No zero vectors as sensitivity anchors.
- Every cell reported, nulls included. Numbers in prose point to a results
  file.
- For any cell where graphs split (some flipped or escaped, others not),
  compare the two groups on the depth of the target (K), the number of
  branches (root out-degree), the slot of the target's parent edge in the
  list (absolute and as a fraction of the list), and the number of parent
  edges into the target. Report that split (`stats.split_by_flag`).

## Sets and measures

- Recipients: held-out test split (419 graphs, never trained on). Pilot =
  test graphs 400-409. n=100 = test graphs 0-99. Serialization pinned per
  graph with seed `((gi + 1000000) << 16) ^ base_seed`; reserialized
  baseline adds 0xABCDEF (same formula as `thoughts.pin_serialization`).
- Donors, averages, probe and Jacobian fits: training graphs.
- Experiment 0 keeps the paper's sets: `data/eval_graphs.json` graphs 0-99
  for the per-branch subtraction, training graphs 0-99 for transplants and
  the final-step swap.
- T = 100 x p(target) / (p(target) + p(decoy)) at the answer position;
  change in T is relative to the same graph's pinned baseline; e = 1 -
  p(target) - p(decoy). Every cell also reports where the probability went:
  target, decoy, other node token, non-node token.
- For a renamed prompt, "target" means the target's new label. For a
  rewritten prompt (where the decoy becomes the right answer), "target" stays
  the original target, so following the edit shows as T falling to 0.
- "Intermediates fixed" injects the original run's thoughts at passes
  0..K-2. "All K fixed" also injects the final one. Both variants are run
  where the plan says so.

## Experiment 0: setup and baseline

Design: retrain two seeds on the laptop (`train.py --seed {0,1}
--fixed-early-stages --full-task-patience 15 --save-every 10`), then on each
model: `evaluate.py` (accuracy, ordering), `fit_probes.py` (probe AUC),
`fit_jlens.py` (causal-Jacobian basis), and `baseline.py` (per-branch
subtraction with random-direction control; transplants: matched donor,
label-swap donor, placebo donor, random donor, self, reserialized, at
intermediate steps, first step only, final step only, all steps; final-step
swap and intermediate swaps in the causal-Jacobian basis).

Predictions (not story-specific; this is the baseline the next paper needs):
accuracy in the 90-96 range, ordering correct at every step, probe AUC near
0.998, subtraction near zero with control near zero, matched-donor transplant
a large negative change in T with low e, label-swap near zero, random donor
with high e, final-step swap near -100 and intermediate swaps near zero.
Position story: all of the above. Identity story: all of the above.

Pilot outcome: (not run)
What the pilot changed: (not run)
n=100 outcome: (not run)

## Experiment 1: necessity, done properly

Design: seven conditions at every step of the same pinned graph, each run
twice (all K passes; intermediate passes only): baseline; the average
thought over 2,000 training graphs at the same pass (present but
uninformative); Gaussian noise at the graph's own thought norm; the thought
removed (latent tokens deleted; the prompt ends `[R] root [A]`); removed
with length kept (latent tokens replaced by the pad token, attended); zero;
random donor. Report the change in T, e, and the probability split.

Position story: no separating prediction. Identity story: no separating
prediction. Shared expectation from the paper's transplant results: every
uninformative substitute breaks the answer (the thought is informative, not
a scaffold as in Kshirsagar's chess model). The two removal conditions are
prompts the model never saw in training; they are reported as such.

Pilot outcome: (not run)
What the pilot changed: (not run)
n=100 outcome: (not run)

## Experiment 2: which heads look by position and which by content

Design: for each of the 16 heads and each query position (edge tokens;
intermediate latents; last latent; answer position), record attention over
edge slots in two runs of the same graph: the original order, and a random
reorder of the edge list with the original thoughts fixed at all K passes.
Position score = correlation of attention over slots, slot for slot. Content
score = the same after undoing the permutation, edge for edge. Also the
share of attention mass that lands on edge slots at all, so heads that do
not read edges are not over-interpreted.

Position story: the layer-2 heads that carry most attention from the
intermediate latents onto edge slots keep their slots (position score near
1, content score near chance). Identity story: those heads follow the edges
(content score near 1, position score near chance). Both stories: layer-1
heads at edge tokens attend within their own slot (Zhu et al.'s copy). A
split between intermediate latents and the last latent or answer position is
the hybrid signature.

Pilot outcome: (not run)
What the pilot changed: (not run)
n=100 outcome: (not run)

## Experiment 3: the counterfactual set (thought fixed, one thing changed)

Design: every cell uses `prompts.py`, which renders the vendor format byte
for byte with an explicit edge order. Each changed prompt is run three ways:
thoughts free (that cell's reserialized-type baseline), intermediates fixed,
all K fixed. Controls per graph: reserialized baseline, self-transplant,
random donor at intermediates and at all K.

Cells and predictions:

- Edges reordered (random permutation of slots), intermediates fixed.
  Position story: breaks; pointers land on other edges; e rises toward the
  random-donor level. Identity story: no effect, within the reserialized
  baseline. Separating, primary.
- Labels renamed (one consistent relabeling of every node token, edge order
  kept), intermediates fixed. Position story: no effect (slots unchanged,
  candidates renamed too). Identity story: breaks (the thought names old
  labels that now denote other nodes). Separating, primary; mirror of the
  cell above.
- Unreachable edges reordered among themselves (no reachable edge moves),
  intermediates fixed. Position story: no effect. Identity story: no effect.
  Fragility control: if this breaks, the thought is coupled to prompt content
  neither story needs.
- Decoy swap (the edge into the target and an edge into the decoy trade
  slots; the graph is unchanged), all K fixed. Position story: clean flip to
  the decoy with low e (complete only when every parent edge of the target
  was swapped; `n_swapped` and `complete` are recorded per graph). Identity
  story: no effect. Separating for the final thought.
- Decoy swap, intermediates fixed. Position story: no effect (the last hop is
  re-read from the prompt). Identity story: no effect. Consistency.
- Last-hop rewrite (the cut edge into the target now points to the decoy,
  same slot; BFS confirms the target is unreachable and the decoy is at depth
  K), all K fixed. Position story: follows the edit (T falls to 0). Identity
  story: keeps the old answer. Separating for the final thought.
- Single-edge rewrite at depth d, thoughts free. Both stories: follows the
  edit. Competence check (Jin et al.).
- Single-edge rewrite at depth d, intermediates fixed, stratified by d.
  Both stories: old answer if d < K-1, new answer if d = K-1. Exploration:
  how much of the search is committed by step K-1.

Availability (from `tests/test_prompts.py` on training graphs 0-299, so the
n=100 cells will have skips): reorder, rename, unreachable-only reorder,
decoy swap and candidate swap are constructible on every graph; the last-hop
rewrite on 201 of 300 (a cut edge into the target is needed); rewrites at
depth 1 on 74, depth 2 on 97, depth 3 (when K=4) on 73. Skips are counted
per cell in the results file.

Reading of the four outcomes of the mirror pair (written before the pilot):

1. Reordered breaks, renamed does not: the mid-search thought is positional.
   The model reads slots and dereferences names later. Then experiment 2
   should show the thought-reading layer-2 heads as positional, and the
   keys-only cache patch (experiment 4b) should do nothing.
2. Renamed breaks, reordered does not: the mid-search thought names nodes.
   The paper's linear edits missed the code, not the content. Then
   experiment 2 should show content heads, and the keys-only cache patch
   should break the answer.
3. Both break. Two readings, separated by the escape levels. Plain
   fragility: any mismatch between a fixed thought and the prompt destroys
   the answer, so e in both cells sits at the random-donor level and the
   unreachable-edges reorder (which neither story cares about) also breaks.
   Mixed code: the thought carries both cues and the readout needs both, so
   removing one cue degrades rather than destroys: e sits between the
   reserialized baseline and the random-donor level, T moves toward 50
   rather than to 0, and the unreachable-edges reorder stays at the
   reserialized baseline. If both cells destroy but the unreachable-edges
   reorder does not, the code is mixed and both cues are load-bearing.
4. Neither breaks. The fixed intermediate thoughts are not what the final
   steps consume, at least not slot-wise or name-wise. Check the random
   donor at intermediates first: if it breaks (as in the paper) while
   neither counterfactual does, the thought carries something invariant to
   both edge order and labels, an unlabeled search shape, and the all-K
   cells and experiment 4 become the main line. If the random donor also
   fails to break, the sustained-transplant result of the paper did not
   replicate on this model and experiment 0 needs a second look before
   anything else.

Pilot outcome: (not run)
What the pilot changed: (not run)
n=100 outcome: (not run)

## Experiment 4: where the effect lives

4a, causal tracing by position. Corrupt the prompt with an on-manifold
counterfactual whose answer differs (candidate swap: target and decoy tokens
transposed in the edge list, question line unchanged; and the last-hop
rewrite), run it, then restore the clean run's residual stream at one
position and one level at a time (level 0 = input to layer 1, which for a
latent position is the recycled thought plus its position embedding; level 1
= after layer 1; level 2 = after layer 2) and record T. Compare edge slots
(source token, target token, separator), the question and root tokens,
latent positions, and the answer position.

Position story: no separating prediction. Identity story: no separating
prediction. Shared expectation: under the candidate swap the intermediate
latent positions carry no effect (their thoughts are near-identical in the
paper, cosine 0.9975); the effect sits at the edge slots holding the
candidates and at the final thought, last latent, and answer position. This
is the map that tells 4b which slices to patch. Conditional follow-up if
experiment 3's reordered cell breaks: un-fix one intermediate thought at a
time to find the step where the pointer bites.

4b, memory-cache patching. Donor: the same graph with edges reordered, run
naturally. Recipient: original order, its own thoughts. Patch the donor's
keys and/or values at chosen positions into the recipient's cache at every
pass.

- Keys only, edge slots, layer 2. Position story: no effect (the position
  part of each key is unchanged, so attention stays put and reads the
  recipient's own values). Identity story: breaks (keys now carry other
  edges' source nodes; attention follows them to slots whose values are
  different edges). Separating.
- Values only, edge slots, layer 2. Both: breaks. Consistency.
- Keys and values, latent positions 1..K-1. Position story: breaks. Identity
  story: no effect. Separating; the cache-level twin of experiment 3's
  primary cell.
- The same three cells in layer 1: no prediction from either story;
  exploration.

For any carrier claimed, three checks (Ding et al.): the right patch moves
the answer; the other slice from the same donor does not; replacing the
slice with the same slice from a random graph's run breaks the recipient.
Plus the self-patch (recipient's own cache back into itself), which must be
exactly zero.

Pilot outcome: (not run)
What the pilot changed: (not run)
n=100 outcome: (not run)

## Experiment 5: later, only if 1-4 favor one story

Mask the answer position's (and separately every latent's) attention to the
two candidate tokens (`attn_hooks.AttnHooks.add_mask`); checkpoints along
training (saved by `train.py --save-every`); three- and four-layer models.
Designs and predictions to be written after 1-4 report.

## Next run

When `ckpts/seed0/best.pt` is present (device `cuda` on the laptop, `mps`
here):

```
.venv/bin/python src/phoenix/evaluate.py --run-name seed0 --device mps
.venv/bin/python src/phoenix/fit_probes.py --run-name seed0 --device mps
.venv/bin/python src/phoenix/fit_jlens.py --run-name seed0 --device mps
.venv/bin/python src/phoenix/baseline.py --run-name seed0 --device mps --mode pilot
```

Then, each after the user says so and after its predictions above are
re-read: `necessity.py`, `heads.py`, `counterfactuals.py`, `tracing.py`,
`cache_patch.py`, each with `--mode pilot`, then `--mode n100` once on both
seeds. Result files: `results/<run>/<driver>_<mode>.json`.

## Code map

- `src/phoenix/prompts.py`: explicit prompt renderer (byte-identical to the
  vendor builder), slot layout, counterfactual constructors (reorder,
  unreachable-only reorder, rename, decoy swap, rewrite at depth, label
  swap), per-graph covariates.
- `src/phoenix/attn_hooks.py`: `AttnHooks` (record attention, overwrite
  cached keys/values by layer/head/position, mask query-key pairs; needs
  `attn_eager=True` on the forward) and `ResidualHooks` (record or overwrite
  the residual stream by level and position).
- `src/phoenix/measure.py`: one forward on explicit token ids, answer split,
  thought capture and injection helpers.
- `src/phoenix/stats.py`: bootstrap intervals, cell summaries, the split
  analysis.
- `src/phoenix/sets.py`: graph sets, pinned seeds, checkpoint check, result
  paths.
- `src/phoenix/fast_coconut.py`: `attn_eager` flag (default off; bit-exact
  path untouched).
- `src/phoenix/train.py`: `--save-every` periodic checkpoints.
- Drivers: `baseline.py`, `necessity.py`, `heads.py`, `counterfactuals.py`,
  `tracing.py`, `cache_patch.py`.
- Tests: `tests/test_fast_equivalence.py` (bit-exact, plus hooks-off and
  eager-path checks), `tests/test_prompts.py`, `tests/test_attn_hooks.py`,
  `tests/test_minimal_pairs.py`.
