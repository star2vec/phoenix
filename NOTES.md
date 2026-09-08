# Notes

Running notes for the position-vs-identity study. A fresh session should be
able to pick up from this file alone. Literature is in `lit/NOTES.md`; the
preliminary paper is `paper/v1.pdf`. Last updated 2026-09-06.

## Amendments

Dated amendments live in `DECISIONS.md`. Amendment 1 (2026-09-06): retrain
on a dataset with labels assigned uniformly at random per graph, so the two
blocked label-based cells of experiment 3 can run. Outcome (2026-09-08):
gate 1 failed on both seeds. Trained on relabeled data with the unchanged
recipe (RunPod RTX 3090), the model memorizes the training graphs (99 and 97
percent on seen graphs) and is at chance on unseen ones (49 to 52 percent
under four serialization seeds; readout ordering fails), with stage
validation accuracy at chance from stage 0 on. The breadth-first label order
of ProsQA is load-bearing for learning the task at this size and budget, not
only for the trained model's competence. The two label cells were not run.
Numbers and files in `DECISIONS.md`, Amendment 1, Outcome.

Comparability of the evaluation across the original and retrained models
(checked 2026-09-06): the accuracy is the argmax token against the target
and does not depend on which tokens the readout counts. The inner-product
readout of `evaluate.py` counted tokens 0..n-1 with n the graph's symbol
count; in every original graph (14,785 train, 257 validation, 419 test)
those are exactly the labels present in the graph. The readout is now
written as "labels present in the graph", which is the identical definition
on the original data and the right one when labels are random over 0..30.
The readouts are therefore identical across the comparison; no rescoring of
the original checkpoints is needed.

## Status

- Experiments 0 to 4 are complete at n=100 on both seeds (2026-09-06).
  Amendment 1 (shuffled-label retrain) ran to its gate 1 and failed it on
  both seeds (2026-09-08): the model does not learn the task without
  ProsQA's label order. Next step is the user's decision (see DECISIONS.md).
  Experiment 5 is not designed yet; the open question left by 3 is where the
  answer is recovered when the thoughts' attention onto the answer path is
  removed (candidates: layer-1 edge reading, and the answer position's own
  content heads).
- Finding worth its own line: ProsQA assigns concept ids in breadth-first
  order and the model uses label id as a depth cue, so consistent
  relabelings are off-distribution for it (experiment 3 block). This is why
  the paper's label-swap donors computed the opposite answer on only 79
  percent of their own prompts.
- Code for experiments 0-4 is written and tested (`tests/`: bit-exact
  equivalence with hooks idle, prompt renderer byte-identical to the vendor
  builder, hook plumbing, and every driver end to end on random weights; the
  smoke test writes to a scratch directory, not to `results/`). No
  experiment has run. `scripts/make_numbers.py` macros get registered once
  result files exist. No trained checkpoint was on this
  machine until 2026-09-06 02:17; `ckpts/seed0/best.pt` and
  `ckpts/seed1/best.pt` now come from the laptop. Every driver checks for its
  checkpoint and stops with a message if it is missing.
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
  actually produced), and the same-answer donor (a training graph with the
  same two candidates, the same correct answer and the same step count,
  transplanted at the intermediate passes and at all K). The same-answer
  donor can only move the answer by breaking the search, so in every cell a
  flip counts as redirection only if the same graph did not flip under it;
  each summary carries the cell's flip rate, the reference rate, and their
  paired difference with an interval (`common.summarize`, key
  "redirection"). No zero vectors as sensitivity anchors.
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
- Experiment 0 keeps the paper's sets for transplants and the final-step
  swap (training graphs 0-99). The per-branch subtraction moved off the
  paper's synthetic set (off-distribution for this model, see the experiment
  0 block) to the natural recipients whose target has exactly one depth-1
  ancestor along shortest paths; that ancestor is the answer branch, another
  child of the root is the sibling. About 70 percent of graphs qualify (210
  of 300 training graphs in `tests/test_prompts.py`); skips are listed.
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

Pilot outcome (seed 0, 2026-09-06; files in `results/seed0/`):
- Accuracy 92.4 (387 of 419; paper 94.5), ordering correct at all four steps
  (`evaluation.json`).
- Probes: 23 nodes, median held-out AUC 0.9972, minimum 0.63 (paper 0.9979
  and 0.63; `probe_basis_report.json`).
- Causal-Jacobian basis: cross-graph concentration by step 0.50, 0.49, 0.71,
  0.93 and cosine to token embeddings 0.41, 0.55, 0.77, 0.78, the same as the
  paper's Table 14 to two decimals (`jlens_basis_report.json`).
- Final-step swap: median -100.0 on 10 of 10; swaps at each intermediate step
  within 0.2 of zero (`baseline_swap_pilot.json`; paper -99.97 and ~0).
- Transplants on training graphs 0-9 (`baseline_transplant_pilot.json`):
  matched donor at intermediate steps median -98.4, interval [-99.7, -25.8],
  7 of 10 flipped, median e 0.001, and 2 graphs escaped (e 0.85, 0.93)
  instead of flipping (paper -91.7, e 0.002); first step only -7.2 (paper
  -2.9); final step only -99.8 (paper -99.95); all steps -99.9 (paper
  -99.97); label swap -0.0 with 1 of 10 flipped (paper -0.0, 2 percent);
  placebo 0.0; interior swap median -0.0 with 2 of 10 flipped (paper -0.08);
  random donor median -99.3 with median e 0.58, 6 of 10 flipped, 5 of 10
  escaped (paper -30.4, e 0.98). Reserialized baseline and self-transplant
  exactly zero.
- Subtraction on `data/eval_graphs.json` graphs 0-9: not interpretable. The
  model answers the decoy or a non-candidate on most of these graphs (119 of
  500 correct over the whole file; the renderer matches the vendor path
  exactly, so this is the model, not the code). Where baseline T is high the
  subtraction is 0.0 in both bases (`baseline_subtraction_pilot.json`).

What the pilot changed:
- The matched donor was mis-specified as a graph with the same correct
  answer; the paper's matched donor is a graph with the same two candidates
  whose correct answer is the recipient's decoy. Fixed in `baseline.py`; the
  same-answer donor is kept as a control (see the next point).
- New observation on this model: a clean flip with low escape is not by
  itself redirection. The same-answer donor, which can only break the search,
  flips 4 of 10 graphs with e near 0 (median -25.6), and the random donor
  flips 4 graphs with e below 0.3. When the search does not find the target,
  this model often answers the other candidate. Consequence for experiment
  3's reading: a cell "breaks" if it moves T or e; whether a flip is a
  redirection or a fallback is decided by comparing, per graph, with the
  same-answer donor and the random donor. The four-outcome reading below is
  amended accordingly.
- The synthetic eval set is off-distribution for this model. Every ProsQA
  root is token 0 or 1, candidates are never 0 or 1, and the name tokens have
  no in-edges; the generator shuffles all labels. Relabeling the root to a
  name token lifts accuracy on the file from 119 to 179 of 500 only, so other
  conventions matter too (in ProsQA the unreachable component hangs off the
  second name token). Decided (user, 2026-09-06): no generator rebuild; the
  subtraction runs on natural test graphs where the target's depth-1
  ancestor is unique. Its pilot is run before its n=100.
- Accuracy gap (92.4 here vs 94.5 in the paper, same checkpoint and split):
  the vendor evaluation draws each graph's edge order from the unseeded
  global RNG, so every run sees different prompts. Evaluated under pinned
  serialization seeds 0-3 (`evaluate.py --serialization-seed`, files
  `evaluation_ser<seed>.json`): 396, 393, 390 and 388
  of 419, that is 94.51, 93.79, 93.08 and 92.60, ordering correct at every
  step under every seed; the unseeded run gave 387 (92.36). The paper's 94.5
  is inside this spread (it equals the seed-0 value), so the gap is
  serialization variance, not a different model. Go-ahead for n=100 on both
  seeds for everything that reproduced (2026-09-06).

Subtraction pilot under the new design (seed 0, test graphs 400-409, 9 with
a unique ancestor; `baseline_subtraction_pilot.json`): median change 0.0 in
the input-embedding and probe bases and under the random-direction control;
one low-confidence graph (base T 81) moved by about 10 in both the real and
the random cell. Same pattern as the paper; nothing changed after this pilot.

n=100 outcome, seed 0 (files in `results/seed0/`):
- Transplants, training graphs 0-99 (`baseline_transplant_n100.json`):
  matched donor at intermediate steps median -95.1 [-98.2, -75.0], 61
  percent flipped [52, 71], median e 0.002 (paper -91.7, e 0.002); first
  step only -1.2 (paper -2.9); final step only -99.9, 97 percent flipped
  (paper -99.95); all steps -100.0 (paper -99.97); label swap -0.0, 1 percent
  flipped (paper -0.0, 2 percent); placebo 0.0; interior swap -0.1, 16
  percent flipped (paper -0.08); random donor -20.4, median e 0.92, 64
  percent escaped (paper -30.4, e 0.98). Reserialized and self-transplant
  exactly zero on all 100.
- Same-answer donor at intermediate steps (the new standing reference):
  median -0.7, 28 percent flipped [19, 37] with e near 0. Against that
  rate the matched donor's flips exceed the reference by +0.33 [+0.20,
  +0.44] (39 percent redirected), the final-step transplant by +0.68 [+0.59,
  +0.78], while the first-step transplant (+0.04 [-0.08, +0.17]) and the
  interior swap (-0.11) do not exceed it: their flips are fallback, not
  redirection.
- Final-step swap (`baseline_swap_n100.json`): median -100.0, 98 percent
  flipped (paper -99.97); swaps at each intermediate step median -0.0.
- Subtraction on natural test graphs 0-99, 68 with a unique ancestor
  (`baseline_subtraction_n100.json`): median change -0.00 [-0.00, -0.00] in
  the input-embedding basis, the probe basis, the random-direction control
  and the sibling subtraction; means -4.7 and -3.6 for the two answer-branch
  bases come from 3-4 low-confidence graphs (11 of 68 have base T below 90).
  The paper's null reproduces.

n=100 outcome, seed 1: evaluation done (unseeded 402 of 419, 95.9; seeds
0-3: 95.5, 96.7, 95.7, 95.5; the paper's 95.7 is inside; ordering correct at
every step). Probes, Jacobian basis, transplant, swap and subtraction:
done on CPU (files in
`results/seed1/`). Probes: 23 nodes, median held-out AUC 0.9919, minimum
0.68. Causal-Jacobian concentration by step 0.50, 0.51, 0.71, 0.92 (seed 0:
0.50, 0.49, 0.71, 0.93). Transplants on training graphs 0-99: matched donor
at intermediate steps -94.9 [-98.9, -63.0], 61 percent flipped, e near 0,
+0.29 [+0.16, +0.43] beyond the same-answer reference (31 percent); first
step only -1.4; final step only -99.9 (94 percent flipped); all steps -99.9;
label swap -0.0 (1 percent flipped); placebo 0.0; interior swap -0.3 (22
percent flipped, below the reference); random donor -33.0 with e 0.77.
Final-step swap -100.0 (98 percent flipped); intermediate swaps -0.0.
Subtraction on 68 natural test graphs: medians -0.01, -0.00, +0.00, +0.00
for the input-embedding basis, probe basis, random control and sibling.
Reserialized and self-transplant exactly zero on all 100.

Experiment 0 is complete on both seeds. Every paper measurement reproduces
on both, with one addition: the same-answer donor's fallback rate (28 and 31
percent) is the reference against which redirection is counted.

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

Pre-pilot note (2026-09-06, before the pilot ran): the same-answer donor is
now a cell here too and is the reference for fallback flips (28 percent on
training graphs in experiment 0). Substitutes that carry no candidate
(average thought, noise, zero) are expected to break mostly by escape or by
fallback flips at about that rate; a cell whose flips exceed the reference
would be surprising and would need the probability split to explain it.

Pilot outcome (seed 0, test graphs 400-409, `results/seed0/necessity_pilot.json`):
every uninformative substitute breaks, as expected. At all K passes: average
thought median -27.2 [-69.4, -5.4] with e 0.94 and 10 of 10 escaped (the
mass, 0.91, lands on other node tokens: the model names nodes from the
blurred thought); random donor -75.9 with e 1.00, 10 of 10 escaped; noise
-5.3 but bimodal (4 flipped with e near 0, 3 escaped); zero -22.7 (3
flipped, 4 escaped; the paper's -42.9 was a median over 100 training
graphs). Removing the latent tokens gives -87.4 with 7 of 10 flipped and e
near 0: without thoughts the model answers the decoy, not chance. Pad tokens
in place of the latents: -15.8, 4 flipped, 3 escaped. Same-answer donor:
at intermediate passes -1.1 with 2 of 10 flipped (the fallback reference);
at all K exactly no flips (the donor's final thought carries the shared
answer). Self-transplant and reserialized exactly zero.
What the pilot changed: nothing in the design. Two observations to carry:
substitutes that contain node content (average, random donor) break by
escape onto other nodes, substitutes without it (noise, zero) break by
fallback flips; and "thought removed" is a systematic decoy answer.
n=100 outcome (both seeds; `results/seed0/necessity_n100.json`,
`results/seed1/necessity_n100.json`): every substitute breaks on both seeds.
At all K passes, medians on seed 0 / seed 1: average thought -14.0 / -22.1,
noise -13.6 / -11.7, zero -25.4 / -19.1, random donor -16.4 / -37.3; flip
rates 34-43 percent, escape 29-90 percent. The same-answer donor at all K
flips nothing on either seed (its final thought carries the shared answer),
so all of those flips exceed the all-K reference (+0.35 to +0.44). At the
intermediate passes the same-answer donor flips 19 / 26 percent and the
substitutes exceed it by +0.11 to +0.27. Removing the latent tokens: -15.7 /
-23.6 with 40 / 42 percent flipped; the pilot's "decoy on 7 of 10" was the
small sample (decoy mass 0.22 / 0.27 at n=100). Pad tokens in place of the
latents: -13.2 / -9.2. Where the mass goes replicates: node-bearing
substitutes put 0.58-0.79 of the probability on other node tokens on both
seeds; noise puts 0.30-0.43 there. Self-transplant and reserialized exactly
zero on all 200 runs. Conclusion: the thought is necessary and informative
on both seeds, as the paper found; no story-separating content here.

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

Pre-pilot note (2026-09-06, before the pilot ran): scores are computed on
the per-slot attention mass (sum over an edge's tokens). A head is read as
an edge-reading head at a query class only if its mean slot mass there is at
least 0.2 (the cutoff protects "this head attends to edges at all"); heads
below it are reported but not classified.

Pilot outcome (seed 0, test graphs 400-409, `results/seed0/heads_pilot.json`):
- Intermediate latents (thoughts 1..K-1 as queries): all 8 layer-2 heads put
  0.89-0.97 of their attention on edge slots and follow the edges after the
  reorder: content scores 0.88-0.93, position scores -0.02 to +0.09. The
  identity story's prediction for this cell; the position story's is
  rejected on all 10 graphs for every layer-2 head. The same holds at the
  root query, the last latent and the answer position (content 0.62-1.00).
- Layer 1 at the intermediate latents: the four edge-reading heads (mass
  0.76-0.93) are mixed, content 0.49-0.67 and position 0.27-0.40; the heads
  with high position scores there carry almost no slot mass (0.01-0.12) and
  are not classified.
- The method does find positional heads: at the answer position layer-1 head
  2 has slot mass 0.99 with position score 0.99 and content 0.07, and heads 5
  and 6 are mostly positional (0.78 and 0.64). Where they look is recorded
  from n=100 on (first-slot and last-slot mass).
- Edge-token queries keep only 0.06-0.10 of their attention inside their own
  slot in both layers, so the layer-1 copy of Zhu et al.'s construction is
  not at the edge's target token in this model.
What the pilot changed: the heads driver now also measures separator-token
queries (the "|" token can see both endpoints of its edge) and reports each
head's mass on the first and last slot, so positional heads can be located.
Scores and cutoffs unchanged.
n=100 outcome (both seeds; `results/seed0/heads_n100.json`,
`results/seed1/heads_n100.json`):
- Intermediate latents, layer 2: all 8 heads on both seeds carry 0.89-0.98 of
  their attention on edge slots and follow the edges. Content scores 0.85-0.90
  (seed 0; intervals within [0.82, 0.92]) and 0.91-0.94 (seed 1; within
  [0.88, 0.96]); position scores between -0.02 and +0.03 on both. The identity
  story's prediction holds for every layer-2 head on both seeds; the position
  story's is rejected. The same holds at the last latent and, for the
  edge-reading layer-2 heads, at the answer position.
- Layer 1 at the intermediate latents: edge-reading heads mixed (content
  0.47-0.71, position 0.23-0.42); the heads with position scores above 0.6
  carry 0.02-0.36 slot mass.
- Zhu et al.'s layer-1 copy is at the separator token: from the "|" token,
  seven of eight layer-1 heads on seed 0 (0.90-0.94) and all eight on seed 1
  (0.95-0.96) put their attention inside the edge's own slot; from the
  edge's target token only 0.05-0.11. Layer 2 does not copy (0.03-0.08).
- A purely positional layer-1 head sits at the answer position on both seeds
  (seed 0 head 2: mass 0.99, position 0.99, content 0.00; seed 1 head 0:
  mass 0.97, position 0.87, content 0.05); it attends neither to the first
  nor the last slot in particular (0.06 and 0.00 of its mass), so it spreads
  over slots by position. What it does is an open item for experiment 4.
Conclusion: the layer-2 query built from a mid-search thought matches edges
by content, not by slot, on both seeds.

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
- Non-candidate swap (as the decoy swap, but the swapped-in edge leads to an
  unreachable node that is not a candidate; the graph is unchanged), all K
  fixed. The cell reports p_watch, the probability the answer puts on that
  node. Position story: the final step dereferences the target's old slot
  and names that node, so p_watch is high (cutoff 0.5 protects "the model
  named the swapped-in node"); T may fall by fallback, but only p_watch is
  the signature, because a broken search cannot put mass on a node it has no
  way to select (the same-answer and random donors give the reference
  level of p_watch). Identity story: no effect, p_watch near zero.
  Separating for the final thought, and immune to the fallback ambiguity.
- Non-candidate swap, intermediates fixed. Both stories: no effect.
  Consistency.
- Last-hop rewrite (the cut edge into the target now points to the decoy,
  same slot; BFS confirms the target is unreachable and the decoy is at depth
  K), all K fixed. Position story: follows the edit (T falls to 0). Identity
  story: keeps the old answer. Separating for the final thought.
- Single-edge rewrite at depth d, thoughts free. Both stories: follows the
  edit. Competence check (Jin et al.).
- Single-edge rewrite at depth d, intermediates fixed, stratified by d.
  Both stories: old answer if d < K-1, new answer if d = K-1. Exploration:
  how much of the search is committed by step K-1.
- Query-key subtraction (added 2026-09-06 at the user's request, before the
  pilot). The answer's edge is the edge (p, target) with p at depth K-1; it
  is read by the layer-2 attention of the query built from thought K-1 (the
  last intermediate thought, pass K-2). For each layer-2 head, the direction
  in thought space that the head's query matrix maps onto that edge's key is
  the query matrix applied to the key (the key averaged over the edge's
  tokens with the head's own attention as weights), scaled by the layer-norm
  gain and centered; this is the first-order direction, ignoring the norm
  rescaling of the layer norm and layer 1's indirect response. The cell
  subtracts that direction from thought K-1 (norm preserved, as the paper's
  SUBTRACT), per head and for all eight heads at once (their span removed).
  Controls: the same coefficients removed along matched random directions
  (per head and for the span), and the same construction for a non-answer
  edge (the frontier edge with the most attention that does not lead to the
  target, else the most attended other edge). Recorded per run: each head's
  attention from that query onto the answer's edge and the control edge,
  before and after; the removed coefficient as a fraction of the thought
  norm; the cosines between the eight directions and between each direction
  and the input-embedding directions of p and of the target.
  Identity story, read through attention geometry: attention to the answer's
  edge drops and the answer flips above the same-answer reference rate
  (redirection above zero), with the drop largest for the all-heads removal.
  Position story: attention drops too if the pointer is linear in the
  thought, since the direction is built from whatever the key carries; no
  prediction on the answer. Both controls: no drop, no flips beyond the
  reference. If nothing moves (attention unchanged, flips at the reference
  rate), the thought's causal content is not in this linear query-key
  geometry, which is where the paper's per-branch nulls would then be
  located, and the cell says so.

Pre-pilot note (2026-09-06): the predictions above and in experiment 4 stand
as written; the same-answer donor is a standing control, and the
non-candidate swap and query-key cells are included in the first pilot.

Added after the first two pilot runs of experiment 3 (2026-09-06, before the
third run; see "What the pilot changed" below for why): the "labels renamed"
cell cannot be run on this substrate, because the model relies on ProsQA's
id-depth convention and fails a fully relabeled prompt with thoughts free.
Two replacement cells that move only two labels, each run three ways and
read against its own free twin (paired per graph: flips and moves of the
fixed variant minus the free variant, `beyond_free_twin`):
- Parent swap, one level up: the target's depth-(K-1) parent trades labels
  with a depth K-2 node. Identity story, intermediates fixed: breaks beyond
  the free twin (thought K-1 names the parent by a token that now denotes a
  node one level up, whose edges do not reach the target). Position story:
  nothing beyond the free twin. All K fixed: both stories nothing beyond the
  free twin (thought K names the target, whose label is untouched).
- Parent swap, same depth: the parent trades labels with another depth K-1
  node. Identity story: nothing beyond the free twin (the depth K-1 frontier
  is the same set of labels, so thought K-1 still names it). Position story:
  nothing. Control for the cell above.
The free twins themselves are expected below baseline (a two-label swap cost
6 to 14 points of accuracy on 50 test graphs in the check that motivated
this), which is why the reading is paired against them.

Pilot outcome, experiment 3 (seed 0, test graphs 400-409; three runs, all
kept: `results/seed0/counterfactuals_pilot_v1_root_relabeled.json`,
`counterfactuals_pilot_v2_full_rename.json`, `counterfactuals_pilot.json`):
- Controls: reserialized and self-transplant exactly zero; random donor at
  intermediate passes -25.1 (4 flipped, 7 escaped), at all K -75.9 with e
  1.00; same-answer donor 2 of 10 flipped at intermediates, none at all K.
- Edges reordered: no effect in any variant (free, intermediates, all K:
  medians -0.0, no flips, no escapes, on all 10 graphs); the unreachable-only
  reorder likewise. The position story's primary prediction fails.
- Decoy swap and non-candidate swap with all K fixed: no effect on any
  graph; p_watch 0.00 on all 10. The position story's prediction for the
  final thought fails. Intermediates fixed: no effect, as both stories said.
- Last-hop rewrite, all K fixed: the old answer is kept on 7 of 7 (identity
  story's prediction; the position story's fails). Thoughts free, the model
  follows the edit on only 3 of 7 (median -14.8); intermediates fixed, 4 of
  7 follow. Rewrites at depths 1, 2, 3 with thoughts free are followed on 2
  of 4, 3 of 5, 1 of 3; with intermediates fixed the old answer stays on all,
  as both stories said. Following an edit is itself unreliable in this model.
- Labels renamed: not runnable as a thought test. Run 1 gave the root a
  concept label and its free twin failed (e 1.00 on 6 of 10). Run 2 swapped
  the two name tokens and deranged the concepts; its free twin still failed,
  6 of 10 answering the decoy with e near 0. Cause, checked in the data:
  ProsQA assigns concept ids in breadth-first order, so label 2 is always at
  depth 1 and the mean depth rises monotonically to 3.0 at label 25; the
  model learned that cue. Accuracy with thoughts free on 50 test graphs:
  original 44; concepts deranged into 2..30, 13; the graph's own labels
  permuted, 22; only the two names swapped, 44; two-label swaps 23-39 of
  31-50 (74-82 percent); the paper's target-decoy swap 40 (the paper: 79
  percent). The renamed cell stays in the driver as exploration only.
- Replacement (run 3): parent swap one level up, free twin breaks on 4 of 10
  (competence loss), intermediates fixed adds nothing beyond it (flips beyond
  the free twin -0.10 [-0.40, +0.20]; moves the same), all K nothing beyond.
  Same-depth parent swap: nothing in any variant, as both stories said. The
  identity story's prediction for the one-level-up swap is not confirmed at
  n=10; power is low because the free twin itself breaks on 4 graphs.
- Query-key subtraction (`qk_subtract/*` cells): removing the span of the
  eight directions from thought K-1 cuts the answer edge's attention, summed
  over the eight layer-2 heads, from 2.10 to 0.38 (drop 1.71 [1.10, 2.36])
  and leaves the control edge at 1.40 to 1.42; the answer flips on 4 of 10
  with e near 0 (median -8.7; flips beyond the same-answer reference +0.20
  [0.00, +0.50]). Matched random directions: attention 2.10 to 2.08, no
  flips. Non-answer edge's directions: that edge's attention 1.40 to 0.17,
  the answer edge's 2.10 to 1.90 (interval spans zero), no flips. Per head:
  heads 3 and 4 carry the most (drops 0.89 and 0.99; coefficients 0.25 and
  0.28 of the thought norm); single-head removals rarely flip (head 4: 2 of
  10). The eight directions are nearly orthogonal to each other (mean
  absolute cosine 0.20) and to the input embeddings of the edge's source
  (about 0.05) and of the target (about 0.1). This is the first linear edit
  of a thought with a causal effect; the paper's per-branch nulls were in
  bases these directions do not lie in.
What the pilot changed: the renamed cell was fixed (name convention) and
then demoted to exploration; the two parent-swap cells were added with
predictions before run 3; every changed prompt is now also read against its
free twin, paired per graph (`beyond_free_twin`). Nothing else.
Reading of the mirror pair: "reordered does not break" is established at
pilot level with no exception in 10 graphs; the renamed half cannot be run on
this substrate. The identity story's positive support therefore rests on
experiment 2 (n=100, both seeds), the query-key cell, and experiment 4.

Before the n=100 run (2026-09-06, user's decisions): the parent-swap cells
are dropped (their free twin breaks too often to read anything from them).
Two additions to the query-key cell, predictions first:
- Per head, at n=100. Identity story: each single-head removal drops that
  head's own attention on the answer edge by an amount that tracks the
  head's coefficient in the thought; the answer flips above the same-answer
  reference only for the heads that carry most of the match (heads 3 and 4
  in the pilot), and the all-heads removal flips more graphs than any single
  head. Position story: the same drop pattern; no prediction on the answer.
  Matched random per head: no drop, flips at the reference rate.
- At every intermediate step (the paper's path elimination, asked in the
  right coordinates). At each pass k = 0..K-2 the answer-path edge read at
  that step is the most attended edge from a depth-(k+1) node on a shortest
  path to the target to a depth-(k+2) node on one (the last step's edge is
  the parent edge). From thought k+1 the span of the eight directions that
  its layer-2 heads map onto that edge's key is removed; all steps in one
  run, directions taken from the unedited run. Identity story: attention
  onto the answer-path edge drops at every step and the answer flips at
  least as often as in the last-step-only cell; the paper's identity-basis
  version of this removal did nothing, and this cell asks whether that null
  was the basis. Position story: attention drops if the pointer is linear in
  the thought; no prediction on the answer. Controls: matched random
  directions at every step (no drop, reference-rate flips); the most
  attended frontier edge off the answer path at every step (its own
  attention drops, the answer path's does not, no flips beyond the
  reference; skipped on graphs where some step has no such edge).
  A per-head every-step version is run as well, read like the per-head
  last-step cells.

n=100 outcome, experiment 3, seed 0 (`results/seed0/counterfactuals_n100.json`;
seed 1 pending):
- Controls: reserialized and self-transplant exactly zero on all 100. Random
  donor at intermediates -8.4 (30 percent flipped, 66 escaped), at all K
  -16.4 with e 1.00. Same-answer donor 19 percent flipped at intermediates,
  none at all K.
- Edges reordered: no effect in any variant (flips 0 to 3 percent, no
  escapes; moves beyond the free twin -0.03 [-0.07, 0.00]). The
  unreachable-only reorder likewise. The position story's primary
  prediction fails at n=100.
- Decoy swap and non-candidate swap: no effect in any variant; p_watch
  0.000 on all 100 graphs. The position story's final-thought prediction
  fails at n=100.
- Last-hop rewrite (68 constructible): all K fixed keeps the old answer on
  68 of 68; thoughts free follow the edit on 29 percent, intermediates fixed
  on 34 percent (+0.04 [0.00, +0.10] beyond the free twin). Depth-1 and
  depth-2 rewrites are followed on 54 and 52 percent with thoughts free and
  on 0 percent with intermediates fixed, as both stories said.
- Labels renamed (exploration; the free twin fails by the label cue, 49
  percent flipped, 49 escaped). Read against its free twin, fixing the
  thoughts adds escape, not flips: moves beyond the free twin +0.17 [+0.08,
  +0.26] at intermediates and +0.32 [+0.23, +0.41] at all K, escape rising
  from 49 to 74 and 95 percent. A thought that names stale labels makes the
  model name other nodes; the position story predicted nothing beyond the
  free twin. Suggestive for the identity story, but not a clean cell.
- Query-key subtraction, last step, all heads: the answer edge's attention
  (summed over the eight layer-2 heads) falls from 1.79 to 0.31 (drop 1.47
  [1.30, 1.67]); the control edge is unchanged (1.11 to 1.13); matched
  random directions leave both (1.79 to 1.77); the non-answer edge's
  directions cut that edge to 0.13 and the answer edge to 1.68. The answer
  does not move beyond the fallback rate: median -0.2, 17 percent flipped,
  -0.02 [-0.11, +0.07] beyond the same-answer reference. The pilot's 4 of 10
  was a small sample. Per head: the head-drops track the coefficients (head
  4: 0.49, head 3: 0.39, head 0: 0.18, head 7: 0.22; heads 1, 2, 5 below
  0.05) and no single head flips beyond the reference.
- Query-key subtraction at every intermediate step, all heads: the
  answer-path edge's attention is removed at every step (mean per-step drop
  1.60 of about 1.8; at the last step 1.79 to 0.13); the answer flips on 19
  percent, 0.00 [-0.08, +0.08] beyond the reference. Random directions at
  every step: no drop, no flips. The off-path control cuts its own edge to
  0.10 and, by overlap of directions, the path edge to 1.14, with 8 percent
  flips (below the reference). Per head: path drops 0.17 to 1.19, flips at
  most 12 percent, none beyond the reference.
  Reading: the linear geometry that drives the thought's layer-2 attention
  onto the answer path is real and removable, and removing it does not
  change the answer beyond fallback. The paper's "rewrite" null replicates
  in the right coordinates. The answer must be recovered elsewhere: layer 1
  also reads edges (experiment 2), and the answer position has its own
  content heads. Where it is recovered is the next question, not answered
  here.
- Splits: none of the story cells split (all at 0 or near 0); the query-key
  cells' flipped graphs are not distinguished by depth, branches, or the
  answer edge's slot (split tables in the file).

n=100 outcome, experiment 4b, seed 0 (`results/seed0/cache_patch_n100.json`;
seed 1 pending): consistent patches from the reordered graph (keys and
values together in layer 1, layer 2 or both; every latent-position slice) do
nothing (flips 0 to 3 percent). Inconsistent ones break: layer-2 keys only
-21.2, 40 percent flipped, +0.21 [+0.10, +0.32] beyond the reference;
layer-2 values only -28.1, 40 percent, +0.21 [+0.09, +0.33]; values in both
layers 6 percent (consistent again through the separator copy). The position
story's keys-only prediction (no effect) fails; the identity story's (breaks)
holds, as does its consistency reading. Random-graph cache: edge slots break
in every variant (34 to 47 percent flipped, +0.15 to +0.29 beyond the
reference); layer-1 values at all latents -35.5 with e 1.00 (43 percent
flipped, 81 escaped) while layer-2 values at the latents and every
intermediate-latent slice do nothing. Only the final latent's layer-1 values
carry the answer to the answer position. Self patch exactly zero.

n=100 outcome, experiment 3, seed 1 (`results/seed1/counterfactuals_n100.json`):
replicates seed 0 on every cell. Reordered, unreachable-only reorder, decoy
swap and non-candidate swap: no effect in any variant (p_watch 0.000 on all
100). Last-hop rewrite with all K fixed keeps the old answer on 68 of 68;
with thoughts free this model follows the edit on 53 percent (seed 0: 29).
Renamed (exploration): fixing the thoughts again adds escape beyond the free
twin, +0.19 [+0.11, +0.27] moves at intermediates and +0.28 [+0.20, +0.37]
at all K, flips unchanged. Query-key subtraction, last step: the answer
edge's attention falls 2.54 to 0.37 (drop 2.17 [1.95, 2.39]), controls flat;
the answer -3.2 median, 15 percent flipped, -0.10 [-0.19, 0.00] beyond the
same-answer reference (26 percent). Every step: attention 2.54 to 0.13, mean
per-step path drop 1.99; answer -3.9, 22 percent flipped, -0.03 [-0.15,
+0.09] beyond the reference. No single head flips beyond the reference on
either seed. Same-answer donor at all K: no flips. Reserialized and
self-transplant exactly zero.

n=100 outcome, experiment 4b, seed 1 (`results/seed1/cache_patch_n100.json`):
replicates seed 0. Consistent patches from the reordered graph: no effect
(edges, any layer, keys and values together: 1 to 3 percent flipped; latent
slices 0 to 1). Layer-2 keys only -34.0, 43 percent flipped, +0.18 [+0.05,
+0.32] beyond the reference; layer-2 values only -19.4, 35 percent, +0.10
[-0.03, +0.22]. Random-graph cache at the edge slots breaks (40 to 44
percent flipped); layer-1 values at all latents -61.8 with e 1.00 (52
percent flipped, 81 escaped); layer-2 values at the latents and every
intermediate-latent slice do nothing. Self patch exactly zero.

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
Amendment after the experiment-0 pilot: on this model a broken search often
ends in a clean flip to the other candidate (same-answer donor: 4 of 10 flip
with e near 0). So "breaks" means T or e moved; a flip counts as
redirection only if the same graph does not also flip under the same-answer
donor. The escape-level comparison in reading 3 uses the random donor's e
and the same-answer donor's outcome as the two references, per graph.

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

Pilot outcome, 4a (seed 0, test graphs 400-409, token granularity;
`results/seed0/tracing_pilot.json`; candidate swap on 10, last-hop rewrite
on 7). Mean recovery by position class and level:
- Final latent, level 0 (the final recycled thought plus its position
  embedding): 0.98 and 0.99. Its levels 1 and 2: 0.00. So the final thought
  acts through the layer-1 keys and values written at the final latent
  position, which the answer position reads.
- Intermediate latents: levels 0 and 1 about 0 (slightly negative under the
  rewrite, -0.13); level 2 (the output that becomes the next thought) 0.38
  and 0.30, which is latent K-1's output, that is thought K, doing the work.
- Changed edge slots: the target token at level 0 recovers 0.26 (candidate
  swap; several slots change) and 1.00 (rewrite; one slot changes); the
  separator at level 1 recovers 0.24 and 0.98; the source token 0.00.
  Candidates, root, and unchanged slots: 0.00. The edge's identity enters at
  its target token and is copied to the separator by layer 1, where layer 2
  reads it; the same separator finding as experiment 2.
No story-separating prediction was made for this map; it is consistent with
the identity story's mechanism and it names the carriers for 4b.

Pilot outcome, 4b (`results/seed0/cache_patch_pilot.json`):
- From the reordered donor: keys and values together, in layer 1, layer 2 or
  both, at the edge slots: no effect (no flips). Every patch of the latent
  positions (intermediate or all, keys, values or both): no effect. Under
  the position story the layer-2 cache of the reordered graph should have
  sent the pointers to other edges; it did not.
- Inconsistent patches break: layer-2 keys only, median -4.0, 3 of 10
  flipped (+0.10 beyond the reference, interval spanning zero); layer-2
  values only, -62.6, 6 of 10 flipped. Values in both layers: no effect,
  because the separator's layer-1 copy then reads the donor's edge and the
  layer-2 keys become consistent with the values again. Reading: a
  consistent reordered cache is read correctly by content; only a cache
  whose keys and values disagree breaks the search. The keys-only prediction
  (identity: breaks; position: nothing) came out partial and cannot be
  called at n=10.
- Random-graph cache (Ding's necessity): edge slots, layer-1 values or keys
  and values, -95 to -97 with 6 of 10 flipped; layer-2 keys, 4 of 10; all
  latents, layer-1 values -50 (5 flipped, 8 escaped) while layer-2 values at
  the latents and every intermediate-latent slice do nothing. Only the final
  latent's layer-1 values carry the answer to the answer position, matching
  4a. Self patch exactly zero; same-answer donor 2 of 10.
What the pilot changed: nothing in the design of 4a or 4b. After a memory
kill lost a near-complete run, tracing now saves each graph as it finishes
and resumes (`tracing_<mode>_rows.jsonl`).
n=100 outcome, 4a, seed 0 (`results/seed0/tracing_n100.json`; candidate swap
on 100 graphs, last-hop rewrite on 68). Mean recovery: final latent at level
0 (the final thought plus its position embedding) 0.98 [0.96, 0.99] and 0.95;
its levels 1 and 2, 0.00. Intermediate latents: levels 0 and 1, 0.08 and
-0.04; level 2 (the output that becomes the next thought), 0.47 and 0.34.
Changed edge slots: target token at level 0, 0.34 (several slots change
under the candidate swap) and 1.00 (one slot under the rewrite); separator at
level 1, 0.31 and 0.96; source token 0.00. Candidates, root, unchanged
slots: 0.00 (root at level 2 under the swap 0.03). The pilot's map holds at
n=100. Seed 1 (`results/seed1/tracing_n100.json`) replicates: final latent
at level 0, 1.02 [0.96, 1.09] and 1.00; its levels 1 and 2, 0.00;
intermediate latents at level 2, 0.44 and 0.35, at levels 0 and 1, 0.03 and
-0.04; changed slot's target token at level 0, 0.30 and 1.00; its separator
at level 1, 0.27 and 0.95; source token, candidates, root and unchanged slots
0.00. Both seeds: the final thought does its work through the layer-1 keys
and values written at the final latent position; an edge's identity enters
at its target token and is copied to the separator by layer 1, where layer 2
reads it; the intermediate latents matter only through the thought they
emit.
n=100 outcome, 4b: see the experiment 3 block's neighbour above and the
seed-0 block ("n=100 outcome, experiment 4b, seed 0").

## Experiment 5: later, only if 1-4 favor one story

Mask the answer position's (and separately every latent's) attention to the
two candidate tokens (`attn_hooks.AttnHooks.add_mask`); checkpoints along
training (saved by `train.py --save-every`); three- and four-layer models.
Designs and predictions to be written after 1-4 report.

## Next run

Device on this Mac: `cpu` (measured 2026-09-06: 25 ms per batch-one forward
on CPU vs 178 ms on MPS, and MPS memory grew 750 MB in 60 forwards; the
seed-1 chain on MPS was killed for memory at 14,000 of 14,785 probe graphs).
Seed 0's experiment-0 files were produced on MPS before this was known; the
two paths differ at rounding level only (`tests/test_fast_equivalence.py`).

```
.venv/bin/python src/phoenix/evaluate.py --run-name seed0 --device cpu --serialization-seed 0
.venv/bin/python src/phoenix/fit_probes.py --run-name seed0 --device cpu
.venv/bin/python src/phoenix/fit_jlens.py --run-name seed0 --device cpu
.venv/bin/python src/phoenix/baseline.py --run-name seed0 --device cpu --mode pilot
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
