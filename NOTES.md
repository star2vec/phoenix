# Notes

Running notes for the position-vs-identity study. A fresh session should be
able to pick up from this file alone. Literature is in `lit/NOTES.md`; the
preliminary paper is `paper/v1.pdf`. Last updated 2026-09-06.

## Status

- Experiment 0 is complete at n=100 on both seeds (see its block).
  Experiments 1 and 2 are piloted on seed 0 (see their blocks); their n=100
  runs wait for the go-ahead. Experiments 3 and 4 are not yet piloted.
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

Pilot outcome: (not run)
What the pilot changed: (not run)
n=100 outcome: (not run)

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
