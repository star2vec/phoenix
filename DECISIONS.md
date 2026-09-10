# Decisions and amendments

Dated entries, written before the work they describe is run. Outcomes are
added below each entry afterwards, never edited into the prediction.

## Amendment 1 (2026-09-06): shuffled-label retrain

**Status (2026-09-08):** run to gate 1 and stopped there; gate 1 failed on
both seeds (see Outcome). No cell was run.

- Data: `src/phoenix/relabel.py --seed 20260906` wrote the three splits to
  `data/relabel/` with `manifest.json` (seed, source and output sha256, and
  the mean depth by label). The relabeled training set's mean depth by label
  runs 2.04 to 2.08 across all 31 tokens (original: 0.00 to 3.23); roots
  fall on all 31 labels. The training file is not committed (30 MB); the
  laptop regenerates it with the same command, and the manifest's sha256
  (`c9225aaba90b3892...`) verifies the bytes. `tests/test_relabel.py` passes.
- Evaluation comparability: the present-labels readout rescored the original
  seed 0 under serialization seed 0 to exactly the stored numbers (396 of
  419; every readout mean and count identical;
  `results/seed0/evaluation_ser0_presentlabels_check.json`).
- Training command, per seed, on the laptop:

      python -u src/phoenix/train.py --seed 0 --run-name seed0/relabel \
        --device cuda --data-dir data/relabel --fixed-early-stages \
        --full-task-patience 15 --save-every 10

  and the same with `--seed 1 --run-name seed1/relabel`. Checkpoints land in
  `ckpts/seed{0,1}/relabel/best.pt`; copy those here. Gate 1 then runs
  `evaluate.py --run-name seed0/relabel --data-dir data/relabel
  --serialization-seed {0,1,2,3}` on both seeds, and nothing else until
  the accuracy numbers are reported.

### Motivation

ProsQA assigns node ids in breadth-first order from the root, so a node's
label is a depth cue: label 2 is at depth 1 in every training graph and the
mean depth rises monotonically to 3.0 at label 25 (`NOTES.md`, experiment 3
block; measured on all 14,785 training graphs). The two name tokens 0 and 1
are always the root and the source of the unreachable component. The model
learned these cues. On 50 test graphs with thoughts free it answers 44
correctly, still 44 when only the two names are swapped, but 13 when the
concept labels are deranged into 2..30 and 22 when the graph's own labels
are permuted (`NOTES.md`, same block; measured 2026-09-06).

This blocked both preregistered label-based cells of experiment 3: labels
renamed with thoughts free (the competence check), and labels renamed with
the thoughts fixed (the identity story's half of the mirror pair). A renamed
prompt fails the model on its own, so nothing about the thought can be read
from what happens when the thought is also fixed. The cell was demoted to
exploration; the position story was refuted by the other half of the pair
(edges reordered, no effect on 100 graphs, both seeds) and by every other
cell it predicted, but the identity story never got its clean positive test.

### Fix

Regenerate the dataset with labels assigned uniformly at random per graph,
retrain the same two-layer model from scratch on it, two seeds, every
hyperparameter unchanged, and rerun the blocked cells on the retrained model.

- **Data.** For every graph in the training, validation and test files, draw
  an independent uniformly random injective assignment of the graph's nodes
  to the 31 node tokens 0..30, and remap edges, root, target, decoy and the
  depth-k node lists with it. Graph structure, candidate pairs, solution
  lengths and file order are unchanged. The two names get random labels like
  every other node, so both cues (name tokens for root and unreachable
  source; id order for depth) disappear. New files under `data/relabel/`,
  one per split, with the generator and its seed recorded next to them; the
  original files are not touched.
- **Training.** `train.py` with the recipe used for the current seeds
  (`--fixed-early-stages --full-task-patience 15 --save-every 10`), seeds 0
  and 1, on the laptop, reading the new files through a `--data-dir` option
  that defaults to the original data. Checkpoints under
  `ckpts/seed{0,1}/relabel/`. Claude does not launch training.
- **Evaluation.** `evaluate.py` on the relabeled test split under
  serialization seeds 0 to 3, as for the original model. The inner-product
  readout counts the node tokens present in the graph. This is the same
  definition the original model was scored under: its readout counted
  tokens 0..n-1, and in every original graph those are exactly the present
  labels (checked on all three splits, 2026-09-06). Accuracy is the argmax
  token against the target and is unaffected either way. So gate 1 compares
  identical metrics across the two models.
- **Cells rerun on the retrained model.** Sample size: n=100 graphs on each
  of the two seeds (recipients = test graphs 0-99 of the relabeled split),
  after a pilot on test graphs 400-409 of seed 0. Cells: labels renamed
  (free, intermediates fixed, all K fixed); edges reordered (three variants)
  and the decoy swap (three variants) as sanity replications. Standing
  controls for this rerun: reserialized baseline and self-transplant (must
  be zero), random donor (breaking reference), same-answer donor (fallback
  reference for flips). No cell in this rerun removes directions, so the
  matched-random-direction control of experiment 3 does not apply and is
  not listed. Results under `results/seed{0,1}/relabel/`; the driver
  checkpoints each graph to a rows file and resumes, as tracing does.
  Existing results directories and the experiment 5 hooks are not touched.
- **Relabeling used inside the cells.** On the retrained model the renamed
  prompt is a uniformly random derangement of every present node over the
  31 tokens (no name convention to respect). The derangement is drawn so
  that the target's old label does not become the new decoy, so the
  signature in prediction 4 below is not confounded with a flip.

### Predictions, written before anything is run

All cell predictions below are for n=100 graphs on each of the two retrained
seeds. The position story is already refuted (experiments 2, 3 and 4, n=100,
both seeds) and predicts nothing here. Lines below are the identity story's,
plus what confirms or refutes the label-depth explanation itself.

1. **Clean accuracy.** The retrained model reaches held-out accuracy within
   the original model's serialization spread, 92.6 to 96.7, and the readout
   ordering holds at every step. If it falls below 88, the label cue was
   load-bearing for accuracy on this model size; report and decide before
   the cells run.
2. **The failure disappears (the label-depth explanation).** On the
   retrained model, renamed prompts with thoughts free score within 3 points
   of clean accuracy on the same graphs, with flips at the reserialized
   level and escape near zero. If renamed prompts still fail, the
   explanation is wrong or incomplete; stop and report, do not run the fixed
   variants.
3. **Labels renamed, intermediates fixed (the identity story's mirror
   half).** Breaks beyond the free twin: thought K-1 names the parent by a
   token that now denotes another node, so the last expansion follows that
   node's edges. Escape dominates (mass on non-candidate node tokens), flips
   at or above the same-answer reference; moves beyond the free twin
   comparable to the random donor at intermediates on the same graphs.
4. **Labels renamed, all K fixed.** The sharp signature: the final thought
   names the target by its old label, and experiment 4 showed the answer
   position reads the final thought's identity through the final latent's
   layer-1 values. So the answer names the token that used to be the
   target's label, now a non-candidate: probability on that token at or
   above 0.5 on most graphs (the cutoff protects "the model named that
   token"), e near 1. Matched reference: under the same-answer donor at all
   K that token gets no mass.
5. **Edges reordered**, any variant: no effect, as on the original model
   (flips and escapes at the reserialized level).
6. **Decoy swap, all K fixed:** no effect, as on the original model.
7. **Controls:** self-transplant exactly zero; reserialized at zero; random
   donor at intermediates breaks (escape or fallback flips); same-answer
   donor at intermediates sets the fallback rate, expected in the range seen
   on the original model (19 to 31 percent), and flips nothing at all K.

What would count against the identity story: prediction 2 holds but 3 and 4
do not (a renamed prompt is solvable, yet fixing thoughts that name the old
labels changes nothing). What would leave the question open: prediction 1 or
2 fails, in which case the retrained model is not a valid substrate for the
test and the amendment stops there.

### Cost

Two retrains on the laptop (about 6 to 10 hours each at the earlier
estimate), one regenerated dataset, then about 30 minutes of CPU per seed
for the evaluation and the cells.

### Earmarked, not part of this amendment

Query-key replication on the retrained model. It answers a different
question (is the query-key result, attention removable but answer unmoved,
an artifact of breadth-first labeling?) and needs every per-head direction
re-derived on the new model, so it is not bundled with the gated cells
above. Sequencing: run this amendment as written; if predictions 2 to 4
hold and the write-up will claim the query-key finding generalizes beyond
the original labeling, add a single-seed query-key replication afterwards
as its own small logged entry here, with its matched-random-direction
control. Worth doing if wall time is cheap; it is insurance for that
headline result.

### Outcome (2026-09-08): gate 1 failed on both seeds; the amendment stops here

Training ran on a RunPod RTX 3090 with the pinned environment (torch 2.5.1,
Python 3.12; equivalence and relabel tests passed there; all three data
hashes matched the manifest). Both seeds completed the four fixed stages and
stopped by early stop at epoch 140 after 40 full-task epochs, about 1.9 hours
each. Files: `results/seed{0,1}/relabel/{metrics.jsonl, train.out,
best.json, evaluation_ser{0..3}.json}`; checkpoints under
`ckpts/seed{0,1}/relabel/` (not committed).

- **Held-out accuracy at chance.** Relabeled test split under serialization
  seeds 0 to 3: seed 0, 216, 209, 206, 215 of 419 (49.2 to 51.6 percent);
  seed 1, 211, 206, 206, 209 (49.2 to 50.4). Best validation accuracy 0.549
  (seed 0, epoch 109) and 0.580 (seed 1, epoch 105). The predicted range was
  92.6 to 96.7; the stop threshold was 88.
- **The readout ordering does not hold.** Seed 0: false at all four steps
  under every serialization seed. Seed 1: true at step 1 only. Optimal and
  frontier means do exceed the others at most steps, but the reachable mean
  sits at or below not-reachable.
- **Failure mode: memorization from the first stage.** In every curriculum
  stage the training loss fell (stage 0: 3.36 to 0.85; full task: to 0.20)
  while the stage validation accuracy stayed at chance: 0.13 to 0.17 for
  "name a child of the root" (about one node in seven), 0.08 to 0.19 in
  stages 1 and 2, 0.44 to 0.54 in stage 3, 0.49 to 0.58 on the full task.
  On 300 training graphs the finished models answer 296 and 292 correctly;
  on 300 unseen test graphs 152 and 144. The original seed 0 answers 299 of
  300 seen and 282 of 300 unseen under the same measurement (2026-09-08,
  CPU). The relabeled models learned no rule at any depth, not even the
  one-hop step, and fit the training graphs instead.

Reading, per the amendment's own rule for prediction 1: the label cue was
load-bearing, and not only for the trained model's competence but for
learning the task at all with this model size, curriculum and budget. Under
ProsQA's breadth-first ids, a two-layer model reaches 94 percent; with the
same graphs under random labels it memorizes. The two label cells cannot be
run on this substrate, and the amendment stops here without running them.

What this does not settle: whether a longer or different schedule would let
the mechanism emerge without the cue (the loss was still falling when each
25-epoch stage ended, and the full-task phase stopped after 40 epochs), or
whether keeping only the name convention (root on token 0 or 1, concepts
random) is enough to make the task learnable. Either is a new amendment with
its own predictions, not a continuation of this one.

## Amendment 2 (2026-09-08): names kept, concept labels random

**Status (2026-09-11):** run to gate 1 and stopped there; gate 1 failed on
both seeds, fork (ii). No cell was run. See Outcome.

### Motivation

Amendment 1 showed that with every label random the same model memorizes
instead of learning (chance on unseen graphs from stage 0 on). ProsQA carries
two label cues: the two name tokens 0 and 1 always mark the root and the
source of the unreachable component, and concept ids are assigned in
breadth-first order so id encodes depth. This amendment removes only the
second. If the model learns, the depth-order cue was dispensable and the
root marker is what made the task learnable at this size; the two label
cells then run with concept-only relabelings, which are on-distribution for
this model. If it memorizes again, the depth order itself is load-bearing and
the label cells cannot be run at this model size and budget.

### Fix

- **Data.** `src/phoenix/relabel.py --seed 20260908 --keep-names --out
  data/relabel_names`: per graph, the two names keep tokens 0 and 1 (the
  root stays on whichever of the two it had), and the concept nodes get an
  independent uniformly random injective assignment into 2..30. Structure,
  candidate pairs, solution lengths and file order unchanged. Manifest with
  seed, source and output sha256 and mean depth by label: concepts 2..30 run
  2.21 to 2.25 (Amendment 1's original range: 0.00 to 3.23);
  roots on token 1 in 9065 graphs and token 0 in 5720. Training
  file not committed (30 MB); regenerated on the pod from the seed; sha256
  `5326364306d5e4b2...`. `tests/test_relabel.py` covers this mode.
- **Training.** Unchanged recipe, seeds 0 and 1, `--run-name
  seed{0,1}/relabel_names --data-dir data/relabel_names`, checkpoints under
  `ckpts/seed{0,1}/relabel_names/`, results under
  `results/seed{0,1}/relabel_names/`. Same pinned environment and tests as
  Amendment 1 on the pod.
- **Everything else** (evaluation under four serialization seeds with the
  present-labels readout, the four cells at n=100 on both seeds, their
  controls, per-graph checkpointing, the derangement constraint) exactly as
  in Amendment 1, with one change: the relabeling used inside the renamed
  cell deranges the concept labels only and leaves the two names untouched,
  matching this model's training distribution.

### Predictions, written before anything is run

Learnability first; neither story predicts it, and the outcome decides
whether the cells run.

1. **Stage 0 is learned.** With the root always on token 0 or 1, the
   one-hop task has a fixed marker: stage-0 validation accuracy rises well
   above chance (above 0.5; Amendment 1 stayed at 0.13 to 0.17). If it
   does not, the root marker is not what made stage 0 learnable and the
   fork below is moot.
2. **Gate 1, the fork.** (i) Held-out accuracy within 92.6 to 96.7 (or at
   least above 88) with the readout ordering holding: the depth-order cue
   was dispensable for learning; proceed to the cells. (ii) Chance, with
   seen-graph accuracy near 100 percent as in Amendment 1: the depth order
   is load-bearing on its own; stop and report. (iii) In between (88 down
   to about 60): partial learning; report, and decide whether a longer
   schedule is worth trying before any cell. My expectation, stated so it
   can be wrong: stage 0 learned, and the deeper stages partly learned,
   landing in (iii) or the low end of (i), because the deeper expansions
   need general content matching that the original data let the model skip.
3. **If (i): the renamed-prompt failure disappears.** Concept-deranged
   prompts with thoughts free score within 3 points of clean accuracy. If
   not, stop; the explanation is incomplete.
4. **If (i): the label cells.** As Amendment 1's predictions 3 and 4:
   intermediates fixed breaks beyond the free twin, escape-dominant;
   all K fixed puts probability at or above 0.5 on the token that was the
   target's old label on most graphs. Position story refuted, predicts
   nothing. Reordered edges and the decoy swap unchanged; controls as
   before.

### Cost

Two retrains on a rented RTX 3090 (about 1.9 hours each, concurrently), then
about 30 minutes of CPU per seed if the gate passes.

### Outcome (2026-09-11): gate 1 failed on both seeds, fork (ii); stopped

Trained on a RunPod RTX 4090 (not the 3090 named in the plan; same pinned
environment, tests passed, all three data hashes matched the manifest). Both
seeds ran the four fixed stages and early-stopped at epoch 139 after 40
full-task epochs, 1.1 hours each concurrently. Files:
`results/seed{0,1}/relabel_names/{metrics.jsonl, train.out, best.json,
evaluation_ser{0..3}.json}`; checkpoints under
`ckpts/seed{0,1}/relabel_names/` (not committed).

- **Prediction 1 held.** Stage-0 validation accuracy at epoch 24: 0.747 and
  0.611 (Amendment 1: 0.125 and 0.148). With the root fixed on a name token
  the one-hop stage is learned.
- **Nothing past one hop is learned.** Stages 1 and 2 peaked mid-stage near
  0.24 and ended at 0.17 to 0.19 while the loss fell to about 1.0; stage 3
  and the full task sat at 0.49 to 0.58 (chance for two candidates). Best
  validation accuracy 0.510 (seed 0, epoch 107) and 0.576 (seed 1, epoch
  101).
- **Gate 1.** Held-out accuracy on the relabeled test split under
  serialization seeds 0 to 3: seed 0, 208, 211, 204, 208 of 419 (48.7 to
  50.4 percent); seed 1, 209, 219, 215, 209 (49.9 to 52.3). Amendment 1
  gave 49.2 to 51.6 and 49.2 to 50.4. Predicted range 92.6 to 96.7; stop
  threshold 88; fork (iii) floor about 60.
- **Readout ordering** holds exactly as far as the learning went and no
  further: seed 0 true at steps 1 and 2, false at 3 and 4; seed 1 true at
  step 1 only. At the failing steps the reachable mean sits below the
  not-reachable mean while frontier and optimal stay largest, the same shape
  as Amendment 1.
- **Memorization check** (2026-09-11, CPU, the check the pod did not run):
  on 300 seen training graphs the models answer 300 and 298 correctly; on
  300 unseen test graphs 151 and 151. Same pattern as Amendment 1 (296/292
  seen, 152/144 unseen); the original seed 0 answers 299 seen and 282 unseen.

Reading, per the fork written above: (ii). The root marker alone makes the
one-hop stage learnable; the depth-order labeling is load-bearing on its own
for every hop after the first. At this model size, curriculum and budget the
model does not learn multi-hop reachability unless concept ids encode depth;
it memorizes the training graphs instead. The two label cells cannot be run
on this substrate, and the amendment stops here. Predictions 3 and 4 were
never reached and are recorded as untested.

Taken with Amendment 1, the substrate finding now has two angles: all labels
random, nothing learned at any depth; names kept, one hop learned and
nothing beyond. Neither retrain touches the results on the original model.
Still open, and not pursued: whether a longer or different schedule would let
the mechanism emerge without the cue.
