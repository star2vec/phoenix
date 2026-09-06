# Decisions and amendments

Dated entries, written before the work they describe is run. Outcomes are
added below each entry afterwards, never edited into the prediction.

## Amendment 1 (2026-09-06): shuffled-label retrain

**Status (2026-09-06, evening):** approved. Data regenerated and code
written; training not started (it runs on the laptop). Waiting at gate 1.

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

### Outcome

(not run)
