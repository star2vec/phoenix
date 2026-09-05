# Literature notes

Rebuilt 2026-09-06 from the papers themselves. Every entry below was reopened
at its current arXiv version (full text via the arXiv HTML or PDF unless the
entry says "abstract only"). The old July 2026 file was written for a
different question (can branches be edited out of a thought) and has been
replaced, not patched.

## The short answer to the two questions we care about

**Has anyone published a causal test of what a continuous thought encodes?**
Partly. Several papers now intervene causally on continuous thoughts, but all
of them treat a thought (or the whole cache written during the thoughts) as a
single object: they zero it, overwrite it, slerp it, transplant it, or swap
the cache. None decomposes a thought into components and tests the components
separately, and none does this on a model trained from scratch on graph
reachability. The closest are Aswal et al. (2606.12689), who run causal
tracing on fine-tuned GPT-2 COCONUT on ProsQA and find that the indirect
effect sits at prompt positions with individual thought positions near zero,
and Ding et al. (2608.27265, SCIT, EMNLP 2026), who show for CODI that the
counterfactual answer travels in the value cache written during the latent
steps and that at 8B the prompt-prefix cache is already sufficient before any
latent step runs. Jin et al. (2607.06648) test verified single-edge rewrites
of ProsQA graphs and show COCONUT stops following them as training proceeds.

**Has anyone published a positional account of latent reasoning?** No. Nobody
has proposed that a continuous thought encodes prompt positions (which edges
are consumed, which to attend to next) with node identity resolved from the
prompt at the last step. The two results above are consistent with prompt
positions doing the work but neither frames the thought as a pointer, and
both are on fine-tuned pretrained models where the thoughts are weakly used.
The ingredients for such an account are well established in a different
literature: pretrained models bind entities with abstract or positional IDs
and dereference them by attention at the answer position (Feng and
Steinhardt 2310.17191; Gur-Arieh, Geva and Geiger 2510.06182; Bogdan and
Lindsey 2604.21139; Oh and Demberg 2606.08644), and "where" heads and "what"
heads are separable and learned in different orders (Urrutia et al.
2605.31558). Note also that Zhu et al.'s own two-layer construction is
positional in layer 1 (each edge token copies its endpoints by relative
position) and identity-matching in layer 2 (the thought is compared with edge
source embeddings). A positional thought would contradict what their
construction says the thought is, while using the same layer-1 machinery.

**Scoop risk.** Moderate and rising. SCIT was accepted at EMNLP 2026 and its
authors name loop transformers and per-iteration carrier mapping as next
steps. Aswal et al. explicitly propose model diffing across curriculum stages.
The SPAR spring 2026 project mentored by Uzay Macar (Anthropic Fellows) lists
"intervention techniques for continuous thoughts" and "extracting multiple
reasoning paths from continuous vectors" as goals for Coconut; the project
page still lists no outputs as of 2026-09-06.

---

# Part 1. Continuous thought on graph reachability: the substrate

## Hao et al., Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)
arXiv 2412.06769, v4 dated 2026-08-23 (v1 2024-12-09), COLM 2025. Abstract page.

Claim: feed the last hidden state back as the next input embedding instead of
a token; on ProsQA this beats textual chain of thought, and the abstract
attributes the gain to continuous thoughts encoding several next steps at
once, a breadth-first search rather than early commitment.

Verified: abstract and version list only. A v4 appeared in August 2026; the
abstract claims are unchanged. Not re-read in full.

Relevance: the method we train. Its BFS interpretation is the claim every
later paper is testing.

## Zhu et al., Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought
arXiv 2505.12514, v3 dated 2025-11-01, NeurIPS 2025. Unchanged since the July
note. Read in full (HTML and PDF, appendix tables extracted from the PDF).

Claim: a two-layer transformer with D continuous-thought steps solves
diameter-D directed reachability; the thought after step c is, in the
construction, the normalized sum of embeddings of all nodes within c hops.
A two-layer model trained from scratch (d_model 768, 8 heads) on a 3-4 hop
ProsQA subset reaches near-perfect accuracy and its thoughts have inner
products with node embeddings that trace a BFS wave.

Verified this session:
- Setup: GPT-2 style, 2 layers, 768 wide, 8 heads, from scratch, AdamW, LR
  1e-4 constant, 25 epochs per stage, 300 epochs total, previous-stage data
  mixed in with probability 0.1 (section 5.1). Two candidates, exactly one
  reachable (section 2). Train 14,785 / val 257 / test 419; mean 22.8 nodes,
  36.5 edges, solution length 3.5 (Table 4). About 24 hours on two A100-80GB
  per run (Appendix C.3).
- Construction: layer 1 has five "attention chooser" heads that use
  positional information (sinusoidal, RoPE variant in B.6) so that each edge
  token gathers its source and target embeddings; layer 2 has one head whose
  query is the thought and whose key at an edge token is the source-node
  embedding, so the thought attends to edges whose source is in the current
  reachable set. The MLP thresholds noise. Thought = (1/sqrt|V_c|) sum of u_v
  over reachable v (Eq. 1). LayerNorm is defined as x/||x||, so thoughts are
  unit norm.
- Empirics: Table 1 is layer-2 attention to edge groups, not inner products
  (Not reachable 0.04 at step 1 rising to 0.12 at step 4; Optimal 2.54, 1.72,
  1.67, 2.23). Table 5 is the inner product between thought and node
  embedding over three seeds: Not reachable -0.37 to 0.02; Reachable 0.53 to
  3.71; Frontier 1.95 to 5.38; Optimal 4.67 to 9.58. The July note's ranges
  were correct. Figure 4: Coconut about 1.0, 2-layer CoT about 0.75, 12-layer
  CoT about 0.83, no-CoT about 0.5. No exact accuracy is printed.
- Figure 5: trained layer-1 edge tokens put nearly all attention on their own
  source and target nodes, i.e. the copying mechanism is instantiated.
- COCONUT-BFS variant (uniform random frontier node as supervision) gives the
  same accuracy and the same pattern.
- No probing, ablation, or causal intervention anywhere. Future work
  paragraph lists lower bounds, training dynamics, and general settings.

Relevance: the substrate and the identity-superposition reading we are
questioning. Their construction's layer-1 positional copy is exactly the
machinery a positional account would need.

## Zhu et al., Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought
arXiv 2509.23365, v3 dated 2026-03-01. Unchanged since the July note. Read in full.

Claim: under the Coconut loss, a single "index-matching logit" mu (the
attention strength from the thought to edge tokens indexed by source node)
converges to a bounded positive value; bounded mu gives soft attention over
all frontier edges (superposition) while divergent mu would commit to one
path. A second pair of logits (carryover to the answer token, candidate
lift) grow together in the prediction stage. Final test accuracy 96.2 percent
(the one exact number for this substrate).

Verified: theorems 1-3 as stated; Figure 3 shows the frontier-minus-non-
frontier attention logit rising and saturating around 60 within stage 1 (about
125 epochs), later stages learn faster; Figure 4 the answer-stage logits.
Attention in the analysis is indexed by source-node identity after a layer-1
copy puts node information on edge tokens. No probing or intervention;
Appendix E.3 notes the candidate lift is not always mediated by the <R> token.
Limitation: theory fixes the thought distribution during the prediction
stage.

Relevance: gives the mechanism vocabulary (index-matching logit, carryover,
lift) and predicts identity-based attention. Supports the identity reading;
does not test it causally.

## Gu and Fu, Emergence of Frontier Superposition: Möbius attractor and Cascade Supervision
arXiv 2605.18820, v1 dated 2026-05-12. Read in full.

Claim: for a depth-D feedforward "cascade" transformer on Erdos-Renyi
reachability in the tree regime, gradient flow contracts embeddings onto a
permutation-symmetric subspace, the layerwise dynamics reduce to a Möbius map
whose zero set contains the equal-weight frontier superposition, and a
"cascade supervision" loss (per-depth, with selectivity bootstrap, gradient
persistence, per-step discrimination) drives every layer to it in BFS order,
while end-to-end supervision attenuates gradients super-polynomially in depth.

Verified: Theorems 4 and 5, Propositions 3 and 7; experiments at D=3, 50
nodes, d=64 with cosine to the target state 0.69 observed vs 0.71 predicted
under cascade supervision and 0.37 vs 0.35 end to end. No interventions, no
recurrence (depth plays the role of steps), no positional analysis.

Relevance: supports the identity-superposition target as a learnable optimum
in a feedforward analogue; explains why staged curricula work. Nearest theory
neighbor after Zhu et al.

## Gozeten et al., Continuous Chain of Thought Enables Parallel Exploration and Reasoning (CoT2)
arXiv 2505.23648, v3 dated 2026-03-05. Unchanged since the July note. Read in full.

Claim: build the continuous token by construction as E^T alpha (a simplex-
weighted sum of vocabulary embeddings from the softmax), train with soft
targets and GRPO, and superposition is imposed rather than emergent. Robust
decoding of a budget-B superposition needs d = Omega(B log(v/B)) (Appendix
D); Figure 2c sweeps d in {16,24,32} and B in {1,4,8,16}.

Verified: no edits to latents after production; sampling policies (multi-token
sampling, Dirichlet) control parallelism at generation time. ProsQA 93.37
(SFT) to 94.21 (GRPO); COCONUT baseline lower in Figure 2b. Explicitly not a
COCONUT imitation.

Relevance: contrast case. Their thought is a mixture over identities by
construction; whether an emergent thought is one is the open question.

## Xu and Sato, A Formal Comparison Between Chain of Thought and Latent Thought
arXiv 2509.25239, v3 dated 2026-05-12. Abstract only.

Claim: latent thought admits more efficient parallel computation; textual
CoT admits approximate counting and sampling via stochastic decoding. No
graph task or superposition specifics in the abstract.

Relevance: citation for the expressivity separation; nothing on encoding.

## Capabilities and Fundamental Limits of Latent Chain-of-Thought
arXiv 2602.01148, v1 dated 2026-02-01. Unchanged. Read in full.

Claim: a "symbolic index" (top-token probability mass) governs an exploration
vs precision trade-off; latent CoT keeps it in 0.2-0.5 while explicit CoT sits
near 1. Compounding-error theorem: E||E_M||^2 = (1-L^{2M})/(1-L^2) d sigma^2
under iid Gaussian per-step noise (Theorem 4.8). Curriculum is necessary
(Theorem 5.1) and sufficient (Theorem 5.2). GPT-2, M=6 latents, GSM8K and
ProsQA; removing the curriculum drops ProntoQA from 99.8 to 52.4 (Table 1).

Verified: only representational analysis is a PCA (Figure 5) showing latents
converge to a compact set. No probing, no intervention, no limitations
section (Remark 4.4 is the closest).

Relevance: gives a functional form for noise sensitivity of latents; nothing
on content.

## Huang et al., Transformers Provably Learn to Internalize Chain-of-Thought
arXiv 2605.28600, v1 dated 2026-05-27. Abstract only.

Claim: k-parity with multi-layer transformers; a logarithmic curriculum
("Log-ICoT") that removes reasoning tokens in geometric chunks; depth log2 k
suffices. No continuous thoughts.

Relevance: curriculum theory only.

---

# Part 2. Does superposition actually happen, and is it used?

## Rizvi-Martel, Rabusseau and Mosbach, The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models
arXiv 2604.06374, v2 dated 2026-08-03 (v1 2026-04-07). Read both versions in full.

Claim: across three regimes only from-scratch models show signs of
superposition. Soft Thinking tokens in pretrained models are processed like
their argmax token (cosine 0.998 for Qwen2-1.5B, 0.996 for QwQ-32B; KL about
1e-4 in middle layers). Fine-tuned Coconut (GPT-2, and in v2 SmolLM2 135M to
1.7B) barely uses its latents: removing them costs at most 1.0 percent in v2
(v1 reported 96.6 without vs 99.0 with for GPT-2), and entity probes show the
target dominating from step 0, a shortcut. From-scratch 2-12 layer GPT-2-style
models on ProsQA with a 40-token symbolic vocabulary lose 94.5 to 13.8 percent
(2-layer) when latents are removed, and their probes show correct-next and
wrong-neighbor entities dominating intermediate steps.

What changed v1 to v2: added SmolLM2 fine-tuned models, a width-vs-depth
ablation at fixed parameters (2-4 wide layers above 90 percent; 8-12 narrow
layers 72-82), a uniform-superposition control (k=3,10,15 tokens; collapse
persists), and Pythia-1B pretraining checkpoints showing collapse emerging with
training tokens. The limitations paragraph now names Butt et al.'s RL soft
tokens, recurrent frameworks, and internalized latent steps as untested; v1's
mentioned layer-wise circuits instead. Discussion now suggests the advantage
of continuous thoughts may be flexibility rather than superposition.

Verified: their causal test is whole-latent removal only; no per-component
edit; no positional analysis; does not engage Zhu et al.'s inner-product
evidence directly, only their capacity remark.

Relevance: nearest empirical neighbor on our exact substrate (from-scratch
2-layer on ProsQA). Their result that the thought is necessary matches ours.
Their probes are entity-identity probes; they do not ask whether identity is
what the thought carries.

## Dilgren and Wiegreffe, Are Latent Reasoning Models Easily Interpretable?
arXiv 2604.04902, v2 dated 2026-08-10 (v1 2026-04-06), COLM 2026. Read both versions.

Claim: for Coconut and CODI on GPT-2 and Llama-3.2-1B, latent tokens are
almost never needed on ProntoQA and ProsQA (stable answers with zero latents),
and a controlled "multi-mode" model shows the training regimen rather than
inference compute drives the gains. When latents are needed (GSM8k-Aug), gold
traces are recoverable from correct predictions by vocabulary projection plus
backtracking 54-93 percent of the time (65-93 in the abstract), and a
forward-chaining decoder finds verified traces for most correct but few
incorrect predictions.

What changed v1 to v2: numbers and sections match; the COLM camera-ready adds
polish rather than experiments as far as the text shows.

Verified: no causal intervention beyond removing latents; authors call their
evidence correlational and note it may be an artifact of gold-trace training.

Relevance: on fine-tuned pretrained models, ProsQA latents are unnecessary,
the opposite of the from-scratch regime. Their vocabulary-projection method is
identity decoding, which is the assumption we are questioning.

## Aswal, Palmeira Ferraz, Zhou and Peyrard, Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models
arXiv 2606.12689, v1 dated 2026-06-10. Read in full.

Claim: on GPT-2 fine-tuned as COCONUT, CODI, and matched controls (pause-as-
thought with no recurrence; a perturbed-curriculum Coconut; base; explicit
CoT) on ProsQA and GSM8k, the BFS-like frontier-mass pattern and the logit-lens
"scratchpad" hits appear identically in controls without recurrence, so they
are not evidence of latent reasoning. Causal tracing (Meng-style, per position
and layer) on ProsQA puts the indirect effect at prompt positions and near
zero at individual thought positions; on arithmetic the thought block matters.
Gradient-subspace interventions: the loss-sensitive subspace of a thought is
low rank (11.8-32.8 of 768 dims on ProsQA, 127-142 on arithmetic); zeroing it
leaves ProsQA accuracy intact, amplifying it flips answers well above random
directions. Trajectories on ProsQA are nearly static (identity map fits best)
and the causal subspace is stable across steps; on arithmetic it rotates.

Verified: Table 3 ranks, Figure 5 stability, limitations paragraph (linear
subspace estimate; off-distribution interventions; small models, K=6; only
final checkpoints).

Relevance: nearest published causal analysis of what a thought does on
ProsQA, and the only one that separates prompt positions from thought
positions. On fine-tuned GPT-2 the answer is "prompt positions"; on our
substrate the thought is necessary, so the same tracing would be informative
rather than trivially null. Their random-direction control for subspace edits
and their observation that causally active directions need not decode
cleanly are both directly usable. Scoop risk: they propose model diffing
across curriculum stages, not per-branch or positional decomposition.

## Zhang, Tang, Ju, Duan and Liu, Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought
arXiv 2512.21711, v1 dated 2025-12-25. Read in full.

Claim: on 7-8B instruct models fine-tuned COCONUT-style (ProntoQA, MMLU,
HotpotQA, AdvBench, PersonalityEdit), latent tokens are insensitive to
orthogonal perturbation (success rate under 5 percent vs up to 50 for textual
CoT), swapping the five latents between samples leaves accuracy near 60
percent with 17.9 percent answer changes (CoT: 43.4, 52.8), and training with
planted shortcuts is exploited. Latents sit far from the vocabulary embedding
manifold; the authors call them "uninterpretable placeholders."

Verified: no ProsQA, no from-scratch model, no positional analysis; authors
say their causal link is not formally established and the shortcut analysis
is speculative.

Relevance: the strongest "thoughts do nothing" result, in the fine-tuned
regime only. The swap-between-instances test is the ancestor of transplant
tests.

## Cui et al., How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?
arXiv 2602.22441, v1 dated 2026-02-25. Abstract only.

Claim: latent methods show shortcut behavior; latent representations "encode
multiple possibilities but do not implement structured search faithfully";
stronger supervision reduces shortcuts but narrows hypothesis diversity.

Relevance: consistent with Rizvi-Martel and Aswal; methods not read.

## Wu et al., LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking
arXiv 2508.03440, v4 dated 2025-10-16. Read in full.

Claim: on 32B reasoning models, soft-token predictions match top-1-token
predictions (Jensen-Shannon divergence concentrated at 0); the model follows
the greedy component. Injecting randomness (Gumbel-softmax, Dirichlet) is what
recovers alternative paths (average 72.1 to 79.6 on their suite).

Relevance: independent confirmation of the training-free collapse that
Rizvi-Martel report; not about emergent thoughts.

## Backour, The Dynamics of Continuous Mixture Collapse in Language Models
arXiv 2609.02049, v1 dated 2026-09-02. Read in full.

Claim: a weighted mixture of token embeddings fed to a pretrained model
collapses for three independent reasons: the architecture already distorts
mixture geometry, training amplifies it, and the softmax readout fed back
autoregressively is a dynamical system with a critical coupling (L=2) above
which one component wins and below which mixtures wash out. Trained
Qwen3.5-4B recovers requested mixture weight with calibration 0.33 vs 0.73
for matched random weights; temperature on the feedback crosses the predicted
threshold. Cites Zhu et al. as showing from-scratch training can create
continuous states that pretrained models lack.

Relevance: explains why superposition collapses in pretrained models and, by
contrast, why a from-scratch model is the right place to ask what a
superposed state actually is. No claim about content.

## Larson, The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA
arXiv 2609.03090, v1 dated 2026-09-02, ICML 2026 Mechanistic Interpretability Workshop. Read in full (PDF text).

Claim: route each CODI latent through a 16x16 matrix so rank becomes an
observable; if singular directions were parallel paths, rank-k truncation
should hurt. Across four training conditions on GPT-2 small (ProsQA and
GSM8k-Aug) the rank-k curve is flat within 0.6 points; three seeds reach
81.0 +/- 2.0 percent at effective ranks {4, 12, 13}; four nonlinear readouts
also flat. A linear probe on the matrix thought predicts the ProsQA target
with AUC 0.673, below a raw pretrained GPT-2 hidden state (0.846). A vanilla
SFT control reproduces the flat curve, so the author notes the truncation
test alone cannot separate rank-blindness from latent positions being
irrelevant.

Relevance: another fine-tuned-GPT-2 result where the latent is not where the
answer lives. The confound the author names (an ablation that shows nothing
because the position does nothing) is one to keep in mind for any
component-level edit.

## Jin, Yang and Wang, Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories
arXiv 2607.06648, v2 dated 2026-09-01 (v1 2026-07-07). Read in full.

Claim: build verified counterfactuals on ProsQA by rewriting a single directed
edge (405 pairs; a BFS oracle gives the new answer) and measure whether a
model that answered the original correctly follows the edit. GPT-2 COCONUT
accuracy rises from 73.1 to 90.1 percent across training while counterfactual
following falls from 71.3 to 27.4; CODI never follows (under 1 percent);
Llama-3.2-1B COCONUT loses 42 points of following during stages 2-3. Latent
patching with zero, mean, and norm-matched noise shows format-dependent
sensitivity on GSM.

Verified: limitations paragraph (oracle only for tasks with a solver;
following does not say how the answer is produced; GSM comparison
descriptive).

Relevance: the cleanest published on-manifold counterfactual for ProsQA
(edit one edge, keep everything else). The finding that a fine-tuned model
learns to stop reading the graph is the fine-tuned-regime version of "the
thought encodes nothing about this graph." Method we can use: single-edge
rewrites as the minimal pair.

## Seddik and Fard, Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs
arXiv 2606.27378, v1 dated 2026-05-07. Read in full.

Claim: a thought representation should be causal (substitutes for the
reasoning tokens with little KL change), minimal, separable (distinguishes
non-equivalent outputs under a bounded probe), and stable. Audited on five
8B-70B models over 23 BBEH tasks, no candidate (last input token, Soft
Thinking, latent thinking at 1-128 steps) beats the plain input embedding on
every axis; within-task separability collapses to near chance for all but the
output embedding. Explicitly brackets what the representation encodes.

Relevance: defines "used" functionally without saying what is encoded; their
separability collapse is another face of Rizvi-Martel's illusion. Not a
scoop.

---

# Part 3. Causal interventions on continuous thoughts

## Li et al., Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure
arXiv 2602.08783, v3 dated 2026-05-28 (v1 2026-02-09, v2 2026-03-17), ICML 2026. Read in full; matches the July note.

Claim: treat each latent step as a variable in a structural causal model and
do single-step overwrites (zero, replacement) on COCONUT and CODI over GPT-2,
Llama3-1B, Qwen3-4B on GSM8K, CommonsenseQA, StrategyQA. Findings: causal
leverage is concentrated in a few steps; arithmetic flips more (0.1-0.2+)
than commonsense; stronger backbones flip less; influence graphs are non-
local (skip connections, locality below explicit CoT's 0.6+); teacher-forced
readouts commit early while probe readouts show competition persisting until
a late collapse.

Verified: whole-step interventions only (Definition 3.1); no decomposition;
no claim about content; limitations paragraph as quoted in the July note.

Relevance: establishes step-level causal method and the readout-dependent
"competing modes" observation. Not on graphs. Method precedent for teacher-
forced vs probe readouts.

## Chang et al., Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention
arXiv 2606.01243, v1 dated 2026-05-31. Unchanged. Read in full; matches the July note.

Claim: on 3-8B COCONUT and CODI (GSM8K family, StrategyQA), latents align
with explicit-CoT states (CKA 0.72, linear mapper cosine 0.75, lexical probe
top-1 0.38); whole-vector causal probes flip answers (slerp toward an answer-
supporting exemplar 21.5 percent, one norm-constrained gradient step 24.1,
whole-sequence transplant 48.3 directional transfer; drop-all 10.7); training-
free decode-time whole-vector edits gain 0.4-1.8 points.

Verified: all edits are whole-vector and norm-preserving (Eqs 12-16); no
superposition, graphs, or content claims; no negative results; limitations
are compute overhead only.

Relevance: precedent for slerp and transplant on continuous thoughts and for
norm-preserving edits. Their transplant works across instances at 8B; ours
only when graph structure differs.

## Ding, Huang and Yang, SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models
arXiv 2608.27265, v1 dated 2026-08-27, EMNLP 2026. Read in full.

Claim: ask which transformer object carries the effect of the latent steps.
The Suffix Cache Interchange Test patches slices of the key/value cache
written during the latent steps (by layer, head, key vs value, suffix length)
from a source rollout with an exact counterfactual into a recipient rollout,
scores by closed-set target-win, and requires sufficiency (oracle patch wins
at least 0.80), selectivity (the main disjoint competitor at most 0.25), and
necessity (matched random corruption of the same slice drops target-win by
0.50). On CODI-GPT2 two-hop arithmetic the carrier is the value cache of
layers 8-9 over at least three latent positions (0.875-0.908 sufficiency;
layers 10-11 0.017-0.029; k=1 only 0.203; corruption to 0.211). Hidden-state-
only, key-only, same-answer, partial-variable, and cross-template sources all
fail. At 1B the latent-tail carrier persists; at 8B the prompt-prefix cache
is sufficient from step 1 and the latent tail is inactive; zeroing prompt K/V
collapses the readout while zeroing the latent suffix does not. New CODI
seeds with high accuracy do not reproduce the layer 8-9 carrier. Graph and
finite-state cells fail their competence or support checks and get no call.

Verified: limitations (one arithmetic template family; single checkpoints;
non-monotonic scale; greedy decoding). Future work names loop transformers
with carriers mapped per iteration, and a training-time interchange loss. No
code link on the abstract page.

Relevance: nearest neighbor in method and closest to a positional account
without stating one: their "context-bound value-cache trajectory" is a cache
of what was written at particular positions, and at 8B the visible prompt
fields are what the readout depends on. They never test a from-scratch model
or graph reachability successfully. Their carrier-status contract
(sufficiency, selectivity, matched-corruption necessity) is a good template
for reporting any transplant result.

## Kshirsagar, The Weight of Silence: A Causal Case for Weights Over the Scratchpad in Latent Chess Reasoning
arXiv 2607.20952, v2 dated 2026-08-24 (v1 2026-07-23). Read in full.

Claim: on Qwen3-14B LoRA COCONUT for chess moves, replacing the four thought
vectors with a fixed average (board-blind) or matched-norm noise leaves legal-
move rate unchanged (48 to 48 percent; 58 to 60 after RL); removing them
costs 4 points; only exact zeros collapse it. J-lens on layers 26-29 shows
cross-board cosine about 0.99 among thoughts. The thoughts are a fixed
scaffold and the training benefit lives in the weights.

Verified: single seed, single domain, 100 test positions (limitations
section).

Relevance: a clean six-condition battery (baseline, substitute, noise,
ablate, length-matched ablate, zero) for asking whether content or mere
presence matters. Also an example of zeros being off-manifold while noise is
not.

## Parekh, Thinking Wrong in Silence: Backdoor Attacks on Continuous Latent Reasoning (ThoughtSteer)
arXiv 2604.00770, v1 dated 2026-04-01. Read in full.

Claim: one learned input embedding for a trigger token hijacks the latent
trajectory of Coconut (GPT-2 on ProntoQA, ProsQA) and SimCoT (Llama-3.2-1B/3B
on GSM8K) with at least 99 percent success. Logit-lens classifiers on each
individual thought still favor the correct entity by 37-43 log-odds while the
output is 100 percent wrong; the maximum single-head causal effect is 0.03 vs
0.90 for token-level backdoors. Thoughts collapse toward a simplex structure
(neural collapse).

Relevance: a striking readable-but-overridden result on ProsQA thoughts: the
per-step identity readout is intact while the answer is determined
elsewhere. Consistent with identity being readable from the thought but not
what the readout consumes.

## Ramjee, Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models
arXiv 2604.23460, v1 dated 2026-04-25. Unchanged; matches the July note.

Claim: dual-trigger backdoor in GPT-2 COCONUT on a moral-reasoning dataset;
linear probes transfer from the released condition to the armed condition
with 89.4 percent at the first latent falling to 58.3 at the sixth ("plan
then suppress"). Detection only; limitations say so verbatim.

Relevance: probe precedent on whole thoughts; no intervention.

## Liang and Pan, Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks
arXiv 2602.00449, v1 dated 2026-01-31. Read in full.

Claim: CODI on modular polynomial iteration. For 2-3 hops the intermediate
state is decodable at latent positions (logit-lens probability 0.36-0.71) and
there are two streams: the latent channel builds intermediates while the final
input is routed to the answer position by copy-like attention heads. For 4 or
more hops only the last intermediate is decodable (a late bottleneck) and with
prime moduli the signatures vanish.

Verified: logit lens, linear probes, attention analysis, activation patching;
the last operand is not represented in the latents but arrives at readout by
attention.

Relevance: the one mechanistic latent-CoT study that finds a split between
what the thought holds and what attention copies from the prompt at readout.
Different task and model; same shape as the positional idea.

## Duraipandian et al., Interpreting Latent CoT Reasoning as Dynamical Systems
arXiv 2607.09698, v1 dated 2026-06-20, ICML 2026 workshop. Read in full.

Claim: on GPT-2 GSM8K, CODI latents contract (stable attractor) while
COCONUT latents expand (positive Lyapunov spike at step 3); SIM-CoT
supervision tightens both. Principal directions encode reasoning phase, not
problem type. No interventions.

Relevance: descriptive; their "phase not problem" finding is a hint that
step index dominates thought geometry.

## Chen et al., What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic Analysis
arXiv 2606.20075, v1 dated 2026-06-18. Read in full.

Claim: mutual information between latents and explicit steps, estimated by a
shared decoder probe, tracks accuracy; outcome-only supervision gives 9.8-
18.3 percent on GSM8k-Aug vs 43.1 explicit, trajectory supervision 31.2,
plus generative reconstruction 39.1; later latent positions get weaker
gradients. Cites the shortcut critiques.

Relevance: probe-only; a reminder that decodability tracks training signal.

## Jerge and Evans, NoisyCoconut: Counterfactual Consensus via Latent Space Reasoning
arXiv 2605.08221, v1 dated 2026-05-06. Abstract only.

Claim: inject noise into latent trajectories, use agreement as confidence,
abstain when paths disagree. No content claims.

Relevance: noise on the trajectory as a tool, nothing more.

## Yerram, He and Choi, Training Continuous Chain of Thought Models: A Tale of Two Regimes
arXiv 2607.16972, v1 dated 2026-07-18. Read in full.

Claim: direct supervision (latents as averaged token embeddings via multi-
token prediction) vs indirect (autoregressive latents distilled from a
teacher) on 1-1.5B models and GSM8k variants; neither matches CoT on realistic
traces. No mechanistic analysis.

Relevance: none beyond background.

## Fan, Svete and Lee, Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers (LOTUS)
arXiv 2606.31779, v2 dated 2026-07-13. Read in full.

Claim: looped padded transformer with parallel cross-entropy supervision on
gold CoT tokens reaches 70.0 vs 71.5 explicit on GSM8K at 3B; post-loop
latents decode gold steps through the base LM head (70.9 percent top-1) and
assign graded probability to unseen valid chains (15.3 top-1, 64.0 top-5).

Relevance: latents made decodable by design; contrast with emergent thoughts.

## Wu et al., J-CoT: Chain-of-Thought in J-Space
arXiv 2607.21981, v1 dated 2026-07-24. Read in full.

Claim: use Gurnee et al.'s vocabulary-indexed J-space as the recurrent
interface between cycles of latent reasoning: write the state as nonnegative
elastic-net coefficients over J-lens vectors at layer 28, read it back at
layer 12 through a layer-specific dictionary. Qwen3-8B-Base: 47.9 percent
average with no training vs 47.5 for SIM-Coconut; 50.2 trained. On ProsQA an
interface sweep peaks at 88.8 for J-thought vs 84.0 dense latent vs 79.0
linguistic. No claims about what dense latents encode.

Relevance: shows a J-lens basis is usable as a carrier for latent reasoning
at 8B; not an analysis of emergent thoughts.

## Yu et al., GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning
arXiv 2608.02585, v2 dated 2026-08-10. Read in full.

Claim: optimize latent states inserted at an intermediate layer at test time
with the model frozen; gradient attribution concentrates on reasoning
connector tokens; early-to-middle layers work best.

Relevance: none for encoding; a test-time latent method.

## Suleymanzade et al., MUX: Continuous Reasoning via Multiplexed Tokens (arXiv 2607.18264, v1 2026-05-19) and Xiong et al., SuperThoughts (arXiv 2606.13862, v1 2026-06-11)
Abstract only (MUX) and full read (SuperThoughts).

Claim: both compress spans of discrete reasoning tokens into one latent by a
lossless (MUX, geometric weights) or learned (SuperThoughts, compressor plus
multi-token prediction) superposition. SuperThoughts contains no
interpretability analysis; MUX's abstract says probes show faithful content.

Relevance: constructed superposition again; a contrast class, not a neighbor.

## Butt et al., Soft Tokens, Hard Truths (arXiv 2509.19170, v2 2025-09-24); Zhang et al., Soft Thinking (arXiv 2505.15778, v1 2025-05-21); Zhang et al., Making Small Language Models Efficient Reasoners (arXiv 2505.07961, v3 2025-05-23)
Abstract only, all three.

Butt: RL-trained continuous CoT with noise for exploration; no
interpretability. Soft Thinking: training-free probability-weighted
embedding mixtures; no intervention analysis. The 2505.07961 "intervention"
is about stopping textual reasoning, not latents; the July note's flag is
resolved as irrelevant.

## Fu et al., DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning
arXiv 2607.00341, v2 dated 2026-07-27. Abstract only.

Claim: in looped transformers the bridge entity becomes nearly perfectly
decodable after the first loop, yet the hidden state stays poorly aligned with
the bridge token's embedding; keeping a discrete embedding channel alongside
the continuous state fixes multi-hop training.

Relevance: a published case of identity being decodable from a latent while
the latent is not near the token embedding. That is the same gap our
embedding-basis edits fell into.

## Chang et al., Why Struggle with Continuous Latents? Interpretable Discrete Latent Reasoning via Rendered Compression
arXiv 2606.29712, v1 dated 2026-06-29. Abstract only.

Claim: continuous latents are hard to train and uninterpretable because they
lack discrete anchors; render them to discrete tokens instead.

Relevance: background.

---

# Part 4. Recurrent-depth and looped models

## Geiping et al., Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)
arXiv 2502.05171, v2 dated 2025-02-17. Read in full; matches the July note.

Claim: 3.5B prelude-core-coda model, core of 4 layers recurred 32 times on
average in training and up to 64 at test. Latent trajectories show
convergence, orbits (arithmetic and structural tokens), and sliders (drift
along one direction, possibly counting). Observational only; no probing or
intervention.

Relevance: parallel substrate; orbits are the one geometric motif anyone has
reported in a recurrent latent.

## Lu et al., Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer
arXiv 2507.02199, v2 dated 2025-09-28. Read in full.

Claim: logit lens and a "coda lens" on Huginn arithmetic show no structured
rank trajectory of intermediate results across recurrences; the two lenses
disagree sharply at the fourth block; GSM8K without CoT goes 3.1 to 4.9
percent across 4-32 steps vs 24.9-38.1 with CoT. No interventions.

Relevance: readout method dependence in a recurrent latent.

## Blayney et al., A Mechanistic Analysis of Looped Reasoning Language Models
arXiv 2604.11791, v1 dated 2026-04-13. Read in full.

Claim: in Ouro-1.4B, Huginn, and retrofitted Llama/OLMo, each layer in the
cycle converges to its own fixed point so the block follows a cyclic
trajectory; attention patterns stabilize after the first iteration; the
recurrent block repeats the same stages of inference as a feedforward model.
No interventions.

Relevance: background for recurrent substrates; nothing on content.

## Wang and Reid, Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?
arXiv 2609.01924, v1 dated 2026-09-01. Read in full.

Claim: fit J-lens families to virtually unrolled loops of Ouro-2.6B and
Huginn. J-space interventions remain causal but interface-dependent: in Ouro
a write must be clamped across every remaining loop (38 vs 21-29 percent) and
the semantic component does not cross the loop boundary; in Huginn writes and
ablations act within a sliding window of about two recurrences and content is
regenerated from the re-injected input rather than carried.

Verified: limitations (mean-Jacobian lenses are linearized, single-token,
blind to content not driving the next token).

Relevance: the one attempt to apply a causal-basis lens across a recurrent
latent; its "regenerated, not carried" finding is the recurrent version of
"the identity is looked up again rather than stored."

## Sharma and Vu, Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models
arXiv 2606.24898, v1 dated 2026-06-12. Read in full.

Claim: scale-invariant readouts (RMSNorm) make hidden-state norm invisible to
the loss while pre-norm recurrence keeps growing it (norms reach 39,207 and
56,051 in 44M and 129M models); the latent carries a variable the readout
cannot see.

Relevance: a concrete example of a latent quantity that is active but
unreadable through the output interface.

## Lam-Muir, The Ignition Is Real, and It Lives at the Readout
arXiv 2608.03263, v1 dated 2026-08-04. Read in full.

Claim: in a 30M recurrent-depth model on a synthetic 2,000-fact world,
intermediate hops are never recoverable through the tied vocabulary readout,
the decision margin jumps 5.8-8.0 logits in one iteration, and post-commit
motion is mostly radial (0.961 of squared norm); the commit is an interface
event.

Relevance: small-model evidence that composed intermediates stay latent and
the answer identity is constituted at readout. Single substrate, single
author, statistical caveats stated.

## Fein-Ashley and Rashidinejad, Solve the Loop: Attractor Models for Language and Reasoning
arXiv 2605.12466, v1 dated 2026-05-12. Read in full.

Claim: train recurrence as a fixed-point problem in output-embedding space;
models learn to make iteration unnecessary at inference. Cites Rizvi-Martel
once for fragility; no interventions.

Relevance: background only.

---

# Part 5. Positional and pointer representations, binding, readout

## Olsson et al., In-context Learning and Induction Heads
transformer-circuits.pub, 2022. Page re-read.

Claim: a previous-token head writes "what came before me" into each position;
an induction head then attends from the current token to the position after
the earlier occurrence and copies what it finds. Six lines of evidence tie the
heads to in-context learning.

Relevance: the canonical two-head circuit where one head does "where" and the
other does "what"; the ancestor of every pointer-style mechanism below.

## Feng and Steinhardt, How do Language Models Bind Entities in Context?
arXiv 2310.17191, v2 dated 2024-05-06, ICLR 2024. Read in full.

Claim: models attach abstract binding-ID vectors to entity and attribute
tokens; swapping attribute activations rebinds (control above 99 percent,
swapped below 3), mean-difference interventions on binding vectors reverse
answers, and rearranging token positions with RoPE interventions barely
changes beliefs, so the IDs are abstract rather than positional. Retrieval
finds the attribute sharing the queried entity's ID. Appendix E shows a task
using direct binding instead.

Relevance: establishes that "which one" can be carried separately from "what"
and dereferenced at readout. Method precedent for mean-difference
interventions on the "which" code.

## Dai, Heinzerling and Inui, Representational Analysis of Binding in Language Models
arXiv 2409.05448, v3 dated 2024-10-25. Read in full.

Claim: an ordering-ID direction (entity sequence order, distinct from absolute
position, shown by fillers and interjections) lives in about two principal
components of middle layers of Llama-2-7B; stepping activations along it
shifts which attribute is retrieved by one slot per step (logit differences
up to about 10).

Relevance: a low-rank "which slot" code, causally steerable, distinct from
both content and raw position.

## Gur-Arieh, Geva and Geiger, Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context
arXiv 2510.06182, v2 dated 2026-05-28. Read in full.

Claim: across nine models and ten binding tasks, retrieval mixes a positional
mechanism (index the group by position; strong at first and last groups,
about 80 percent of behavior there, about 20 percent in the middle), a
lexical mechanism (retrieve from the group containing the query), and a
reflexive direct pointer. Distinguished by interchange interventions whose
counterfactuals separate the three; the mixed model fits at 0.95 Jensen-
Shannon similarity vs 0.44 positional alone.

Relevance: the clearest statement that "where" is one of several retrieval
codes and that counterfactual design is what separates them. Their
counterfactual taxonomy is reusable.

## Bogdan and Lindsey, Slot Machines: How LLMs Keep Track of Multiple Entities
arXiv 2604.21139, v1 dated 2026-04-22. Read in full.

Claim: Qwen3-32B holds a current-entity slot and a prior-entity slot in
orthogonal subspaces on the same tokens; at the answer position attention
reads the current-entity slot on the original entity tokens, not the prior-
entity copy, a dissociation between information that is present and
information that is used. Patching keys matters more than values for
sequence retrieval.

Relevance: readable-but-unused shown inside a pretrained model with attention
doing the readout; slots are positional relative to entity order.

## Oh and Demberg, A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models
arXiv 2606.08644, v1 dated 2026-06-07. Read in full.

Claim: after a swap, models keep the original bindings and rebind at
retrieval through a five-group head circuit (38 heads, 5.7 percent, recover
0.89 of accuracy) in which a dereferencer group matches binding IDs in query
and key subspaces and an answer group attends to the object token.
Interchange on the dereferencer moves the pointer (delta logit 6.79) with
almost no content transfer (0.26).

Relevance: pointer dereference by attention at the answer position, shown
causally with pointer and content separated.

## Urrutia et al., Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization
arXiv 2605.31558, v1 dated 2026-05-29. Read in full.

Claim: a head is positional if its attention is invariant to token
permutations and symbolic if equivariant; on two structurally matched multi-
hop tasks trained from scratch (12 layers, one head each, RoPE), positional
heads are learned hop by hop while symbolic heads appear for all hops at once,
RoPE realizes both in one layer, and symbolic mechanisms generalize to 53
times the training length while positional ones barely generalize. No
ablations or patching.

Relevance: gives a permutation-based score for classifying a head as "where"
or "what" that applies to any trained model, including a two-layer one.

## Sanford, Hsu and Telgarsky, Transformers, parallel computation, and logarithmic depth
arXiv 2402.09268, v1 dated 2024-02-14. Abstract only.

Claim: constant self-attention layers simulate constant rounds of massively
parallel computation; logarithmic depth suffices for tasks (including k-hop
pointer chasing) intractable for other sequence models.

Relevance: theory background for pointer chasing by attention.

## Mazaheri, Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form
arXiv 2608.15022, v1 dated 2026-08-15. Read in full.

Claim: in Qwen3.6-27B, language identity is linearly decodable everywhere but
becomes verbalizable at the query position only after attention gathers it in
a mid-depth window (layers 36-42, carrying 85 percent of the effect; MLPs
oppose); the variable is transported, not unmasked.

Relevance: identity made readable at the readout position by an attention
lookup, in a pretrained model; the same shape as "identity is looked up from
the prompt at the last step."

## Oskin, Off-Axis, On Purpose: Where a Transformer Computes Concepts and Why it Does So
arXiv 2608.10251, v1 dated 2026-08-10. Read in full.

Claim: in GPT-2-small trained from scratch, attention writes sit 75-96
degrees off the unembedding axis at every depth; rotating values toward the
readout before mixing is 64-84 times more damaging than a random rotation;
the answer is added late rather than rotated in (R^2 of a rotation fit 0.38).

Relevance: an argument that computation lives in directions the readout
cannot decode, and that readable and used components are misaligned by
design. Many from-scratch runs fail to converge (stated).

## Niu, Repeated Shared Access Enables Grokking, but Edit Propagation Depends on an Addressable Memory
arXiv 2606.20737, v2 dated 2026-06-23. Read in full.

Checked and set aside: the "addressable memory" is an external expert store,
not attention over prompt positions. Not relevant.

---

# Part 6. Readable versus used; causal-basis methods

## Belinkov, Probing Classifiers: Promises, Shortcomings, and Advances
arXiv 2102.12452, v4 dated 2021-09-22, Computational Linguistics. Abstract page.

Claim: decodability does not imply use; recommends control tasks, amnesic
probing, and causal methods.

Relevance: the standard citation for the distinction.

## Hewitt and Liang, Designing and Interpreting Probes with Control Tasks
arXiv 1909.03368, v1 dated 2019-09-08, EMNLP 2019. Abstract page.

Claim: pair each probe with a control task of random word-type to label
assignments; selectivity is linguistic accuracy minus control accuracy;
dropout does not improve selectivity, other regularizers do.

Relevance: control-task baseline for any probe we fit.

## Ravfogel et al., Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection
arXiv 2004.07667, v2 dated 2020-04-28, ACL 2020. Read at method level (ar5iv); matches the July note.

Claim: repeatedly train a linear classifier for the attribute, project the
data onto its nullspace, and stop when no classifier beats chance; compute the
combined projection stably as the nullspace of the sum of rowspace
projections. Only linear guarding: an RBF SVM still recovers the attribute.

Relevance: the subspace-removal method we use.

## Elazar et al., Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals
arXiv 2006.00995, v3 dated 2021-02-19, TACL 2021. Read at method level (ar5iv); matches the July note.

Claim: remove a property with INLP, run the original model on the projected
representation, measure the behavioral change; compare against removing the
same number of random directions (Rand); optionally re-inject the gold
property and fine-tune to check selectivity. Probe accuracy does not
correlate with behavioral importance (Spearman 0.085); phrase boundaries are
decodable but unused.

Relevance: the rank-matched random control is the load-bearing control for
any subspace removal.

## Dobrzeniecka, Fokkens and Sommerauer, Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE
arXiv 2506.11673, v1 dated 2025-06-13. Read in full.

Claim: INLP's many iterations (up to 820 projections for 41 classes) distort
representations broadly (cosine to original 0.31 vs 0.83 for mean projection
on dependency labels) and can hurt more than random projections; mean
projection (one direction per class: class mean minus rest mean) and LEACE
remove the target more surgically and give comparable results; the three-step
practice (verify presence, compare with rank-matched random, selectivity
control) is recommended. Single-run results, limited properties.

Relevance: argues for mean projection or LEACE over INLP when removing
identity information from a thought.

## Kantamneni et al., Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
arXiv 2502.16681, v1 dated 2025-02-23. Abstract page.

Claim: SAE probes do not consistently beat logistic regression under scarcity,
imbalance, noise, or covariate shift.

Relevance: probes first when labels exist; this is the "SAE critique" the
July note cited without a reference.

## Gurnee et al., Verbalizable Representations Form a Global Workspace in Language Models
arXiv 2607.15495, v1 dated 2026-07-16 (transformer-circuits.pub 2026-07-06). Read in full for the first time this session.

Claim: the Jacobian lens averages the Jacobian from layer l to the final layer
over positions and 1,000 prompts, J_l = E[d h_final,t' / d h_l,t], and reads
concepts as softmax(W_U norm(J_l h)). The J-space is small (never more than
10 percent of activation variance) yet carries what the model can report and
use: swapping J-space coordinates swaps verbal reports 88 percent of the time
vs 5 for non-J components, two-hop factual swaps succeed about 70 percent on
Sonnet 4.5 and Opus 4.5, and ablating J-space kills flexible inference while
continuation and anomaly detection survive. Interventions: steer h + a v,
project out, and swap h + V(sigma(c) - c) with c = V^+ h. A clamp experiment
shows non-J components act only by re-deriving the concept into J-space.

Verified: limitations (single-token concepts; approximate; small integers may
be computed outside the workspace). No mention of latent CoT, COCONUT, or
recurrent models. The readable-vs-used distinction is explicit in section 3.1.

Relevance: the causal-basis method. Everything the July note said about the
method is confirmed by the arXiv text; the findings can now be cited.

## Yan et al., Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens (arXiv 2608.25347, v1 2026-08-26); Gong and Wang, The First Token Is a Clue (arXiv 2608.31084, v1 2026-08-31)
Abstract only.

Yan: the J-lens is an optimal local linear approximation with very sparse
energy concentrated in a few positions, short-horizon by construction. Gong
and Wang: multi-token concepts can be recovered from the first-token J-lens
vector plus model completion (Rank@10 43.1 percent).

Relevance: caveats and an extension for the J-lens; the short-horizon point
matters if a thought's effect only appears several steps later.

## Ma et al., Hidden APIs in Language Models: Discovering Reusable Causal Interfaces from Forked Futures
arXiv 2607.27617, v1 dated 2026-07-30. Abstract only.

Claim: compare prefix states by the response distributions they induce under
operations sampled afterward; identical outputs can hide functionally
different states; probes miss this.

Relevance: a different route to "used" directions (compare futures rather
than fit labels). Not read in detail.

---

# Part 7. Non-paper items

- **SPAR spring 2026, "Interpreting latent reasoning: methods for
  understanding continuous chain-of-thought"**, mentor Uzay Macar (Anthropic
  Fellows). Project page lists five aims for Coconut: causal structure between
  thoughts, anchor identification, probing and decoding, intervention
  techniques analogous to CoT editing, faithfulness. No outputs, papers, or
  posts listed as of 2026-09-06. No arXiv paper by this mentor on latent
  reasoning surfaced in search.
- **Latent and Implicit Thinking workshop, ICLR 2026 (April 27).** The
  workshop's papers page still shows no accepted list; OpenReview PDFs are
  behind a verification screen. Known papers published there: Li et al.
  (2602.08783) and LatentChem. Could not enumerate the rest.
- **The July note's "SAE/crosscoder on thought vectors" gap** still holds:
  no paper training dictionaries on continuous thoughts surfaced.

---

# Techniques from this literature worth considering

- Causal tracing by position and layer on the from-scratch model, so that
  prompt positions, edge tokens, and thought positions are compared under the
  same corruption (Aswal et al.).
- Cache interchange rather than hidden-state interchange: patch keys and
  values written at particular positions, separately, with sufficiency,
  selectivity, and matched-corruption necessity all reported (Ding et al.).
- Single-edge rewrites of the graph as the minimal pair, with a BFS oracle
  giving the counterfactual answer (Jin et al.).
- Counterfactuals designed to separate positional, lexical, and pointer
  retrieval, as in the binding literature (Gur-Arieh et al.), and mean-
  difference interventions on a "which one" code (Feng and Steinhardt; Dai et
  al.).
- The permutation-invariance score for classifying each head as positional or
  symbolic (Urrutia et al.).
- The six-condition presence-vs-content battery: baseline, board-blind
  substitute, matched-norm noise, remove, length-matched remove, zero
  (Kshirsagar).
- Mean projection or LEACE instead of many-iteration INLP when removing a
  labeled property, always with the rank-matched random control (Dobrzeniecka
  et al.; Elazar et al.).
- Gradient-derived low-rank causal subspaces with random-direction controls,
  and checking whether those directions decode at all (Aswal et al.).
- Jacobian-lens directions as a causal basis, with the caveats that they are
  local, short-horizon, and single-token (Gurnee et al.; Yan et al.), and the
  recurrent adaptation of clamping across remaining steps (Wang and Reid).
- Reading the same thought through several readouts (teacher-forced logits,
  trained probe, logit lens) and reporting the disagreements (Li et al.; Lu
  et al.).

# Open questions the field has not answered

- What an emergent continuous thought encodes when it is necessary. Every
  causal study so far is on fine-tuned pretrained models where the thought is
  weakly used or a scaffold; the from-scratch case has only inner products and
  entity probes.
- Whether the BFS-shaped inner-product pattern is causal. Aswal et al. show
  the pattern appears in controls without recurrence; nobody has shown that
  changing the pattern changes the answer on a model where the thought
  matters.
- Whether identity is stored in the thought or looked up from the prompt at
  readout. The binding literature shows pointer dereference at readout in
  pretrained models; no latent-CoT paper has asked the question.
- Whether "positional" means absolute position, relative position, or an
  abstract ordering code, in any latent-CoT model.
- Why per-component linear edits in the embedding basis fail while whole-
  vector edits and transplants succeed. DiscoLoop's decodable-but-misaligned
  bridge entity and Oskin's off-axis result are suggestive, not answers.
- How to find a causal basis for a two-layer model, where a Jacobian to the
  answer logits is exact and cheap but nobody has reported doing it.
- Whether carriers change across training (Jin et al. and Aswal et al. both
  point at training trajectories; only Jin et al. measured anything, and only
  behaviorally).
- Whether anything holds at scale: SCIT finds the carrier moves from the
  latent tail to the prompt prefix between 1B and 8B, with a 4B model giving
  no call.
