# Lit scan — thought arithmetic niche (2026-07-06, ~6 queries + 4 paper fetches)

**Verdict: NARROW BUT OPEN.** Existence of thought-superposition is proven (theory) and
correlationally verified; per-branch CAUSAL editing and capacity measurement are absent —
confirmed by the limitation sections of the exact papers that would otherwise scoop us.

## Load-bearing finds (fetched and read at summary level — figures NOT yet verified)

- **arXiv:2505.12514 "Reasoning by Superposition"** — THE theory anchor. Proves 2-layer
  transformer + D continuous-thought steps solves diameter-D graph reachability;
  construction: thought at step i ≈ weighted sum of embeddings of all ≤i-hop frontier
  nodes; superposition emerges in training without supervision. Empirics (v3 html,
  fetched): inner products between thoughts and node embeddings on a 2-layer FROM-SCRATCH
  model (d=768, 8 heads, ProsQA); frontier nodes score high, BFS-wave pattern. Explicitly
  does NOT: probes/SAEs, capacity/interference measurement, causal per-branch ablation,
  pretrained-scale. → Our exact opening.
- **arXiv:2509.23365 "Emergence of Superposition"** — training-dynamics theory (2-layer,
  graph reachability); index-matching logit mechanism balances explore/exploit. No
  decomposition, no capacity. Citation, not competitor.
- **arXiv:2602.08783 "Dynamics Within Latent CoT"** (Feb 2026) — closest empirical work.
  SCM/do-interventions on COCONUT + CODI at WHOLE-STEP level: staged functionality,
  non-local routing, competing answer modes persist, early-output vs late-commitment gap.
  Does NOT decompose within thoughts. Overlap: confirms parallel modes exist → our framing
  must be "causal anatomy + capacity," not "does superposition exist."
- **arXiv:2602.01148 "Capabilities and Fundamental Limits of Latent CoT"** — theory:
  exploration-execution tradeoff (Symbolic Index), compounding-error theorem
  ‖E_M‖² ∝ (1−L_F^(2M))/(1−L_F²)·d·σ_h², curriculum necessity proof. Explicitly NO
  capacity bounds / interference measurement / feature decomposition. → Their error
  theorem gives a functional form our capacity curve can CONFRONT. GPT-2/ProsQA/GSM8K.

## Threats / unverified

- **SPAR sp26 project "Interpreting latent reasoning: methods for understanding
  continuous chain-of-thought"** (sparai.org) — active mentored cohort, page truncated,
  methods unknown. Speed matters; assume months not a year.
- **arXiv:2604.23460 "Ulterior Motives"** — probes continuous-thought models for
  misaligned reasoning (safety). ~~NOT fetched in full~~ → RESOLVED 2026-07-06 evening,
  read in full: detection-only, no intervention; see verification section below.
- No "SAE/crosscoder on thought vectors" paper surfaced in any query; crosscoder-across-
  thought-steps unseen. Moderate confidence only — obvious-next-step territory.
- House rule: before the pre-reg freezes, re-verify 2505.12514's empirical claims against
  its actual figures (not the fetched summary), and skim Ulterior Motives in full.
  → BOTH DONE 2026-07-06 evening; see verification section below. Figure-level check of
  2505.12514 surfaced two bar-affecting design facts (binary task ⇒ 50% chance floor;
  Gate B vacuous as drafted) — resolved with user before freeze.

## 2026-07-06 (evening) — pre-freeze verification: both house-rule debts PAID

**arXiv:2505.12514 v3 verified against the actual PDF (NeurIPS 2025), figures included.**

- Substrate spec confirmed: GPT-2-style decoder, 2 layers, d_model=768, n_heads=8, from
  scratch; AdamW (β1=0.9, β2=0.95, wd=1e-2), constant LR 1e-4; curriculum per COCONUT:
  25 epochs/stage, 300 epochs total, previous-stage mix p=0.1. Code released:
  github.com/Ber666/reasoning-by-superposition (prefer their config for S1).
- Data (Table 4): ProsQA subset, 3–4 hop questions; train 14,785 / val 257 / test 419;
  avg |V|=22.8, |E|=36.5, sol. len 3.5. Each node = dedicated vocab token (clean u_v basis).
- Accuracy: text says "near-perfect"; Fig. 4 bar ≈ 0.97–1.0; NO exact number printed.
  2-layer COCONUT >> 12-layer discrete CoT (83%) and No-CoT (~75%).
- Fig. 6 + Table 5 (3 seeds) verified at figure level: inner products ⟨thought_i, u_v⟩
  segregate Not-Reachable (≈0, −0.37..0.02) << Reachable (0.6–3.7) < Frontier (1.9–5.4)
  < Optimal (4.7–9.6), per-step BFS wave, consistent across seeds. Correlational claim
  REPLICATED IN PRINT as described. Superposition is biased toward frontier+optimal, not
  uniform over reachable set (Lemma 2's uniform 1/√|V_c| is the construction, not the fit).
- **DESIGN FACT WITH TEETH: binary forced choice, "one and only one of c1,c2 is
  reachable by r."** The decoy is globally unreachable ⇒ (a) no sibling branch ever
  contains a candidate — draft's sibling-share S and Gate B's "answer lies in b" are
  ill-defined/vacuous on this task; (b) chance floor = 50% ⇒ max theory-predicted
  subtraction effect ≈ baseline − 50 ≈ 47 pts. The 50-pt drop bar approved this morning
  EXCEEDS the theoretical ceiling → reformulation surfaced to user pre-freeze.
- COCONUT-BFS variant (§5.4): uniform random frontier-node supervision instead of
  optimal-path → same near-perfect accuracy, same superposition pattern. Useful backup
  curriculum + evidence the pattern isn't an artifact of optimal-path supervision.
- Compute (C.3): ~24 h on 2×A100-80GB PER RUN. On M1/MPS this is days, not "hours-scale"
  as §4 assumed — early stopping per stage likely needed; surfaced to user.
- Mechanistic details for interventions: LayerNorm(x)=x/‖x‖₂ (their def) ⇒ thoughts are
  unit-norm — SUBTRACT should renormalize to stay on-manifold; answer head literally
  measures superposition membership of c1/c2 (Theorem 1 proof) ⇒ sharp exploratory
  prediction for INJECT: adding γ·u_decoy should raise decoy share.

**arXiv:2604.23460 "Ulterior Motives" read IN FULL (ICLR 2026, Ramjee).**

- Dual-trigger backdoor in COCONUT-style GPT-2 (124M) on MoralChain (Moral Stories +
  GPT-4o reasoning paths). [T] arms misaligned latent reasoning w/ aligned outputs; [O]
  releases it. PCA trajectories + per-token logistic probes; probe transfer 89.4% at z1
  decaying to 58.3% at z6 ("plan then suppress"; monitor early tokens).
- **Detection only. Limitations §7 verbatim: "we demonstrate detection but not
  intervention — future work should investigate whether identified misalignment
  directions enable inference-time steering."** No decomposition within thoughts, no
  per-branch causality, no capacity. Overlap = linear probes on whole thoughts (rung 3
  of our basis-fallback ladder only). Our niche remains open; cite as probe precedent.

## 2026-07-07 — arXiv:2505.23648 v3 "CoT2" (Gozeten et al., UMich+Google) read IN FULL (34 pp incl. appendix)

User flagged as possible scoop. **Verdict: NOT a scoop of the core claim; real overlap
only on the exploratory capacity tier.** This is the paper 2505.12514 cites for
superposition in the arithmetic domain, now expanded (v3, 2026-03-05) with GRPO/RL,
ProsQA/ProntoQA, and a GSM8K/Qwen3-0.6B preliminary.

- Their continuous token is CONSTRUCTED, not emergent: z_t = E^T α_t, a simplex-weighted
  sum of vocab embeddings from the softmax, trained to match hindsight top-B trajectory
  distributions (CSFT). Superposition is imposed by design — on CoT2 "subtract a branch"
  is trivial α-editing, no open causal question. Explicitly differentiates itself from
  COCONUT ("does not initialize from, nor attempt to mimic"; COCONUT is a baseline).
- **No interventions anywhere** (main + appendix): control over parallelism is via
  sampling policies (MTS: average K sampled token embeddings; Dirichlet) and supervision
  budget B — generation-time construction, never editing an existing latent state.
  Closest instrument: C.1 entropy probe (token-level entropies match ln 2^t uniform
  superposition) — correlational, on constructed thoughts.
- **Capacity overlap (matters for our EXPLORATORY tier):** D.1 packing bound — robust
  decoding of budget-B superpositions needs d = Ω(B log(v/B)) — plus Fig. 2c empirical
  budget-vs-dimension sweep (d ∈ {16,24,32}, B ∈ {1,4,8,16}) showing the threshold.
  Our capacity curve must now be framed as: interference in EMERGENT thoughts under
  causal editing, confronted with TWO published functional forms (this packing bound;
  2602.01148's error dynamics). Cite both; the curve is corroboration, not a novelty
  pillar (it was already non-gating).
- Context numbers: their ProsQA (structured-token variant, 4-layer d=32) 93.37 SFT →
  94.21 GRPO; COCONUT baseline 90.03.
- New adjacent pointers surfaced, NOT yet verified: Soft Thinking (2505.15778,
  training-free softmax-weighted concept tokens); Xiong et al. 2025 (superpose multiple
  candidate outputs into one token); Zhang et al. 2505.07961 ("intervention" in title —
  likely training-time efficiency intervention, same group; CHECK before any novelty
  claim about interventions).
- Meta: this group shipped v1→v3 in ~9 months with substantial additions. Combined with
  the SPAR cohort, reinforces the speed-matters posture. Substrate S1 is training as of
  tonight; kill-test next.

## 2026-07-07 (late) — external brief received; NEW UNVERIFIED lit debts

A planning brief from a parallel Claude session (user-supplied, brief.txt) names two
papers absent from this scan. NEITHER verified yet — house rule applies before any
novelty claim or citation:

- **arXiv:2606.01243 "Unlocking the Black Box of Latent Reasoning"** — described as
  training-free steering of cached latent vectors in COCONUT/CODI toward a "reasoning
  manifold." If accurate, this is our NEAREST NEIGHBOR (closer than 2602.08783) and
  the delta must be sharp: they steer whole thoughts toward a global quality manifold;
  we decompose into branches and edit search topology with graph-derived causal
  predictions. TOP-PRIORITY VERIFY (fetch + read before any writeup claim).
- arXiv:2604.04902 "Are Latent Reasoning Models Easily Interpretable?" — skim tier.

Brief's technical points adopted/parked (see LOG 2026-07-07 late): ridge decomposition
(empirical check: trained node embeddings mean |cos| 0.063, max 0.50 — bleed real but
bounded; Gate A-ii catches it behaviorally), counterfactual patching / transplant
(H5 "carrier vs cache") as post-verdict exploratory arms, per-step re-application
control (matches frozen threat (e)), wrong-step + dose-response controls.

## 2026-07-08 — arXiv:2606.01243 VERIFIED (full-text HTML v1, tables 1/2/4/5, figs 2–3,
related-work + limitations read); top-priority debt PAID. Delta is sharp.

- **arXiv:2606.01243 "Unlocking the Black Box of Latent Reasoning: An
  Interpretability-Guided Approach to Intervention"** (Chang, Bai, Zhang, Ma, Liu,
  Liao, Miao, Niu; v1 2026-05-31). Verified against the actual HTML full text
  2026-07-08, NOT abstract-only. The brief's description was directionally right but
  overstated the proximity: there is NO "reasoning manifold" formalism and no
  steering toward one.
- What they actually do: COCONUT + CODI paradigms on Qwen3-8B / Llama-3.1-8B /
  Llama-3.2-3B; GSM8K (+GSM-Hard, SVAMP OOD) and StrategyQA. Probes: CKA vs explicit
  CoT states, linear recoverability mapper (cos 0.75), lexical probe (top-1 0.38).
  Causal probes: drop@k truncation, whole-vector slerp toward an answer-supporting
  exemplar (flip 21.5%), single norm-constrained gradient step (flip 24.1%),
  whole-sequence transplant (directional transfer 48.3%) — Table 2. Interventions:
  training-free decode-time whole-thought edits (mapper transport + slerp + gradient
  + embedding-subspace / weight-tying projections + energy descent), gains +0.4–1.8
  pts (Tables 4–5).
- Delta from METIS, now verifiable claim-by-claim: (1) they never decompose a thought
  into parallel branch components — latent vectors are treated as compressed encodings
  of a SINGLE trajectory; "superposition" appears only as the 2505.12514 citation in
  related work, unelaborated; (2) no graph tasks, no search-topology predictions, no
  ProsQA; (3) all edits are whole-vector; per-branch arithmetic (subtract / collapse /
  inject) absent; (4) no capacity question; (5) zero negative results — no discussion
  of intervention failure or latent non-linearity (limitations = compute overhead
  only). Our kill finding (readable + load-bearing, yet per-branch-linearly INERT) is
  complementary, not scooped: their whole-vector edits working at 3–8B scale vs our
  per-branch edits failing on the cleanest substrate is a contrast the writeup can
  use, not a collision.
- Independent corroboration of DECISIONS #005: their edits are explicitly
  norm-preserving (Eq. 13 slerp rescaled to ‖z₂‖; Eq. 14 gradient step with strict
  norm constraint). Nobody edits latents at unit norm; the #005 correction matches
  field practice.
- Citation obligations for the writeup: cite as nearest intervention-on-latents
  neighbor; acknowledge their transplant/slerp probes as precedent for our
  post-verdict exploratory transplant arm (theirs = whole-sequence across instances;
  ours = carrier-vs-cache on graphs).
- Still open (skim tier, unverified): arXiv:2604.04902.

## 2026-07-08 (PM) — Anthropic "global workspace" / J-lens (published 2026-07-06,
user-flagged); primary research page read; DIRECTLY relevant new instrument

- **Anthropic, "A Global Workspace in Language Models"** (anthropic.com/research/
  global-workspace, 2026-07-06; open-source tooling github.com/anthropics/
  jacobian-lens; third-party replications on Qwen-1.5B/7B reported in secondary
  coverage). Verified at the level of Anthropic's own research page 2026-07-08;
  the underlying transformer-circuits paper + repo are the REMAINING deep-verify
  debt before any writeup claim.
- UPDATE (same day, later): the actual paper is "Verbalizable Representations Form
  a Global Workspace in Language Models" (Gurnee et al.,
  transformer-circuits.pub/2026/workspace/index.html). METHOD-level verification
  DONE (equations extracted from the paper itself): J_l = E_[t,t'>=t,prompt]
  [dh_final,t'/dh_l,t], J-lens vectors = rows of W_U J_l, aggregation = plain mean
  over ~1k prompts + positions; interventions = steering h+av, ablation = project
  out component (== our frozen SUBTRACT), and a concept-SWAP patch
  h + V(sigma(c)-c), V=[v_s v_t], c=V^dag h. Stated limits: single-token concepts
  only; middle-band structure needs depth (2-layer S1 may lack it — but S1's
  intervention locus is the recycled thought, the model's only inter-step channel,
  so the adaptation differentiates through THAT, not through a layer band).
  Full-paper close read + repo audit still owed before citing their FINDINGS
  (J-space/GWT claims) in any writeup; method use is covered. Their scope remains
  token-by-token models; latent-CoT application is ours (DECISIONS #009).
- Method (J-lens): for each vocabulary token, find the activation direction that
  most increases P(model says that token at some future point) — directions
  defined by CAUSAL INFLUENCE (Jacobians), not decodability. Causal swaps of
  J-space patterns swap downstream behavior (soccer→rugby; spider→ant flips a
  math answer 8→6); ablating J-space kills multi-step reasoning, preserves
  fluency. Stated caveats: single-token concepts only; approximate.
- Relevance to METIS: our kill showed decodability-optimal (probes) and
  geometric (wte) directions are causally inert — the exact failure mode J-lens
  was designed around. Their scope is token-by-token models ONLY (no latent CoT;
  checked) — the niche stays open, and our negative independently corroborates
  their motivation on a toy substrate. A J-lens-style basis for S1 node tokens
  (d answer-logit / d thought, exact gradients on a 15M model; nodes are single
  tokens so their main caveat doesn't bite) is the strongest remaining candidate
  basis. VERDICT-NEUTRAL: the frozen ladder ended at probes; a J-lens basis can
  run only as a post-verdict exploratory arm (new DECISIONS entry, recipe pinned
  pre-fit) or as a new preregistration (METIS-2) — it cannot retroactively flip
  the KILL.

## 2026-07-08 (PM, later) — verify-before-cite debts from external review triage

- **Amnesic probing / INLP — Ravfogel et al. ("Null It Out"), Elazar et al.
  ("Amnesic Probing")**: UNVERIFIED. Must be read from actual papers before the
  writeup cites the readable≠causal lineage, AND before implementing the
  registered INLP-style subspace-removal arm (PREREGISTRATION_M2 §4) — the arm's
  recipe should follow their actual iterative-nullspace procedure, not memory.
- **Belinkov probing survey; Hewitt & Liang selectivity**: UNVERIFIED, skim tier;
  related-work placement for the dissociation claim.
- Reminder of standing debts: Gurnee et al. 2026 full close read + repo audit
  (before citing their FINDINGS; method-level use already covered);
  arXiv:2604.04902 skim.

## 2026-07-09 — INLP + amnesic probing VERIFIED from ar5iv full text (debt PAID)

Both read at method-section level from the actual papers (ar5iv HTML), per house rule,
before implementing the registered INLP arm (PREREGISTRATION_M2 §4). The 2026-07-08
"UNVERIFIED" note above is now superseded.

- **Ravfogel et al., "Null It Out: Guarding Protected Attributes by INLP"
  (arXiv:2004.07667, ACL 2020) — Algorithm 1 verified:** for i = 1..n: (1) train a
  linear classifier Wᵢ on the current (projected) X to predict the attribute Z;
  (2) compute the nullspace projection P_{N(Wᵢ)}; (3) X ← P_{N(Wᵢ)} X, P ← P_{N(Wᵢ)} P.
  Stop when no classifier beats majority-class accuracy. Geometric content: zeroing X
  along Wᵢ's ROWSPACE removes the info Wᵢ used. Numerically-stable form they recommend:
  don't multiply projection matrices — collect the classifier directions and project
  out their combined rowspace (intersection of nullspaces = N(ΣᵢP_R(wᵢ))); equivalently
  P = I − BBᵀ for B = orthonormal basis of the collected directions. Each iteration
  removes ~rank-1 for a binary attribute; after n iters rank(g(X)) ≥ r − n. Classifiers:
  L2-regularized SVM / logistic regression; no centering noted.
- **Elazar et al., "Amnesic Probing" (arXiv:2006.00995, TACL 2021) — protocol
  verified:** use INLP to remove property Z from representations H, then feed the
  PROJECTED representation through the ORIGINAL downstream head and measure the
  behavioral change (LM accuracy; KL between pre/post token distributions) — the causal
  analogue of our ΔT. **Load-bearing control = "Rand": remove the SAME NUMBER of RANDOM
  directions** (build the projection from random vectors' nullspace intersection); if
  Rand's behavioral impact ≈ INLP's, the property is not causally important — the effect
  is just rank loss. Selectivity control (re-inject a small gold feature vector and
  finetune to restore) exists but is out of scope for our non-gating arm.
- **Mapping to METIS INLP arm (PREREGISTRATION_M2 §4):** our #006 probe rung removed
  ONE direction/node (the logistic "node-v-in-depth-1-frontier" direction) and found
  ΔT≈0 (readable-but-inert) — the KILL. The amnesic objection: one direction can
  under-delete. The arm iterates the #006 probe recipe per node (INLP) to remove the
  full linearly-decodable identity subspace, re-measures the answer-branch SUBTRACT ΔT
  under the frozen M2/kill norm-preserving protocol, and includes Elazar's rank-matched
  random-subspace control. Null survives iff full-subspace ΔT is still ≈0; a large
  INLP ΔT with small Rand ΔT would give the objection teeth; both large = rank-loss
  artifact. Non-gating (cannot flip the frozen Gate-A verdict). Recipe pinned in
  DECISIONS #014 before any measurement.
- Caveat: read via ar5iv→markdown + a summariser; the two facts our code depends on
  (the I−BBᵀ rowspace-projection form, and the rank-matched Rand control) are the
  canonical, widely-reproduced core of both papers — high confidence.

## Adjacent context

- SAE downstream-utility critique (2025): probes beat SAEs when labels exist → ProsQA has
  labels → probes-first, dictionaries only for unsupervised phase-2 questions.
- Recurrent-depth models (Huginn-3.5B): parallel substrate bet; orbits observed in latent
  trajectories; transfer target if S1/S2 results are clean.
