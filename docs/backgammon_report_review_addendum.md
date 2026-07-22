# Archived Review & Addendum: SOTA AlphaZero-Style Backgammon Report

> **Background only.** This review predates the current BackgammonNet contract
> and the discovery of GNUBG label/equity problems. Its literature discussion
> remains useful, but its engine calibration numbers and project recommendations
> are not validated project status. Use [`backgammon_status.md`](backgammon_status.md)
> for current decisions.

**Verdict:** The report is technically strong and the core recommendation — explicit-chance AlphaZero with afterstates, exact dice, and Gumbel root search — is the right call. The tree structure (decision/afterstate/chance), the "never max over dice" rule, duplicate-afterstate collapse, candidate-afterstate scoring, and paired-dice evaluation are all correct and well-prioritized. The stratified chance-selection formula argmax P(r)/(N\_r+1) even matches what DeepMind's mctx library implements internally for Stochastic MuZero.

What follows: (1) corrections and precision fixes, (2) one big strategic recalibration the report underweights, (3) missing high-value techniques, (4) ecosystem/code the report doesn't mention, and (5) suggested edits to the build target.

---

## 1\. Corrections and precision fixes

### 1.1 Stochastic MuZero numbers (report §1.2, §11.1)

The report says Stochastic MuZero "with 1600 simulations per move reached GNUbg-level play." More precisely, per the paper and the first author's thesis:

- Stochastic MuZero was trained with **400 simulations** per move and evaluated at **1600**.  
- The deterministic MuZero backgammon baseline used 200 (train) / 800 (eval) and plateaued below.  
- Stochastic MuZero **outperformed GNUbg Grandmaster**, and kept improving as the simulation budget grew.  
- The instructive comparison: GNUbg Grandmaster is a 3-ply expectimax over \~20 moves × 21 rolls, searching **millions of positions per move**, while Stochastic MuZero used \~1600 NN evaluations. The lesson is not "you need 1600 sims" — it's that learned policy/value \+ selective search is \~1000× more search-efficient than brute expectimax. This strengthens the report's low-simulation Gumbel recommendation.  
- Backgammon experiments in the paper used **sample-based search (Sampled MuZero)**, i.e., candidate sampling, consistent with the report's §7.2 design.

Update the §11.1 table row accordingly ("400 sims training / 1600 eval; outperformed GNUbg Grandmaster").

### 1.2 Policy target under Gumbel search — internal contradiction (§2.5 vs §9.2)

§9.2 trains the policy on visit counts N^(1/τ). That is the vanilla AlphaZero target and it is **wrong under Gumbel/Sequential Halving at low budgets**, which §2.5 recommends. Sequential Halving deliberately concentrates visits on surviving candidates, so visit counts are a biased, high-variance target at 16–64 sims. The Gumbel paper's contribution is precisely a better target:

pi\_target \= softmax(policy\_logits \+ sigma(completedQ))

i.e., the "completed Q-values" improved policy over all actions (searched and unsearched), not visit counts. Similarly, final move selection under Gumbel is the Sequential Halving winner (argmax of Gumbel-perturbed logit \+ σ(Q)), not argmax N.

**Fix:** make §9.2 conditional — visit-count targets for pUCT-with-noise mode; completed-Q targets for Gumbel mode. LightZero's Gumbel implementations do this (they note the completed-Q weighting is a sensitive hyperparameter).

### 1.3 Dirichlet noise \+ Gumbel (§4.4)

The report says root Dirichlet noise is "usually unnecessary" with Gumbel. Slightly stronger: it should be **omitted**. Gumbel's exploration comes from the Gumbel noise itself, and adding Dirichlet breaks the policy-improvement guarantee that motivates using Gumbel in the first place. Keep the Dirichlet guidance only for the vanilla-pUCT configuration.

### 1.4 Opening roll is not the same distribution (§3.1, §15.2)

The 21-outcome / doubles-at-1/36 distribution is correct for every roll **except the opening roll**. In standard rules, each player casts one die and doubles are re-rolled, so the first chance event is uniform over the **15 non-double** combinations (1/15 each) and simultaneously determines who moves first. Add this to the §15.2 move-generator test list — it's a classic simulator bug, and it slightly changes opening-book statistics if ignored.

### 1.5 Value targets: use the stored root value (§9.1, §9.3)

§9.1 stores the MCTS root value but §9.3/§9.4 never use it. Modern practice (MuZero, KataGo, LightZero defaults) mixes the game outcome z with the search value to reduce target variance:

value\_target \= beta \* z \+ (1 \- beta) \* search\_value      (beta \~ 0.5–1.0, or n-step TD)

This matters more in backgammon than in Go: outcomes are extremely noisy (dice), so pure-z targets are high-variance. Bootstrapped/mixed targets are one of the cheapest variance reductions available. (Complementary option below in §3.1: exact bearoff values as targets near the end.)

### 1.6 Forced moves (§4.2, §9.2)

Backgammon has many positions with exactly one legal play (or a forced dance from the bar). Add explicitly:

- **Skip search entirely** at forced positions (zero simulations).  
- **Exclude forced positions from the policy loss** (they carry no policy information; still usable for value).  
- Exclude them from error-rate metrics as well — GNUbg/XG error rates count unforced decisions only, so this keeps your metrics comparable.

### 1.7 Candidate-scorer inference cost (§11.2)

The NN-eval formula omits that a candidate-afterstate scorer must encode **all \~20 legal afterstates at every decision**, even at simulation count 0\. Effective cost per decision is roughly (sims \+ num\_candidates) afterstate evaluations, not sims. At 16 sims this nearly doubles the estimate. Worth fixing so hardware plans stay honest — and it is another argument for a small, fast trunk.

### 1.8 Minor: TD-Gammon version table (§11.1)

The 300k-game/1-ply system with raw board input was TD-Gammon **0.0**; TD-Gammon 1.0 added Neurogammon's hand-crafted features at similar game counts. Doesn't change the lesson, but worth correcting if the table is kept.

---

## 2\. The big strategic recalibration: in backgammon, the value function is the product; search is a multiplier

The report is written from an AlphaZero-centric worldview where search depth is the engine of strength. Backgammon's empirical record says something different, and it should reshape the reduced-hardware plans:

- **GNUbg 0-ply** — three tiny MLPs (contact/crashed/race, \~250 engineered inputs, hundreds of hidden units, 1990s-style sigmoid nets trained supervised on rollouts) — plays at strong expert level *with no search at all*. GNUbg's higher plies add only 2–3 plies of expectimax.  
- **XG2**, the world reference standard, is likewise a strong net \+ 3–4-ply expectimax \+ rollouts; tournament Performance Ratings are calibrated against XG2 rollouts.  
- **Strehl's backgammon-ai-engine (2024–25, open source)**: a **562k-parameter** network trained purely by self-play TD with exact 1-ply Bellman backups — no MCTS — beats gnubg 0-ply in DMP and cubeful money play and achieves **PR ≈ 1.06** measured by XG++ analysis. For scale: elite human pros are typically PR \~2–5, and PR 0 is defined by XG rollout. That is essentially superhuman play from a half-million-parameter model with no tree search.  
- Chance nodes damp the value of depth: every extra ply multiplies branching by \~21 rolls and averages over them, so deep tactical lines wash out. This is why no top backgammon engine searches deep, and why Stochastic MuZero's 1600 sims should be read as a research ceiling, not a requirement.

**Directional consequences:**

1. **Build the TD/1-ply baseline first, not just the "CPU prototype."** A Strehl-style pipeline (afterstate value net, exact 1-ply expectimax backup over 21 rolls, self-play) is dramatically cheaper than MCTS self-play and is already near-SOTA for checker play. It also reuses your exact simulator, afterstate encoding, and evaluation harness — nothing is thrown away when you add the Gumbel MCTS layer on top.  
2. **Reframe the payoff of MCTS/AlphaZero machinery**: it buys (a) a principled route to cube and match play inside one framework, (b) research comparability with Stochastic MuZero, and (c) the last few tenths of PR beyond what a great value net gives. It is not the cheapest route to "strong."  
3. **The single-consumer-GPU plan (§12.2) is too pessimistic.** Given the Strehl data point, one GPU should be expected to reach superhuman *checker* play (PR \~1–2) via value-centric training; the open question at that hardware tier is cube/match refinement, not checker strength.  
4. **A separate policy head may be optional early.** TD-Gammon, GNUbg, and wildbg have no policy network — ranking afterstates by value *is* the policy. For Gumbel priors you can use softmax(V(afterstates)/T). Add a learned policy scorer later only if profiling shows candidate evaluation is the bottleneck. This deletes a whole subsystem from the v1 build.

---

## 3\. Missing high-value techniques

### 3.1 Exact bearoff databases (biggest omission)

The report mentions "bearoff database value" only as an auxiliary head. Exact endgame databases deserve first-class status, the way tablebases do in chess:

- GNUbg ships a **one-sided bearoff DB (15 checkers on the last 6 points)** and an in-memory **two-sided DB**, with larger ones downloadable/generatable. The two-sided DB gives *exact* win probabilities and cubeful money equities.  
- GNUbg **truncates rollouts into the two-sided database "with no error at all"** — the same trick applies to self-play: terminate games on DB entry and use the exact value as the target. This (a) removes the noisiest final segment of every game, (b) shortens games → more games/hour, (c) provides perfect endgame play for free, and (d) fixes the classic NN weakness in non-contact bear-ins.  
- Early-training bonus: random self-play games can run hundreds of moves; DB truncation plus a max-length cap keeps throughput sane before the net is competent.

Recommended additions to the blueprint: bearoff\_db.lookup(board) \-\> exact distribution/equity, game truncation on DB entry, and DB values as supervised targets for in-DB positions.

### 3.2 KataGo's self-play efficiency techniques (missing reference)

KataGo (David Wu, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565) is the definitive "AlphaZero on a budget" playbook and is absent from the report. Its reported efficiency gains over AlphaZero-style baselines were on the order of 30–50×. Directly transferable pieces:

- **Playout cap randomization**: most self-play moves get a tiny search (cheap, good value data), a random \~25% get the full budget (good policy data). This decouples the value-data/policy-data cost tradeoff and is trivially compatible with Gumbel roots.  
- **Auxiliary targets** to densify the learning signal — the report's §8.7 list is good; KataGo is the evidence it works and the reference for loss weighting.  
- **Forced playouts \+ policy target pruning** (for the pUCT configuration).  
- Its general discipline of measuring *strength per self-play FLOP* matches the report's §15.4 advice.

### 3.3 Luck-adjusted evaluation and error-rate metrics (§13 upgrade)

Paired dice is correct but incomplete:

- **Adopt error rate / PR as the primary progress metric**, not win rate. Analyze your agent's games with GNUbg (scriptable via its Python interface) or XG and track average equity loss per unforced decision. This is how the whole backgammon community measures strength (wildbg reports "error rate \~5.9 vs gnubg 2-ply"; Strehl reports PR 1.06), it has far lower variance than game results, and it gives you free phase breakdowns (checker vs cube errors) matching §13.3.  
- **Luck adjustment / variance reduction**: GNUbg's rollout VR uses the ply-to-ply equity difference of each dice roll as a control variate ("luck"). The same luck-adjusted scoring can be applied to head-to-head evaluation matches on top of paired dice, further shrinking confidence intervals.  
- Also note GNUbg's observation that truncated rollouts estimate *relative* equities (move A vs move B) better than absolute equities — correlated errors cancel. This is the same phenomenon as the report's §8.6 and justifies truncated-rollout targets for reanalyse.

### 3.4 Analytic cube bootstrap: Janowski \+ Match Equity Tables (§10.2 Stage 3–4)

The report trains cube heads from scratch. There is a well-established shortcut the whole field uses:

- **Janowski's cubeful equity formulas** convert a cubeless outcome distribution into cubeful equities via a cube-efficiency parameter x (GNUbg uses x ≈ 0.6–0.68, and derives x from exact bearoff equities where available).  
- For match play, standard **Match Equity Tables** (GNUbg ships several, including Rockwell-Kazaross) map game-outcome distributions to match equity at any score.

Reduced-hardware path: train only the cubeless distributional head, get near-strong cube/match decisions *analytically* from Janowski \+ MET, then optionally learn residual corrections. This also gives you a sanity baseline for any learned cube head — if end-to-end cube RL (which Strehl showed works for money play) can't beat Janowski-on-your-own-distribution, the cube head isn't earning its complexity.

### 3.5 Phase-specialized networks

GNUbg's strength rests partly on splitting into **contact / crashed / race** nets with a hard classifier. A single net with a phase auxiliary head (report §8.7) is the modern equivalent, but if race-phase accuracy lags, a dedicated race net (or bearoff DB reliance) is the proven fix. Cheap to A/B.

---

## 4\. Ecosystem: code, data, and baselines the report should cite

**Frameworks**

- **Pgx** (arXiv:2303.17503) — JAX-vectorized game simulators **including backgammon**, designed to pair with mctx; thousands of parallel games on one GPU. For a JAX pipeline this likely replaces a hand-rolled Rust simulator for training (keep a native one for correctness cross-checks and deployment). [https://github.com/sotetsuk/pgx](https://github.com/sotetsuk/pgx)  
- **mctx** (DeepMind) — JAX MCTS with gumbel\_muzero\_policy and stochastic\_muzero\_policy. **Caveat: it does not ship the combination** (Gumbel root \+ chance nodes) — that's a known open request (issue \#66). Your "Explicit-Chance Gumbel AlphaZero" requires writing that fusion yourself or using LightZero. [https://github.com/google-deepmind/mctx](https://github.com/google-deepmind/mctx)  
- **LightZero** (already cited in the report) — has both Gumbel and Stochastic MuZero configs; note their finding that the completed-Q balance weight materially affects Gumbel results.  
- **OpenSpiel** (DeepMind) — includes a backgammon environment with first-class chance nodes in C++; useful as a reference implementation and cross-check for the move generator.

**Backgammon engines & pipelines**

- **wildbg** (Carsten Wenderdel) — open-source Rust engine, PyTorch-trained / ONNX-deployed nets, fully public training pipeline (wildbg-training); \~5.9 error rate vs gnubg 2-ply as of early 2024\. The closest existing project to the report's Rust recommendation; its input-feature docs and rollout-data generation are directly reusable. [https://github.com/carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg)  
- **alexstrehl/backgammon-ai-engine** — self-play TD \+ exact 1-ply Bellman backups, GPU-accelerated, beats gnubg 0-ply in DMP (51.84% over 10M games) and cubeful money (+57.8 mEq/game), **PR 1.06**, and learns money-cube actions purely via RL. The single most important calibration point for the reduced-hardware sections. [https://github.com/alexstrehl/backgammon-ai-engine](https://github.com/alexstrehl/backgammon-ai-engine)  
- **bgsage** — open NN engine with documented multi-ply search, cubeful evaluation, and truncated/VR rollouts; good documentation of the classical machinery. [https://github.com/markbgsage/bgsage](https://github.com/markbgsage/bgsage)  
- **GNUbg tooling** — Python scripting and an external-player socket interface make automated benchmark matches and bulk analysis scriptable; bearoff/hypergammon databases downloadable from gnubg.org. Joseph Heled's GNUbg training-program notes document the original supervised-on-rollouts pipeline.

**Benchmark ladder (concrete "SOTA" definition the report lacks)**

random-legal  →  gnubg 0-ply  →  gnubg 2-ply  →  GNUbg Grandmaster (3-ply)

→  XG2 3-4 ply  →  XG2 rollout (PR 0 reference)

human calibration: world-class pros ≈ PR 2–5

existing open-source SOTA: PR ≈ 1 (Strehl), error ≈ 5.9 vs 2-ply (wildbg)

---

## 5\. Suggested edits to the final build target

Keep the architecture; amend the plan:

1. **Phase 0 (new): value-first baseline.** Exact simulator \+ bearoff DB \+ afterstate value net \+ 1-ply exact-expectation backup, trained by self-play TD (Strehl-style) or supervised on gnubg rollout data. Expected outcome on one GPU: PR \~1–2 checker play. This becomes the warm-start network and the strength floor for everything after.  
2. **Add bearoff DB integration** to Search, Trainer, and Env modules; truncate self-play/rollouts into it.  
3. **Fold in KataGo tricks**: playout cap randomization, mixed z/search-value targets, auxiliary heads from day one.  
4. **Gumbel mode uses completed-Q policy targets and SH move selection**, no Dirichlet; visit-count targets only in the pUCT configuration.  
5. **Policy head optional in v1** — value-softmax priors over candidate afterstates; add a learned scorer only if profiling demands it.  
6. **Cube via Janowski/MET bootstrap first**, learned cube heads as a Stage-3+ refinement with the analytic method as the baseline to beat.  
7. **Primary metric: PR / average equity error per unforced move** (via scripted GNUbg/XG analysis), with paired-dice \+ luck-adjusted match results as secondary confirmation.  
8. **Tooling decision point:** JAX route \= Pgx \+ mctx (writing the Gumbel×stochastic fusion yourself) vs PyTorch route \= LightZero configs vs native route \= Rust engine à la wildbg with ONNX nets. All three are viable; the JAX route maximizes self-play throughput per GPU, the native route maximizes deployment speed.

---

## 6\. Additional references to append to §16

17. **David J. Wu. *Accelerating Self-Play Learning in Go.* arXiv:1902.10565, 2019\.** Playout cap randomization, policy target pruning, auxiliary targets; the standard reference for AlphaZero-style training on small budgets. [https://arxiv.org/abs/1902.10565](https://arxiv.org/abs/1902.10565)  
18. **Sotetsu Koyamada et al. *Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning.* NeurIPS 2023\.** JAX-vectorized backgammon environment. [https://arxiv.org/abs/2303.17503](https://arxiv.org/abs/2303.17503)  
19. **DeepMind mctx.** JAX implementations of Gumbel MuZero and Stochastic MuZero search. [https://github.com/google-deepmind/mctx](https://github.com/google-deepmind/mctx)  
20. **Carsten Wenderdel. *wildbg.*** Open Rust/ONNX backgammon engine with public training pipeline. [https://github.com/carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg)  
21. **Alexander Strehl. *backgammon-ai-engine.*** Self-play TD with exact 1-ply backups; PR ≈ 1.06; RL-learned money cube. [https://github.com/alexstrehl/backgammon-ai-engine](https://github.com/alexstrehl/backgammon-ai-engine)  
22. **Rick Janowski. *Take-Points in Money Games / cubeful equity formulas.* 1993\.** Basis of GNUbg's cubeful equity calculation (cube-efficiency parameter x). See GNUbg manual, "Cubeful equities."  
23. **GNU Backgammon manual & bearoff database documentation.** Bearoff DBs, rollout truncation, variance reduction, METs, Python scripting. [https://www.gnu.org/software/gnubg/manual/gnubg.html](https://www.gnu.org/software/gnubg/manual/gnubg.html)  
24. **OpenSpiel (DeepMind).** Reference backgammon environment with explicit chance nodes. [https://github.com/google-deepmind/open\_spiel](https://github.com/google-deepmind/open_spiel)
