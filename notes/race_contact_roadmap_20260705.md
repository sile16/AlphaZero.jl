# Race→Contact Roadmap (living checklist, started 2026-07-05)

Strategy: memory `strategy-race-contact`. Goal: finalize race model, then break the
contact plateau via exact-race-frontier curriculum. Bar: 99.9999% correct race + eval;
one interface per engine; evals valid/strong/fast/frequent.

## GOAL (2026-07-05): contact model that BEATS wildbg, SOTA techniques. Re-test
## chance-node MCTS — prior attempts likely confounded by now-fixed training-stack bugs.

## Phase 0 — Correctness + consolidation
- [x] MAP duplicate game loops + chance samplers + table↔NN handoff (agent) — KEY findings:
      handoff CORRECT at all live sites (no live bugs); combined/one-sided table wired
      into NOTHING but eval_table_vs_wildbg (must wire into training MCTS); footguns fixed.
- [x] Footgun fixes (15ba328): bearoff_table_equity self-guard, eval_race/eval_vs_wildbg
      3.0→reward_scale, removed dead _sample_chance.
- [x] Housekeeping: deleted 225GB Neo work_* intermediates; n18 table copying Neo→Jarvis
      (background, ~slow; NOT on critical path — experiment runs on Neo which has the table).
- [x] WIRE COMBINED TABLE INTO MCTS (ad6eb72): make_bearoff_evaluator +
      bearoff_table_equity + bearoff_post_dice_equity use generic bearoff_covers/bearoff_lookup;
      selfplay_client loads CombinedBearoff(k7, n18 onesided) when present else k7-only.
      EXP2 log confirmed Jarvis loaded k7 + n18 and used the combined exact evaluator.
      NOTE: this is MCTS leaf evaluation. EXP2 had bearoff_truncation=false, so it did
      NOT stop rollouts/training traces at the race frontier.
- [x] Fix distributed eval submit/finalization before next run: src/distributed/server.jl
      final-chunk path uses stats.n_games, but EvalManager.finalize_eval returns
      num_games. Symptom: final eval submit returns HTTP 400 then retry 409, although
      the training loop later finalizes most jobs. Also make training_server wait for
      the final iter eval before exit, or run final eval out-of-band. DONE: patched
      stats.num_games, added handler regression, factored final eval logging, and
      added bounded final-eval wait.
- [x] Operational launch note: from the Codex exec environment, plain `nohup ... &`
      children were reaped after the shell returned. Use `setsid bash -c 'exec julia ...'`
      for detached Jarvis training/eval processes; verified with EXP3.
- [ ] Stop orphan self-play client wrapper after bounded runs, or teach start_client.sh /
      selfplay_client to exit cleanly when the server is intentionally complete.
      Partial: added selfplay_client --eval-only so bootstrap/eval runs can avoid
      self-play upload contamination.
- [ ] Game-loop consolidation: fold eval_game_from_position (#4, the outlier) into play_game;
      dedup PositionValueSample + _sample_from_policy (3 copies). Defer full self-play
      unification (zero-alloc constraint, CLAUDE.md lesson 13).
- [ ] Consolidate to ONE `play_game` used by selfplay + all eval paths (supports: trace
      capture, value-sample capture, bearoff/race-frontier truncation, temperature,
      chance mode, perspective). Migrate callers one at a time, test each.
- [ ] ONE chance-sampler, called from the one loop.
- [ ] AUDIT table↔NN handoff: bearoff_covers gates exactly; uncovered race (>18pip) +
      contact fall to NN (never silent wrong/out-of-domain table value); scale +
      perspective consistent end-to-end. Add regression tests for the coverage boundary.

## Phase 1 — Evals: valid, strong, fast, frequent
- [ ] Race eval: correct positions/scoring, enough games+MCTS, parallel, every N iters.
- [ ] Coverage-boundary eval: race positions just below/above the 18-pip table edge —
      confirm NN accuracy on the UNCOVERED band (this is what "finalize race" needs).
- [ ] Contact eval: DIRECT MATCH vs wildbg = primary metric; rollout/exact diagnostics
      for weakness-finding only (wildbg-capped).
- [ ] Add direct post-run eval script/checklist for final checkpoints. EXP2 created
      iter-200 eval, then exited before clients could run it; best measured point is
      iter 195, not iter 200.

## Phase 2 — Finalize race
- [ ] Definitive race eval across candidate checkpoints (current fixed code) → pick best.
- [ ] Confirm exact-table absorption (bearoff corr vs table ≥ 0.99; ship <50MB table).
- [ ] Freeze race_best.data as the frozen race evaluator.

## EXP1 LAUNCHED (2026-07-05 ~15:15): dual 128×3 (283k params), 200 iters, cold-start,
## eval-vs-wildbg every 10 iters (contact_eval_2000.jls, 200 games). Server PID on Jarvis
## (data-dir /home/sile/alphazero-contact-exp1, port 9090); eval-capable client on
## Jarvis-localhost (Neo blocked by Julia-LAN networking). Combined-table curriculum ACTIVE
## as MCTS exact leaf evaluator (not trace truncation unless --bearoff-truncation is set).
## Signal = match win% vs wildbg (goal >50%).
## Logs: exp1_server.log / exp1_client.log. Validated live: combined table loads, contact
## opening, self-play ~2 games/sec, eval chunks assigned. NOTE cold-start won't reach parity
## fast — watching trajectory; if it plateaus below wildbg, next lever = bootstrap-to-parity
## then chance-node A/B (passthrough vs progressive-widening).

## BOOTSTRAP LEVER BUILT (2026-07-05 ~15:45): scripts/generate_contact_bootstrap.jl —
## wildbg-vs-wildbg contact imitation samples with SOFT policy target from wildbg
## per-move eval + outcome value/equity, matches convert_trace_to_samples.
## Actual generated file: 100k samples, 271MB, 4505 wildbg games →
## /homeshare/projects/AlphaZero.jl/eval_data/contact_bootstrap_wildbg.jls.

## EXP2 COMPLETED (2026-07-05 15:52→23:02): dual 128×3, 200 iters, 100k soft
## contact bootstrap, 15 bootstrap-train iters, then self-play. Data-dir:
## /home/sile/alphazero-contact-exp2. Logs: exp2_server.log / exp2_client.log.
## Config facts: PER=false, Reanalyze=false, constant 400 selfplay MCTS, eval 200
## MCTS, eval every 5 iters, 400 games/eval, combined k7+n18 exact MCTS leaf
## evaluator loaded; bearoff_truncation=false; fixed bearoff eval/gate disabled.
## Result: learned, but NOT near parity. Eval trajectory:
##   iter 0   -1.752 equity,  8.8% win
##   iter 5   -0.980 equity, 20.8% win
##   iter 10  -1.122 equity, 19.0% win  (EXP1 cold-start iter10 was -1.418/11.5)
##   iter 50  -1.145 equity, 16.5% win
##   iter 100 -0.912 equity, 22.2% win
##   iter 145 -0.838 equity, 25.2% win
##   iter 180 -0.812 equity, 26.5% win
##   iter 195 -0.798 equity, 24.2% win  (best measured equity)
##   iter 200 checkpoint saved, but final eval was created then server exited; run
##            standalone final eval before ranking iter200.
## Lessons: 100k bootstrap helps cold-start but is far below "bootstrap-to-parity";
## after bootstrap buffer clear at iter16, self-play recovers slowly but plateaus
## around -0.8 to -0.9 equity. This does NOT test PER/Reanalyze, promotion gate, or
## trace truncation. It DOES test combined exact MCTS leaf evaluator + soft bootstrap
## plumbing under load.

## NEXT EXPERIMENT (EXP3 candidate): fix eval first, then bootstrap-to-parity before
## self-play. Proposed rung:
## - [x] Stop orphan Jarvis selfplay client from EXP2 (server exited; client retries).
## - [x] Patch stats.n_games→stats.num_games and add regression for final chunk submit.
## - [x] Ensure final eval completes before training process exits.
## - [ ] Standalone eval contact_iter_200/contact_latest (same 200-pos, 200-MCTS quick
##       eval, then larger eval if promising).
## - [x] Launch first bootstrap-only/pretrain rung from contact_bootstrap_wildbg.jls before
##       self-play: 100k samples, 60 train iters, eval every 5 iters. This isolates
##       supervised bootstrap quality before self-play contamination.
## - [ ] Continue bootstrap-only/pretrain sweep from contact_bootstrap_wildbg.jls before
##       self-play: 100k vs 300k+ samples, 15/30/60 train iters, evaluate each.
##       Gate to EXP3 self-play only when bootstrap checkpoint is much closer to
##       wildbg (target at least ~35-40% win quick eval, not 20%).
##       Use selfplay_client --eval-only so eval chunks run without self-play uploads.
## - [ ] EXP3 self-play: start from best bootstrap checkpoint; enable PER + Reanalyze
##       if eval path is clean; compare bearoff_truncation=false vs true as a controlled
##       A/B only after bootstrap quality is acceptable.
## - [ ] Keep contact net small (128×3/×5) until method breaks the contact plateau.

## EXP3 BOOTSTRAP-ONLY RUNG COMPLETED (2026-07-06 10:55→11:07): dual 128×3,
## 100k soft contact bootstrap, bootstrap-only, 60 iters, no self-play uploads.
## Data-dir: /home/sile/alphazero-contact-exp3-bootstrap100k-60.
## Commit: 92b22b0. Server PID 2484306, eval-only client PID 2485524, both detached
## with setsid. Logs: exp3_bootstrap_server_setsid.log /
## exp3_bootstrap_eval_client_setsid.log.
## Config facts: PER=false, Reanalyze=false, eval 200 positions x 2 sides,
## 200 MCTS, eval every 5 iters, combined k7+n18 exact evaluator loaded by
## eval client; server fixed bearoff eval/gate disabled. Client registered as jarvis;
## Neo was not used. Final eval completed before server exit; orphan eval-only client
## was stopped after it began polling the completed server.
## Eval trajectory:
##   iter 0  -1.7525 equity,  8.8% win
##   iter 5  -0.9350 equity, 24.2% win
##   iter 10 -1.0100 equity, 20.5% win
##   iter 15 -0.9200 equity, 22.5% win
##   iter 20 -0.9425 equity, 23.8% win
##   iter 25 -1.0900 equity, 19.2% win
##   iter 30 -0.9375 equity, 22.5% win
##   iter 35 -1.0350 equity, 20.5% win
##   iter 40 -1.0000 equity, 20.8% win
##   iter 45 -1.0150 equity, 19.2% win
##   iter 50 -1.0350 equity, 18.5% win
##   iter 55 -1.0700 equity, 18.2% win
##   iter 60 -1.1175 equity, 17.0% win
## Result: 60 iters bootstrap-only overfit/regressed after the early iter-5 peak and
## did NOT beat EXP2's ~24-26% plateau. Do not launch EXP3 self-play from iter60.
## Next bootstrap rung should change the data/optimization setup, not just train longer:
## larger/diverse bootstrap set, held-out bootstrap validation, LR schedule/early stop,
## and/or PER/Reanalyze only after the supervised rung can reach the gate target.

## EXP4 LAUNCHED (2026-07-06): DATA-SCALING TEST — 300k vs EXP3's 100k, one clean
## variable. Hypothesis: EXP3's iter-5 peak then monotone decline = OVERFIT/data-starved.
## 300k soft-policy wildbg contact bootstrap, dual 128×3, bootstrap-only, 40 iters,
## eval every 2 iters (resolve the early peak precisely). Everything else IDENTICAL to
## EXP3 (same net, PER/Reanalyze off, constant LR/MCTS, 200-pos×2 eval @ 200 MCTS).
## Decisive read: if peak win% rises (24% → 30%+) the imitation ceiling is DATA-BOUND
## (keep scaling / distill); if it stays ~24% it's ARCHITECTURE/OBJECTIVE-bound (noisy
## outcome value target, net capacity) → redirect to value-distillation / arch, NOT more data.
## Data gen: 20 shards (8 Jarvis seeds 101-108 + 12 Neo seeds 201-212), 15k each, merged
## via scripts/merge_bootstrap_shards.jl → contact_bootstrap_wildbg_300k.jls (811MB).
## Data-dir /home/sile/alphazero-contact-exp4-bootstrap300k. Eval clients on BOTH Jarvis
## (localhost) + Neo (192.168.20.40:9090, eval file copied local).

## EXP4 COMPLETED (2026-07-06 ~11:55→12:00): 300k bootstrap-only, dual 128×3, 40 iters,
## eval every 2. Data-dir /home/sile/alphazero-contact-exp4-bootstrap300k. Both machines
## evaluated (Jarvis localhost + Neo via reverse SSH tunnel — Neo's Julia-LAN EHOSTUNREACH
## gremlin returned post-reboot; nc/curl/ssh reach 9090 but Julia HTTP.jl does not, so
## tunnelled Neo→loopback→Jarvis). Run: ~13s/iter, ~3 min total. contact_loss 5.09→3.92.
## Eval trajectory (win% vs wildbg large, 200pos×2sides=400 games, 200 MCTS):
##   iter 0  -1.7525  8.8%      iter 22 -0.7975 25.8%
##   iter 2  -1.12   18.5%      iter 24 -0.82   25.2%
##   iter 6  -0.9625 23.0%      iter 26 -0.9025 22.5%
##   iter 10 -0.7625 26.0%      iter 28 -0.7525 27.3%
##   iter 12 -0.84   25.2%      iter 30 -0.8175 25.8%
##   iter 14 -0.885  23.0%      iter 32 -0.805  25.0%
##   iter 16 -0.97   22.0%      iter 34 -0.71   27.5%  <- BEST equity
##   iter 20 -0.91   22.0%      iter 36 -0.75   29.2%  <- BEST win%
##                              iter 38 -0.84   22.8%
##                              iter 40 -0.7875 25.8%
##   (iters 4/8/18 skipped — eval job replaced when training briefly outran eval.)
## VERDICT — MIXED, leans METHOD/OBJECTIVE-bound:
## - 3x data FIXED EXP3's overfit collapse: EXP3(100k) peaked iter5 24% then fell to 17%
##   by iter60; EXP4(300k) does NOT collapse — holds -0.7..-0.9 band, best equity -0.71
##   (vs EXP3 ~-0.92), equity mildly IMPROVING late. So EXP3's decline WAS data starvation.
## - BUT the CEILING moved only modestly: peak win% 24% -> 29% (+5pts), still FAR from the
##   ~35-40% gate. 3x data bought ~+5% win = diminishing returns; reaching 40% by data
##   volume alone would need >>1M samples. The residual gap is NOT primarily data-bound.
## - Best measured contact-imitation point to date: iter34 -0.71 eq / iter36 29.2% win.
## NEXT (recommended) EXP5 = VALUE DISTILLATION (objective change, no extra games):
##   regenerate bootstrap with wildbg's per-position EQUITY ESTIMATE as the value/5-head
##   target (low variance) instead of the single-game OUTCOME ±1/±2/±3 (high variance).
##   Hypothesis: the noisy outcome value target caps MCTS-eval play quality; distilling
##   wildbg's equity should lift the ceiling without more data. Same 300k states, 128x3,
##   bootstrap-only, eval every 2 — one clean variable (value target) vs EXP4. If this
##   also plateaus ~29%, the ceiling is capacity/policy-search-bound → curriculum/search.
## Artifacts: contact_bootstrap_wildbg_300k.jls (811MB, kept); 20 shards deleted post-merge.

## EXP5 COMPLETED+KILLED (2026-07-06 ~14:20→14:35): SELF-PLAY CURRICULUM, the actual
## beat-wildbg mechanism test. Warm-start = EXP4 contact iter40 (~26% win), COLD buffer
## (via new --resume weights-only patch d3eab3a: isfile-guard on buffer load). Levers:
## bearoff TRACE TRUNCATION ON (combined k7+n18 exact frontier, confirmed loaded on BOTH
## clients), PER ON, Reanalyze OFF, 400 MCTS, dual 128×3. Both machines self-play (Jarvis
## 4w server-colocated + Neo 16w eval-capable via reverse SSH tunnel). Data-dir
## /home/sile/alphazero-contact-exp5-selfplay. Rationale (Fable consult): EXP2's plateau was
## NOT valid evidence — it lacked truncation, had PER off, AND ran before the 2026-07-03
## value-scale/terminal-gammon fixes (corrupted frontier arithmetic). So the mechanism was
## never actually tested. This tested it.
## RESULT — REGRESSED, did NOT climb:
##   iter 0  -0.7875 25.8% (warm start)   iter 15 -1.135  16.8%
##   iter 5  -0.9475 21.8%                iter 20 -1.0975 18.5%
##   iter 10 -0.995  20.0%
## Self-play dragged the 26% imitation warm-start DOWN to a ~17-18% fixed point while
## contact_loss fell (4.36→3.93) — fitting self-play targets WORSE than wildbg-imitation.
## Met Fable's kill criterion decisively (iters 10/15/20 all far below 31%, declining not
## plateauing). Killed at iter 22.
## DIAGNOSIS — two competing explanations, must distinguish:
##   (A) RACE-NET CONFOUND (flagged pre-launch): EXP4's RACE net is UNTRAINED (contact-only
##       bootstrap, partition race=0 → random race net). Truncation gives exact targets at
##       COVERED race positions, but the UNCOVERED race band (early race, >18-frame, between
##       contact-end and table coverage) is played+valued by the random race net → corrupt
##       value backup propagates to contact → contact value head learns wrong values →
##       worse contact play. FIXABLE: seed a TRAINED frozen race net.
##   (B) FUNDAMENTAL: the net+400-sim+truncation policy-improvement operator has a fixed
##       point BELOW wildbg for contact (search isn't better than wildbg → self-play pulls
##       DOWN). = "contact method-bound / self-play caps at teacher", now shown to cap BELOW
##       teacher from an above-fixed-point start.
## NEXT: distinguish A vs B by re-running with a TRAINED frozen race net. CAVEAT: value-head
## format changed at v9 (joint cumulative) so pre-v9 race nets (distributed_race_20260314,
## 128×3) are WRONG format; recent good race nets (v7/v11) are 256×5. Options: train a fresh
## 128×3 (or 256×5) race net in current format first, or run a cheap diagnostic. Checkpoints
## saved every 5 iters in exp5 dir for inspection.

## Phase 3 — Contact training (small net + exact-race-frontier curriculum)
- [ ] Race-frontier truncation for contact traces: at a race position, evaluate with
      combined table (covered) else frozen race NN, stop rollout there. EXP2 only
      used the combined table as MCTS leaf evaluator; trace truncation remains untested.
- [ ] `contact` training mode: freeze race net, train small contact net (128×3/×5).
- [ ] Loop: contact self-play → match-eval vs wildbg + diagnostics → iterate.

## Housekeeping
- [x] Delete ~240GB intermediate work_* files on Neo (final n18 table is ~118GB, keep).
