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

## Phase 3 — Contact training (small net + exact-race-frontier curriculum)
- [ ] Race-frontier truncation for contact traces: at a race position, evaluate with
      combined table (covered) else frozen race NN, stop rollout there. EXP2 only
      used the combined table as MCTS leaf evaluator; trace truncation remains untested.
- [ ] `contact` training mode: freeze race net, train small contact net (128×3/×5).
- [ ] Loop: contact self-play → match-eval vs wildbg + diagnostics → iterate.

## Housekeeping
- [x] Delete ~240GB intermediate work_* files on Neo (final n18 table is ~118GB, keep).
