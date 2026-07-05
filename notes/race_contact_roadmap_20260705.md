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
- [ ] WIRE COMBINED TABLE INTO MCTS (next, testable on Neo): make_bearoff_evaluator +
      bearoff_table_equity + bearoff_post_dice_equity → generic bearoff_covers/bearoff_lookup;
      include bearoff_onesided.jl + bearoff_combined.jl in selfplay_client; BEAROFF_TABLE =
      CombinedBearoff(k7, onesided) when onesided present else k7-only. TEST on Neo: combined
      evaluator returns correct exact race values; k7-only path unchanged.
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

## Phase 2 — Finalize race
- [ ] Definitive race eval across candidate checkpoints (current fixed code) → pick best.
- [ ] Confirm exact-table absorption (bearoff corr vs table ≥ 0.99; ship <50MB table).
- [ ] Freeze race_best.data as the frozen race evaluator.

## EXP1 LAUNCHED (2026-07-05 ~15:15): dual 128×3 (283k params), 200 iters, cold-start,
## eval-vs-wildbg every 10 iters (contact_eval_2000.jls, 200 games). Server PID on Jarvis
## (data-dir /home/sile/alphazero-contact-exp1, port 9090); eval-capable client on
## Jarvis-localhost (Neo blocked by Julia-LAN networking). Combined-table curriculum ACTIVE
## (self-play truncates at exact race frontier). Signal = match win% vs wildbg (goal >50%).
## Logs: exp1_server.log / exp1_client.log. Validated live: combined table loads, contact
## opening, self-play ~2 games/sec, eval chunks assigned. NOTE cold-start won't reach parity
## fast — watching trajectory; if it plateaus below wildbg, next lever = bootstrap-to-parity
## then chance-node A/B (passthrough vs progressive-widening).

## Phase 3 — Contact training (small net + exact-race-frontier curriculum)
- [ ] Race-frontier truncation for contact MCTS: at a race position, evaluate with
      combined table (covered) else frozen race NN, stop rollout there.
- [ ] `contact` training mode: freeze race net, train small contact net (128×3/×5).
- [ ] Loop: contact self-play → match-eval vs wildbg + diagnostics → iterate.

## Housekeeping
- [ ] Delete ~240GB intermediate work_* files on Neo (final n18 table is ~118GB, keep).
