# Session Handoff — 2026-07-03

State of the world for whoever picks this up (human or agent). Everything below
is committed on `master` unless marked otherwise.

## What happened today (commit trail)

- `8eb1a74` Review round-1 fixes: terminal-bearoff gammon, MCTS value scale
  (`GI.reward_scale`), buffer torn reads, doubles mid-turn (+39 regression tests)
- `98e351c` Fixed-set bearoff eval infra + `eval_table_vs_wildbg.jl` validation
  harness (duplicate-dice paired; the harness that CAUGHT the doubles bug)
- `6100d4b` Docs/cleanup; `a920989` harness MCTS/objective modes + findings
- `24a480d` Promotion gate (publication gated on bearoff-eval value MAE)
- `5de15c8` MCTS identity staircase rungs 1–3: search exact to machine epsilon
- `52b286d` Review round-3 fixes: gate resume split (train vs published state),
  calibrated fail-closed, reanalyze generation counters, last scale site
- `072b236` Error-response curves rungs 6–7

Full analysis notes: `notes/review_fixes_20260703.md` (3 review rounds),
`notes/mcts_identity_staircase_20260703.md`, `notes/mcts_convergence_sweep_20260703.md`,
`notes/error_response_curves_20260703.md`, `notes/mcts_objective_validation_20260703.md`,
`notes/ab_fix_impact_20260703.md`, `notes/promotion_gate_design_20260703.md`.

## Key results to carry forward

1. **MCTS is correctness-clean** (identity-proven, permanent tests). Finite-budget
   behavior: 400 iters = knee with exact evaluators; 30 iters is WORSE than no
   search (visit-argmax discards exact Q — max-Q root selection would fix; open
   experiment).
2. **MCTS corrects variance, never frontier bias.** Frozen (NN-like) value error
   plateaus at the 1-ply anchor at every sims budget. Only depth past the frontier
   corrects bias.
3. **Never use the NN as pre-dice frontier evaluator** (OOD, +0.59 raw-pts bias →
   44× worse regret than NN-as-oracle+passthrough). Keep exact k=7 table cutoff.
4. **Track differential (within-sibling) RMSE, not raw MAE/RMSE** — 94.7% of the
   v12 net's frontier value error is common-mode shift that cancels in argmax.
   Raw error overstates move error ~4×. NOTE: the promotion gate currently gates
   on raw value MAE — refining it to differential RMSE is an open improvement.
5. **No gammon-go/save skill edge exists in pure bearoff** (objective-argmax ≠
   money-argmax on only 0.15–5.6% of decisions). That skill lives in
   pre-bearoff/contact — the n=18 table band.
6. **One-sided n=18 race table validated**: vs exact two-sided k=7 at n=7,
   pW mean err 5e-4, equity MAE 0.0012. Gammon-term tail errors (≤0.25) in
   deep-back-stack gammon-live positions = the efficiency-personality limitation;
   gammon-save sibling personality is the designed fix.

## In flight right now

- **Neo: n=18 one-sided table build** — nohup'd, detached, survives everything;
  ~12–20h total; checkpointed every 50M positions.
  - Monitor: `ssh neo 'tail -f ~/github/BackgammonNet.jl/tools/bearoff_onesided/build_n18.log'`
  - Alive: `ssh neo 'pgrep -fl "build_onesided.jl --n 18"'`
  - Resume after any interruption (byte-identical): same command + `--resume`
  - Output: `.../bearoff_onesided/bearoff_n18/` (~118 GB when done)
- **Jarvis: eval_race `--bearoff-eval` agent** — added optional exact-table
  evaluator to `scripts/eval_race.jl` (uncommitted; default-off flag), measuring
  its effect on v12 iter-50 (control + table-assisted, 1000 games each @600).
  If interrupted: the edit is in the working tree; runs are cheap to redo.

## Next steps in priority order

1. **Commit `eval_race.jl` bearoff-eval work** once its measurement lands (the
   only uncommitted file).
2. **Longer-race validation rung** (when n=18 finishes): build the harness swap —
   `BackgammonNet.jl/src/bearoff_onesided.jl` mirrors the k7 lookup API
   (`lookup`, `combine_indices`); race-start data already staged on Neo at
   `~/eval_data/`. Ladder: one-sided-table policy vs wildbg from 18-point race
   starts (paired, like eval_table_vs_wildbg.jl) → +MCTS → +objectives. THIS is
   the band where wildbg should actually be beatable and where v12 plateaued.
3. **Gammon-save personality** for the one-sided table (fixes the gammon-term
   tail; format is versioned for sibling personalities).
4. **v13 race training launch**: all fixes + promotion gate are in. Suggested
   deltas vs v12: gate enabled (default), keep k=7 evaluator, consider
   differential-RMSE gate metric. Expect training-side gains from the
   doubles/terminal fixes (eval-side was measured ≈ neutral; the corruption was
   in training targets).
5. **Open experiments**: max-Q root selection at bearoff nodes (frees ~10×
   endgame sim budget); one-ply NN-value priors vs policy head (cold-start aid);
   progressive-widening retest once model > wildbg.

## Machine/infra notes

- Neo llama-server instances were killed this session to free RAM for the build
  (`pkill -f llama-server`) — the openclaw preset router on port 8081 is DOWN
  until manually restarted.
- Suite: `julia --project -e 'using Pkg; Pkg.test()'` — 1799 assertions, needs
  the k=7 table for the identity/doubles testsets (auto-skip if absent).
- Memory (Claude): `~/.claude/projects/-home-sile-github-AlphaZero-jl/memory/`
  mirrors this handoff.
