# A/B Experiment: Play-Strength Impact of 2026-07-03 Bug Fixes (Eval-Time)

> Superseded 2026-07-06: AlphaZero bearoff helpers now live in BackgammonNet
> v0.6.2+; legacy `scripts/bearoff_eval_common.jl` references are historical.

**Date:** 2026-07-03
**Question:** How much do today's fixes (commit `8eb1a74` + `98e351c`) change *eval-time play strength*, holding the trained network constant?

## Design

- **Checkpoint (constant in both arms):** `/home/sile/alphazero-server-race-v12/checkpoints/race_iter_50.data` (race model, 256w×5b)
- **Arm A (pre-fix):** git worktree at `26a6a45` (last commit before fixes), run with `--project=<worktree>` so the worktree's own `src/` is loaded. Manifest.toml copied from main tree (worktree has none; Project.toml deps identical, only `[extras]` differ).
- **Arm B (post-fix):** main tree at `6100d4b` (current master, all fixes).
- **Eval:** `scripts/eval_race.jl`, 500 fixed race positions × 2 sides = 1000 games/arm, `--mcts-iters=600`, `--num-workers=10`, `--threads 12`, cpuct=1.5, temperature=0, no Dirichlet noise, `--inference-backend=auto` (FastWeights both arms).
- **Positions:** first 500 of `eval_data/race_eval_2000.jls` — identical set and per-game RNG seeds (`seed=job`) in both arms, so dice sequences start identical; games diverge only after the first move where the two MCTS versions disagree.
- **Opponent:** wildbg `libwildbg.so` (20MB build, loaded as large nets) — same binary both arms.
- **Sequential runs** on an otherwise idle Jarvis.
- **Script interface:** identical CLI in both revisions; the only `eval_race.jl` diff between arms is the NN-value ×3 scaling in the *value-metric reporting* path (fix #5 below), which does not affect play.

Raw logs: scratchpad `results/armA_prefix.log`, `results/armB_postfix.log` (smoke runs: `smokeA.log`, `smokeB.log`).

## Which fixed code paths are actually active in this eval

| Fix (from `8eb1a74`/`98e351c`) | Active in eval_race.jl? | Why |
|---|---|---|
| **#2 MCTS reward-scale** (`GI.reward_scale`, `batched_mcts.jl` `traverse_to_leaf!` normalizes terminal rewards ±1/±2/±3 → ±1/3/±2/3/±1) | **YES — the only play-affecting difference** | Eval uses `GameLoop.MctsAgent` → `BatchedMCTS.BatchedMctsPlayer` → `traverse_to_leaf!`. Race positions reach game end within the 600-iter search horizon constantly, so terminal rewards are backed up through the tree and (pre-fix) mixed at raw scale with NN values (equity/3 ∈ [-1,1]) in the same Q totals. |
| #1 Terminal bearoff moves hard-coded as simple wins | NO | Lives in `selfplay_client.jl` training-target and bearoff-evaluator paths. `eval_race.jl` constructs `MctsAgent(...)` **without** `bearoff_eval` (defaults to `nothing`) — verified in both revisions. |
| #4 Doubles mid-turn bearoff mis-scoring (`bearoff_eval_common.jl`) | NO | Same reason — no bearoff evaluator wired into eval MCTS. (`bearoff_eval_common.jl` doesn't exist in the pre-fix revision, and nothing in the eval path references it.) |
| #3 Buffer torn reads (`distributed/buffer.jl`) | NO | Training-server only. |
| #5 Value-metric ×3 scale (`eval_race.jl` reporting) | YES, reporting only | Changes value MSE/MAE/bias printed by the script; zero effect on move selection. |
| `src/mcts.jl` reward-scale change | Compiled but not exercised | Eval traversal goes through `batched_mcts.jl`, not the standard engine's `run_simulation!`. |

**Attribution conclusion: the entire play-strength delta between arms is the MCTS reward-scale fix (fix #2), full stop.** The bearoff-evaluator fixes (#1, #4) are not exercised by this eval — their play impact would only show up in selfplay/training or in evals that wire `bearoff_eval` into `MctsAgent`.

## Results

1000 games per arm (500 positions × both sides), same checkpoint, same positions, same seeds.

| Metric | Arm A (pre-fix `26a6a45`) | Arm B (post-fix `6100d4b`) | Δ (B − A) |
|---|---|---|---|
| **Equity** | **+0.026** | **+0.036** | **+0.010** |
| Win% | 50.0% | 50.0% | ~0 |
| Equity as white | +0.026 | +0.024 | −0.002 |
| Equity as black | +0.026 | +0.048 | +0.022 |
| Time | 58.9 s | 59.0 s | — |

Value metrics (within-arm only — **NOT comparable across arms**: pre-fix code compares NN equity/3 against wildbg raw points, so arm A's MSE/MAE/bias are computed on mismatched scales by construction):

| Metric | Arm A (mismatched scales) | Arm B (matched scales) |
|---|---|---|
| n samples | 6560 | 6571 |
| MSE | 0.6137 | 0.2221 |
| MAE | 0.7132 | 0.2101 |
| Bias | −0.0466 | −0.0258 |
| **Correlation** (scale-invariant) | **0.9267** | **0.9258** |
| NN mean±std | 0.010±0.416 | 0.033±1.245 |
| WB mean±std | 0.056±1.152 | 0.058±1.150 |

The correlation being identical (0.927 vs 0.926) is the expected sanity check — it's the same network; only the reporting scale changed (arm A NN std 0.416 ≈ arm B's 1.245/3). The apparent "MSE improvement" 0.61 → 0.22 is purely the ×3 reporting fix, not a model change.

## Significance

The script reports aggregates only (no per-game rewards), so a paired test isn't possible. Rough unpaired bound: per-game reward std is ≈1.0–1.1 (50% wins, mostly ±1 with some gammons), giving SE(mean) ≈ 0.033 per arm and SE(Δ) ≈ 0.047 unpaired. The shared position set and shared dice seeds partially pair the arms (win% identical at exactly 50.0% suggests most games played out identically), so the true SE of Δ is likely smaller — but even so:

**Δ equity = +0.010 is well within noise (|z| ≈ 0.2 unpaired). No statistically detectable play-strength change.**

## Interpretation

1. **The reward-scale bug had ~no measurable effect on eval-time play strength for this model on race positions.** Plausible reasons:
   - Race games are short and gammon-poor from these mutual-race starts, so pre-fix terminal rewards were mostly a *uniform* 3× inflation of ±1 outcomes. A uniform scaling of the dominant value source distorts the PUCT exploration/exploitation balance (effectively raising cpuct's relative weight on NN-leaf branches) but often leaves the visit-count argmax unchanged.
   - The model is already at ~parity with wildbg on races; most race moves are pip-count-obvious and both search versions agree.
   - At 600 iters with deep terminal coverage, the search is dominated by real outcomes either way.
2. **This experiment does NOT bound the fixes' impact on training** (buffer races, terminal-gammon targets, selfplay bearoff evaluator scale/doubles bugs all sit in the training path) or on **contact models / gammon-heavy positions**, where ±2/±3 rewards mix non-uniformly with NN values and the pre-fix distortion is not a uniform scaling.
3. The dramatic value-MSE drop in the logs (0.61 → 0.22) is a **reporting fix**, not a model or search improvement — do not cite it as a strength gain.
4. Slight per-side asymmetry moved (black +0.026 → +0.048) but with per-side n=500 the SE is ~0.047/side; not meaningful.

## Reproduction

```bash
# Arm A
git worktree add /tmp/az_prefix 26a6a45
cp Manifest.toml /tmp/az_prefix/
julia --threads 12 --project=/tmp/az_prefix /tmp/az_prefix/scripts/eval_race.jl \
  /home/sile/alphazero-server-race-v12/checkpoints/race_iter_50.data \
  --width=256 --blocks=5 --num-workers=10 --mcts-iters=600 --num-games=500 \
  --wildbg-lib=/home/sile/github/wildbg/target/release/libwildbg.so \
  --positions-file=eval_data/race_eval_2000.jls

# Arm B: same command with --project=. from master (6100d4b)
```
