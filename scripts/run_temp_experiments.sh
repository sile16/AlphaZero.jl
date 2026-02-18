#!/bin/bash
set -e

COMMON="julia --threads 16 --project scripts/train_distributed.jl \
    --num-workers=14 --total-iterations=50 --games-per-iteration=500 \
    --mcts-iters=400 --inference-batch-size=50 --buffer-capacity=600000 \
    --learning-rate=0.001 --use-per --use-reanalyze --seed=42"

echo "=========================================="
echo "Temperature Experiments (4x 50-iter)"
echo "Started: $(date)"
echo "=========================================="

# Exp 1: Step30 (τ=1→0 at move 30)
echo ""
echo ">>> Exp 1: Step30 (τ=1→0 at move 30)"
echo ">>> Started: $(date)"
$COMMON --temp-move-cutoff=30 2>&1 | tee /homeshare/projects/AlphaZero.jl/training_temp_step30.log
EXP1_SESSION=$(ls -td /homeshare/projects/AlphaZero.jl/sessions/distributed_*_temp30 | head -1)
echo ">>> Exp 1 session: $EXP1_SESSION"
echo ">>> Exp 1 GnuBG eval starting: $(date)"
julia --threads 16 --project scripts/eval_vs_gnubg.jl \
    "$EXP1_SESSION/checkpoints/latest.data" minimal 500 128 3 8 100 2>&1 | tee /homeshare/projects/AlphaZero.jl/gnubg_eval_temp_step30.log
echo ">>> Exp 1 complete: $(date)"

# Exp 2: Soft20 (τ=1→0.3 at move 20)
echo ""
echo ">>> Exp 2: Soft20 (τ=1→0.3 at move 20)"
echo ">>> Started: $(date)"
$COMMON --temp-move-cutoff=20 --temp-final=0.3 2>&1 | tee /homeshare/projects/AlphaZero.jl/training_temp_soft20.log
EXP2_SESSION=$(ls -td /homeshare/projects/AlphaZero.jl/sessions/distributed_*_temp20 | head -1)
echo ">>> Exp 2 session: $EXP2_SESSION"
echo ">>> Exp 2 GnuBG eval starting: $(date)"
julia --threads 16 --project scripts/eval_vs_gnubg.jl \
    "$EXP2_SESSION/checkpoints/latest.data" minimal 500 128 3 8 100 2>&1 | tee /homeshare/projects/AlphaZero.jl/gnubg_eval_temp_soft20.log
echo ">>> Exp 2 complete: $(date)"

# Exp 3: IterDecay (τ=1.0→0.3 linear across iterations)
echo ""
echo ">>> Exp 3: IterDecay (τ=1.0→0.3 linear)"
echo ">>> Started: $(date)"
$COMMON --temp-iter-decay 2>&1 | tee /homeshare/projects/AlphaZero.jl/training_temp_iterdecay.log
EXP3_SESSION=$(ls -td /homeshare/projects/AlphaZero.jl/sessions/distributed_*_iterdecay | head -1)
echo ">>> Exp 3 session: $EXP3_SESSION"
echo ">>> Exp 3 GnuBG eval starting: $(date)"
julia --threads 16 --project scripts/eval_vs_gnubg.jl \
    "$EXP3_SESSION/checkpoints/latest.data" minimal 500 128 3 8 100 2>&1 | tee /homeshare/projects/AlphaZero.jl/gnubg_eval_temp_iterdecay.log
echo ">>> Exp 3 complete: $(date)"

# Exp 4: Combined (Step30 + iter decay to 0.5)
echo ""
echo ">>> Exp 4: Combined (Step30 + IterDecay→0.5)"
echo ">>> Started: $(date)"
$COMMON --temp-move-cutoff=30 --temp-iter-decay --temp-iter-final=0.5 2>&1 | tee /homeshare/projects/AlphaZero.jl/training_temp_combined.log
EXP4_SESSION=$(ls -td /homeshare/projects/AlphaZero.jl/sessions/distributed_*_temp30_iterdecay | head -1)
echo ">>> Exp 4 session: $EXP4_SESSION"
echo ">>> Exp 4 GnuBG eval starting: $(date)"
julia --threads 16 --project scripts/eval_vs_gnubg.jl \
    "$EXP4_SESSION/checkpoints/latest.data" minimal 500 128 3 8 100 2>&1 | tee /homeshare/projects/AlphaZero.jl/gnubg_eval_temp_combined.log
echo ">>> Exp 4 complete: $(date)"

echo ""
echo "=========================================="
echo "All 4 temperature experiments complete!"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Log files:"
echo "  training_temp_step30.log + gnubg_eval_temp_step30.log"
echo "  training_temp_soft20.log + gnubg_eval_temp_soft20.log"
echo "  training_temp_iterdecay.log + gnubg_eval_temp_iterdecay.log"
echo "  training_temp_combined.log + gnubg_eval_temp_combined.log"
