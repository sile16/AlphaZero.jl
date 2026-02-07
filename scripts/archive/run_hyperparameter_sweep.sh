#!/bin/bash
# Hyperparameter sweep for finding optimal training setup
# Each experiment runs 30 iterations with GnuBG evaluation

set -e

COMMON_ARGS="--game=backgammon-deterministic --network-type=fcresnet-multihead --num-workers=6 --total-iterations=30 --games-per-iteration=50 --eval-interval=10 --eval-games=50 --eval-mcts-iters=50 --final-eval-games=200"

echo "============================================================"
echo "Hyperparameter Sweep - $(date)"
echo "============================================================"

# Experiment A: Baseline (128w x 3b, 100 MCTS)
echo ""
echo ">>> Experiment A: Baseline (128w x 3b, 100 MCTS)"
julia --project --threads=8 scripts/train_cluster.jl \
    $COMMON_ARGS \
    --network-width=128 \
    --network-blocks=3 \
    --mcts-iters=100 \
    --seed=42

# Experiment B: Wider network (256w x 3b, 100 MCTS)
echo ""
echo ">>> Experiment B: Wider network (256w x 3b, 100 MCTS)"
julia --project --threads=8 scripts/train_cluster.jl \
    $COMMON_ARGS \
    --network-width=256 \
    --network-blocks=3 \
    --mcts-iters=100 \
    --seed=42

# Experiment C: Deeper network (128w x 6b, 100 MCTS)
echo ""
echo ">>> Experiment C: Deeper network (128w x 6b, 100 MCTS)"
julia --project --threads=8 scripts/train_cluster.jl \
    $COMMON_ARGS \
    --network-width=128 \
    --network-blocks=6 \
    --mcts-iters=100 \
    --seed=42

# Experiment D: More MCTS (128w x 3b, 200 MCTS)
echo ""
echo ">>> Experiment D: More MCTS (128w x 3b, 200 MCTS)"
julia --project --threads=8 scripts/train_cluster.jl \
    $COMMON_ARGS \
    --network-width=128 \
    --network-blocks=3 \
    --mcts-iters=200 \
    --seed=42

echo ""
echo "============================================================"
echo "Sweep complete - $(date)"
echo "============================================================"
