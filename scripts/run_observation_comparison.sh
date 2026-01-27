#!/bin/bash
# Run observation type comparison tests
# Each test: ~70 iterations, 50 games/iter = 3500 games + 1000 final eval = 4500 games
# Feature counts (BackgammonNet v0.2.8): minimal=780, full=1612, biased=3172

set -e

# Common parameters (matching previous test run)
TOTAL_ITERS=70
GAMES_PER_ITER=50
FINAL_EVAL=1000
MCTS_ITERS=100
WORKERS=6
NETWORK_WIDTH=128
NETWORK_BLOCKS=3

echo "================================================================"
echo "Observation Type Comparison Test"
echo "================================================================"
echo "Parameters:"
echo "  Total iterations: $TOTAL_ITERS"
echo "  Games per iteration: $GAMES_PER_ITER"
echo "  Final eval games: $FINAL_EVAL"
echo "  MCTS iterations: $MCTS_ITERS"
echo "  Workers: $WORKERS"
echo "  Network: FCResNet-MultiHead ($NETWORK_WIDTH width, $NETWORK_BLOCKS blocks)"
echo "================================================================"
echo ""

# Test 1: MINIMAL (780 features)
echo "================================================================"
echo "TEST 1/3: MINIMAL observation (780 features)"
echo "================================================================"
BACKGAMMON_OBS_TYPE=minimal julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=$NETWORK_WIDTH \
    --network-blocks=$NETWORK_BLOCKS \
    --num-workers=$WORKERS \
    --total-iterations=$TOTAL_ITERS \
    --games-per-iteration=$GAMES_PER_ITER \
    --mcts-iters=$MCTS_ITERS \
    --final-eval-games=$FINAL_EVAL \
    --wandb-run-name=obs-minimal-${NETWORK_WIDTH}w${NETWORK_BLOCKS}b

echo ""
echo "================================================================"
echo "TEST 2/3: FULL observation (1612 features)"
echo "================================================================"
BACKGAMMON_OBS_TYPE=full julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=$NETWORK_WIDTH \
    --network-blocks=$NETWORK_BLOCKS \
    --num-workers=$WORKERS \
    --total-iterations=$TOTAL_ITERS \
    --games-per-iteration=$GAMES_PER_ITER \
    --mcts-iters=$MCTS_ITERS \
    --final-eval-games=$FINAL_EVAL \
    --wandb-run-name=obs-full-${NETWORK_WIDTH}w${NETWORK_BLOCKS}b

echo ""
echo "================================================================"
echo "TEST 3/3: BIASED observation (3172 features)"
echo "================================================================"
BACKGAMMON_OBS_TYPE=biased julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=$NETWORK_WIDTH \
    --network-blocks=$NETWORK_BLOCKS \
    --num-workers=$WORKERS \
    --total-iterations=$TOTAL_ITERS \
    --games-per-iteration=$GAMES_PER_ITER \
    --mcts-iters=$MCTS_ITERS \
    --final-eval-games=$FINAL_EVAL \
    --wandb-run-name=obs-biased-${NETWORK_WIDTH}w${NETWORK_BLOCKS}b

echo ""
echo "================================================================"
echo "ALL TESTS COMPLETE"
echo "Check WandB for results comparison"
echo "================================================================"
