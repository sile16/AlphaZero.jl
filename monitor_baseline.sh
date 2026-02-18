#!/bin/bash
# Monitor baseline training, log progress every 10 minutes
LOG="/homeshare/projects/AlphaZero.jl/training_200iter_baseline.log"
while kill -0 1295102 2>/dev/null; do
    echo "=== $(date) ==="
    grep -E "^┌ Info: Iteration" "$LOG" | tail -1
    grep "avg_loss" "$LOG" | tail -1
    grep "Eval results" "$LOG" | tail -1
    echo ""
    sleep 600
done
echo "=== Training complete: $(date) ==="
grep "avg_loss" "$LOG" | tail -1
