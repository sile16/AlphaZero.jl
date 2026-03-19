#!/bin/bash
# Transition from v4 to v5: cosine LR + PER-only (no reanalyze)
# Run on Neo: bash scripts/launch_v5.sh

set -e

SERVER="http://127.0.0.1:9090"
JARVIS="jarvis"
V5_DATA_DIR="/home/sile/alphazero-server-race-v5"

echo "=== Waiting for v4 to complete (iter 200) ==="

while true; do
    ITER=$(curl -s --connect-timeout 3 "$SERVER/api/status" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['iteration'])" 2>/dev/null || echo "0")
    if [ "$ITER" -ge 200 ] 2>/dev/null; then
        echo "v4 reached iteration $ITER — complete!"
        break
    fi
    echo "$(date '+%H:%M:%S') v4 at iter $ITER/200..."
    sleep 60
done

echo ""
echo "=== Recording v4 final results ==="
ssh $JARVIS "python3 -c \"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('/home/sile/alphazero-server-race-v4/tb/')
ea.Reload()
equity = ea.Scalars('eval/equity/value')
winpct = ea.Scalars('eval/win_pct/value')
loss = ea.Scalars('loss/race/value')
print('Iter | Equity | Win% | Loss')
print('-----|--------|------|-----')
for e, w in zip(equity, winpct):
    step = e.step
    l = [x for x in loss if x.step == step]
    lv = f'{l[0].value:.3f}' if l else '-'
    print(f'{step:4d} | {e.value:+.3f}  | {w.value:.1f}% | {lv}')
\"" 2>/dev/null | tee /tmp/v4_final_results.txt

echo ""
echo "=== Killing v4 processes ==="

# Kill Neo client
echo "Killing Neo selfplay client..."
pkill -f "selfplay_client.*neo-cpu" 2>/dev/null || true
sleep 2

# Kill Jarvis client and server
echo "Killing Jarvis selfplay client..."
ssh $JARVIS "pkill -f 'selfplay_client.*jarvis-cpu'" 2>/dev/null || true
sleep 2
echo "Killing Jarvis training server..."
ssh $JARVIS "pkill -f 'training_server.*race-v4'" 2>/dev/null || true
sleep 3

echo "v4 processes killed."

echo ""
echo "=== Launching v5: cosine LR + PER-only (no reanalyze) ==="

# Re-establish SSH tunnel (may have dropped)
pkill -f "ssh.*9090.*jarvis" 2>/dev/null || true
sleep 1
ssh -f -N -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o TCPKeepAlive=yes -L 9090:127.0.0.1:9090 $JARVIS
echo "SSH tunnel established"

# Launch v5 server on Jarvis
echo "Starting v5 training server on Jarvis..."
ssh $JARVIS "bash -l -c 'cd /home/sile/github/AlphaZero.jl && mkdir -p $V5_DATA_DIR && nohup julia --threads 16 --project scripts/training_server.jl \
  --port 9090 \
  --data-dir $V5_DATA_DIR \
  --api-key alphazero-dev-key \
  --race-width 256 --race-blocks 5 \
  --contact-width 256 --contact-blocks 5 \
  --total-iterations 200 \
  --games-per-iteration 500 \
  --training-mode race \
  --use-per \
  --mcts-iters 1600 \
  --inference-batch-size 50 \
  --batch-size 256 \
  --buffer-capacity 3000000 \
  --checkpoint-interval 10 \
  --buffer-checkpoint-interval 50 \
  --eval-mcts-iters 600 \
  --eval-workers 4 \
  --seed 42 \
  --lr-schedule cosine \
  --lr-min 0.0001 \
  --start-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls \
  --eval-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls \
  --bootstrap-file /homeshare/projects/AlphaZero.jl/eval_data/bootstrap_race_samples.jls \
  > /tmp/training_server_race_v5.log 2>&1 &'"
echo "Waiting for server to start..."
sleep 30

# Verify server is up
for i in $(seq 1 10); do
    if ssh $JARVIS "curl -s --connect-timeout 3 http://127.0.0.1:9090/api/status" 2>/dev/null | python3 -m json.tool 2>/dev/null; then
        echo "Server is up!"
        break
    fi
    echo "Waiting for server... (attempt $i/10)"
    sleep 10
done

# Launch Jarvis client
echo "Starting Jarvis selfplay client..."
ssh $JARVIS "bash -l -c 'cd /home/sile/github/AlphaZero.jl && nohup julia --threads 16 --project scripts/selfplay_client.jl --server http://127.0.0.1:9090 --api-key alphazero-dev-key --num-workers 8 --client-name jarvis-cpu > /tmp/selfplay_jarvis_cpu_v5.log 2>&1 &'"
echo "Waiting for Jarvis client to connect..."
sleep 30

# Launch Neo client
echo "Starting Neo selfplay client..."
cd /Users/sile/github/AlphaZero.jl
nohup julia --threads 30 --project scripts/selfplay_client.jl \
  --server http://127.0.0.1:9090 \
  --api-key alphazero-dev-key \
  --num-workers 22 \
  --client-name neo-cpu \
  > /tmp/selfplay_neo_cpu_v5.log 2>&1 &
echo "Neo client launched (PID: $!)"

echo ""
echo "=== v5 launched! ==="
echo "Config: cosine LR (0.001 -> 0.0001), PER-only, no reanalyze, 3M buffer, 1600 MCTS"
echo "Buffer checkpoints every 50 iterations"
echo "Server data: $V5_DATA_DIR"
echo "Logs: /tmp/training_server_race_v5.log, /tmp/selfplay_neo_cpu_v5.log, /tmp/selfplay_jarvis_cpu_v5.log"
echo ""
echo "Monitor with: curl -s http://127.0.0.1:9090/api/status | python3 -m json.tool"
