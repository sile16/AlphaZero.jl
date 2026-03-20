#!/bin/bash
# v6: 10x more gradient steps per iteration (4000 vs ~390), 2000-game eval
# Run on Neo: bash scripts/launch_v6.sh

set -e

SERVER="http://127.0.0.1:9090"
JARVIS="jarvis"
V6_DATA_DIR="/home/sile/alphazero-server-race-v6"

echo "=== Launching v6: high replay ratio + 2000-game eval ==="
echo "Key change: --training-steps 4000 (10x v5), --eval-games 1000 (10x v5), --eval-workers 14"

# Re-establish SSH tunnel
pkill -f "ssh.*9090.*jarvis" 2>/dev/null || true
sleep 1
ssh -f -N -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o TCPKeepAlive=yes -L 9090:127.0.0.1:9090 $JARVIS
echo "SSH tunnel established"

# Launch v6 server on Jarvis
echo "Starting v6 training server on Jarvis..."
ssh $JARVIS "bash -l -c 'cd /home/sile/github/AlphaZero.jl && mkdir -p $V6_DATA_DIR && nohup julia --threads 16 --project scripts/training_server.jl \
  --port 9090 \
  --data-dir $V6_DATA_DIR \
  --api-key alphazero-dev-key \
  --race-width 256 --race-blocks 5 \
  --contact-width 256 --contact-blocks 5 \
  --total-iterations 200 \
  --games-per-iteration 500 \
  --training-steps 4000 \
  --training-mode race \
  --use-per \
  --mcts-iters 1600 \
  --inference-batch-size 50 \
  --batch-size 256 \
  --buffer-capacity 3000000 \
  --checkpoint-interval 10 \
  --buffer-checkpoint-interval 50 \
  --eval-mcts-iters 600 \
  --eval-games 1000 \
  --eval-workers 12 \
  --seed 42 \
  --lr-schedule cosine \
  --lr-min 0.0001 \
  --start-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls \
  --eval-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls \
  --bootstrap-file /homeshare/projects/AlphaZero.jl/eval_data/bootstrap_race_samples.jls \
  >> /tmp/training_server_race_v6.log 2>&1 &'"
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
ssh $JARVIS "bash -l -c 'cd /home/sile/github/AlphaZero.jl && nohup julia --threads 16 --project scripts/selfplay_client.jl --server http://127.0.0.1:9090 --api-key alphazero-dev-key --num-workers 12 --client-name jarvis-cpu >> /tmp/selfplay_jarvis_cpu_v6.log 2>&1 &'"
echo "Waiting for Jarvis client to connect..."
sleep 30

# Launch Neo client
echo "Starting Neo selfplay client..."
cd /Users/sile/github/AlphaZero.jl
nohup julia --threads 30 --project scripts/selfplay_client.jl \
  --server http://127.0.0.1:9090 \
  --api-key alphazero-dev-key \
  --num-workers 32 \
  --client-name neo-cpu \
  >> /tmp/selfplay_neo_cpu_v6.log 2>&1 &
echo "Neo client launched (PID: $!)"

echo ""
echo "=== v6 launched! ==="
echo "Config: cosine LR, PER, 4000 gradient steps/iter (10x v5), 2000-game eval, 14 eval workers, 32+12 selfplay workers"
echo "Server data: $V6_DATA_DIR"
echo "Logs: /tmp/training_server_race_v6.log, /tmp/selfplay_neo_cpu_v6.log, /tmp/selfplay_jarvis_cpu_v6.log"
echo ""
echo "Monitor with: curl -s http://127.0.0.1:9090/api/status | python3 -m json.tool"
