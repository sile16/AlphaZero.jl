#!/bin/bash
# v7: distributed eval (server never blocks) + unified game loop
# Key changes from v6: eval moved to clients, server stays responsive
# Jarvis client: eval-capable (12 workers), Neo client: self-play only (32 workers)
# Run on Neo: bash scripts/launch_v7.sh

set -e

SERVER="http://127.0.0.1:9090"
JARVIS="jarvis"
V7_DATA_DIR="/home/sile/alphazero-server-race-v7"

echo "=== Launching v7: distributed eval + unified game loop ==="
echo "Key changes: eval on clients (not server), checkpoint-interval 5, 32+12 workers"

# Re-establish SSH tunnel
pkill -f "ssh.*9090.*jarvis" 2>/dev/null || true
sleep 1
ssh -f -N -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o TCPKeepAlive=yes -L 9090:127.0.0.1:9090 $JARVIS
echo "SSH tunnel established"

# Launch v7 server on Jarvis (no server-side eval — eval is distributed to clients)
echo "Starting v7 training server on Jarvis..."
ssh $JARVIS "bash -l -c 'cd /home/sile/github/AlphaZero.jl && mkdir -p $V7_DATA_DIR && nohup julia --threads 16 --project scripts/training_server.jl \
  --port 9090 \
  --data-dir $V7_DATA_DIR \
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
  --checkpoint-interval 5 \
  --buffer-checkpoint-interval 50 \
  --eval-mcts-iters 600 \
  --eval-games 1000 \
  --eval-workers 0 \
  --seed 42 \
  --lr-schedule cosine \
  --lr-min 0.0001 \
  --start-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls \
  --eval-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls \
  --bootstrap-file /homeshare/projects/AlphaZero.jl/eval_data/bootstrap_race_samples.jls \
  >> /tmp/training_server_race_v7.log 2>&1 &'"
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

# Launch Jarvis client (eval-capable: does self-play + eval when available)
echo "Starting Jarvis selfplay+eval client..."
ssh $JARVIS "bash -l -c 'cd /home/sile/github/AlphaZero.jl && nohup julia --threads 16 --project scripts/selfplay_client.jl \
  --server http://127.0.0.1:9090 \
  --api-key alphazero-dev-key \
  --num-workers 12 \
  --client-name jarvis-cpu \
  --eval-capable \
  --eval-mcts-iters 600 \
  --wildbg-lib /home/sile/github/wildbg/target/release/libwildbg.so \
  --eval-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls \
  >> /tmp/selfplay_jarvis_cpu_v7.log 2>&1 &'"
echo "Waiting for Jarvis client to connect..."
sleep 30

# Launch Neo client (self-play only — Neo is 3.6x less efficient at eval per core)
echo "Starting Neo selfplay client..."
cd /Users/sile/github/AlphaZero.jl
nohup julia --threads 30 --project scripts/selfplay_client.jl \
  --server http://127.0.0.1:9090 \
  --api-key alphazero-dev-key \
  --num-workers 32 \
  --client-name neo-cpu \
  >> /tmp/selfplay_neo_cpu_v7.log 2>&1 &
echo "Neo client launched (PID: $!)"

echo ""
echo "=== v7 launched! ==="
echo "Config: distributed eval (Jarvis), cosine LR, PER, 4000 steps/iter, checkpoint every 5 iters"
echo "  Jarvis: 12 workers (self-play + eval-capable)"
echo "  Neo: 32 workers (self-play only)"
echo "Server data: $V7_DATA_DIR"
echo "Logs: /tmp/training_server_race_v7.log, /tmp/selfplay_neo_cpu_v7.log, /tmp/selfplay_jarvis_cpu_v7.log"
echo ""
echo "Monitor: curl -s http://127.0.0.1:9090/api/status | python3 -m json.tool"
echo "Eval status: curl -s http://127.0.0.1:9090/api/eval/status | python3 -m json.tool"
