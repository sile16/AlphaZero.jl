#!/bin/bash
# v8: multihead masking fix + strict checksums + robustness improvements
# Key change from v7: conditional equity heads now correctly masked during training
# Direct IP connection (no SSH tunnel) — Jarvis at 192.168.20.40
# Jarvis client: eval-capable (12 workers), Neo client: self-play only (32 workers)
# Run on Neo: bash scripts/launch_v8.sh

set -e

JARVIS_IP="192.168.20.40"
SERVER="http://127.0.0.1:9090"
V8_DATA_DIR="/home/sile/alphazero-server-race-v8"

echo "=== Launching v8: multihead masking fix ==="
echo "Key changes: conditional head masking, strict weight checksums"
echo "Server (via tunnel): $SERVER"

# NOTE: Direct IP (192.168.20.40) works for curl/nc but Julia's libuv
# gets EHOSTUNREACH on macOS (likely App Management restriction on
# non-Apple-signed binaries). SSH tunnel is required for Neo.

# Re-establish SSH tunnel (required for Neo — see note above)
pkill -f "ssh.*9090.*jarvis" 2>/dev/null || true
sleep 1
ssh -f -N -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o TCPKeepAlive=yes -L 9090:127.0.0.1:9090 jarvis
echo "SSH tunnel established"

# Sync code to Jarvis
echo "Syncing code to Jarvis..."
ssh jarvis "cd /home/sile/github/AlphaZero.jl && git pull"
echo "Code synced"

# Launch v8 server on Jarvis
echo "Starting v8 training server on Jarvis..."
ssh jarvis "bash -l -c 'cd /home/sile/github/AlphaZero.jl && mkdir -p $V8_DATA_DIR && nohup julia --threads 16 --project scripts/training_server.jl \
  --port 9090 \
  --data-dir $V8_DATA_DIR \
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
  --eval-interval 5 \
  --eval-games 1000 \
  --eval-workers 0 \
  --seed 42 \
  --lr-schedule cosine \
  --lr-min 0.0001 \
  --start-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls \
  --eval-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls \
  --bootstrap-file /homeshare/projects/AlphaZero.jl/eval_data/bootstrap_race_samples.jls \
  >> /tmp/training_server_race_v8.log 2>&1 &'"
echo "Waiting for server to start..."
sleep 30

# Verify server is up (via tunnel)
for i in $(seq 1 10); do
    if curl -s --connect-timeout 3 "$SERVER/api/status" 2>/dev/null | python3 -m json.tool 2>/dev/null; then
        echo "Server is up!"
        break
    fi
    echo "Waiting for server... (attempt $i/10)"
    sleep 10
done

# Launch Jarvis client (eval-capable: does self-play + eval when available)
echo "Starting Jarvis selfplay+eval client..."
ssh jarvis "bash -l -c 'cd /home/sile/github/AlphaZero.jl && nohup julia --threads 16 --project scripts/selfplay_client.jl \
  --server http://127.0.0.1:9090 \
  --api-key alphazero-dev-key \
  --num-workers 12 \
  --client-name jarvis-cpu \
  --eval-capable \
  --eval-mcts-iters 600 \
  --wildbg-lib /home/sile/github/wildbg/target/release/libwildbg.so \
  --eval-positions-file /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls \
  >> /tmp/selfplay_jarvis_cpu_v8.log 2>&1 &'"
echo "Waiting for Jarvis client to connect..."
sleep 30

# Launch Neo client (self-play only — via SSH tunnel to localhost)
echo "Starting Neo selfplay client..."
cd /Users/sile/github/AlphaZero.jl
nohup julia --threads 30 --project scripts/selfplay_client.jl \
  --server "$SERVER" \
  --api-key alphazero-dev-key \
  --num-workers 32 \
  --client-name neo-cpu \
  >> /tmp/selfplay_neo_cpu_v8.log 2>&1 &
echo "Neo client launched (PID: $!)"

echo ""
echo "=== v8 launched! ==="
echo "Config: multihead masking fix, cosine LR, PER, 4000 steps/iter, checkpoint every 5 iters"
echo "  Jarvis: 12 workers (self-play + eval-capable) via localhost"
echo "  Neo: 32 workers (self-play only) via SSH tunnel"
echo "Server data: $V8_DATA_DIR"
echo "Logs:"
echo "  Server: ssh jarvis 'tail -f /tmp/training_server_race_v8.log'"
echo "  Jarvis: ssh jarvis 'tail -f /tmp/selfplay_jarvis_cpu_v8.log'"
echo "  Neo:    tail -f /tmp/selfplay_neo_cpu_v8.log"
echo ""
echo "Monitor: curl -s http://127.0.0.1:9090/api/status | python3 -m json.tool"
echo "Eval:    curl -s http://127.0.0.1:9090/api/eval/status | python3 -m json.tool"
