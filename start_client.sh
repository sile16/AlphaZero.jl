#!/usr/bin/env bash
#
# Auto-restarting selfplay client wrapper.
# On exit: sleep 30, git pull, restart.
# Server can trigger restart via: curl -X POST -H "Authorization: Bearer <key>" http://jarvis:9090/api/restart-clients
#
# Usage:
#   ./start_client.sh --threads 30 [--server http://jarvis:9090] [extra args...]
#
# Examples:
#   ./start_client.sh --threads 30
#   ./start_client.sh --threads 10 --eval-capable --wildbg-lib /path/to/libwildbg.dylib

set -euo pipefail
cd "$(dirname "$0")"

THREADS=""
SERVER="http://jarvis:9090"
API_KEY="alphazero-dev-key"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)
            THREADS="$2"; shift 2 ;;
        --server)
            SERVER="$2"; shift 2 ;;
        --api-key)
            API_KEY="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$THREADS" ]]; then
    echo "Usage: $0 --threads <N> [--server URL] [extra args...]"
    exit 1
fi

LOG="/tmp/selfplay_$(hostname)_$(date +%Y%m%d_%H%M%S).log"

echo "=== AlphaZero Self-Play Client ==="
echo "Threads: $THREADS (= workers)"
echo "Server: $SERVER"
echo "Log: $LOG"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

while true; do
    echo "[$(date)] Starting client..."

    julia --threads "$THREADS" --project scripts/selfplay_client.jl \
        --server "$SERVER" --api-key "$API_KEY" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOG" || true

    echo ""
    echo "[$(date)] Client exited. Pulling latest code..."
    git pull --ff-only 2>&1 || echo "  git pull failed (non-fatal)"
    echo "[$(date)] Sleeping 30s before restart..."
    sleep 30
done
