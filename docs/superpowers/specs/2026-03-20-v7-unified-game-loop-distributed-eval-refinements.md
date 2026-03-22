# v7 Refinements: Unified Game Loop + Distributed Eval

Following the architectural review of the v7 design, the following refinements and considerations should be integrated during implementation to ensure robustness and efficiency.

## 1. Deterministic Worker Synchronization
The current plan uses `PAUSE_SELFPLAY` with a 2-second `sleep` to wait for in-flight games. For more robust synchronization:
- Use an atomic counter `ACTIVE_SELFPLAY_GAMES` on the client.
- Workers increment this before starting a game and decrement after.
- The `check_and_do_eval!` loop should wait until `ACTIVE_SELFPLAY_GAMES` hits zero before starting evaluation games, ensuring complete isolation of resources (CPU/Memory/GPU) during eval.

## 2. Network & Fault Tolerance
Distributed evaluation introduces network dependencies.
- **Client Retries:** Implement exponential backoff for `POST /api/eval/submit` and `POST /api/eval/heartbeat`. If a submission fails, the client should retry several times before considering the chunk "lost."
- **Heartbeat Resilience:** A single failed heartbeat due to a transient network blip should not cause the client to immediately abandon a chunk. Implement a "missed heartbeat" threshold (e.g., 3 consecutive failures) before abandonment.

## 3. Memory & Payload Management
- **Aggregation Safety:** For 4,000 games, `vcat(vsamples...)` is manageable (~10-20MB), but if `PositionValueSample` is recorded for every move in a long game, it can grow. 
- Ensure the server-side `/api/eval/submit` endpoint has a reasonable body size limit (e.g., 50MB) and that MsgPack serialization is used to keep the wire format compact.

## 4. Position Distribution Efficiency
- If the `GET /api/eval/positions` endpoint is used (for non-NFS clients), positions should be served in batches matching the `CHUNK_SIZE` (50 games).
- The position data (25 bytes per state) is small enough that the server can serve the entire range for a chunk in a single MsgPack-encoded response.

## 5. Result Integrity
- **Version Pinning:** Ensure the `weights_version` returned during `checkout` is strictly enforced. If a client receives a chunk for a version it doesn't have, it must download and build the corresponding `FastWeights` before proceeding.
- **Side-Specific Stats:** Verify the aggregation logic correctly maps `az_is_white` from the chunk metadata to the final TensorBoard logs to prevent "side-swapping" errors in the equity/win% metrics.
