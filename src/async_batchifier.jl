#####
##### Async Batchifier - Non-blocking batched inference
#####

"""
Async batchifier that processes queries eagerly without waiting for full batches.

Key improvements over original batchifier:
1. Timeout-based batching: Process queries after timeout even if batch not full
2. Eager processing: Start inference as soon as min_batch queries available
3. Better GPU utilization: More frequent, smaller batches keep GPU busy

The tradeoff is potentially smaller batch sizes, but this is offset by:
- Reduced worker idle time (workers don't wait for each other)
- Better GPU utilization (more frequent inference calls)
- Lower latency (queries processed sooner)
"""
module AsyncBatchifier

using ..AlphaZero: MCTS, Util, ProfUtils

export AsyncBatchedOracle, launch_async_server

const DEFAULT_TIMEOUT_NS = 1_000_000  # 1ms in nanoseconds
const DEFAULT_MIN_BATCH = 1

"""
    launch_async_server(f; num_workers, batch_size, min_batch=1, timeout_ns=1_000_000)

Launch an async inference server that processes batches eagerly.

# Arguments
- `f`: Function to evaluate batches (e.g., neural network forward pass)
- `num_workers`: Number of workers that will query the server
- `batch_size`: Maximum batch size (for GPU memory management)
- `min_batch`: Minimum queries before processing (default: 1)
- `timeout_ns`: Nanoseconds to wait for more queries before processing (default: 1ms)

# Behavior
The server processes a batch when ANY of these conditions are met:
1. `pending >= batch_size` (batch is full)
2. `pending >= min_batch` AND timeout has expired
3. All workers are done

This ensures:
- Workers don't wait unnecessarily for each other
- GPU stays busy with frequent smaller batches
- Memory limits are respected via max batch_size
"""
function launch_async_server(f;
    num_workers,
    batch_size,
    min_batch=DEFAULT_MIN_BATCH,
    timeout_ns=DEFAULT_TIMEOUT_NS)

  @assert min_batch >= 1 "min_batch must be at least 1"
  @assert batch_size >= min_batch "batch_size must be >= min_batch"
  @assert batch_size <= num_workers "batch_size must be <= num_workers"

  channel = Channel(num_workers * 2)  # Larger buffer for async operation

  Util.@tspawn_main Util.@printing_errors begin
    num_active = num_workers
    pending = []
    last_process_time = time_ns()
    effective_batch_size = batch_size

    while num_active > 0
      # Non-blocking: collect all available queries
      while isready(channel)
        req = take!(channel)
        if req == :done
          num_active -= 1
          if num_active < effective_batch_size
            effective_batch_size = max(num_active, min_batch)
          end
        else
          push!(pending, req)
        end
      end

      # Check if we should process
      current_time = time_ns()
      time_elapsed = current_time - last_process_time
      should_process = false

      if length(pending) >= effective_batch_size
        # Batch is full
        should_process = true
      elseif length(pending) >= min_batch && time_elapsed >= timeout_ns
        # Have minimum queries and timeout expired
        should_process = true
      elseif num_active == 0 && length(pending) > 0
        # All workers done, process remaining
        should_process = true
      end

      if should_process && length(pending) > 0
        # Process up to batch_size queries
        to_process = pending[1:min(length(pending), batch_size)]
        remaining = length(pending) > batch_size ? pending[batch_size+1:end] : []

        batch = [p.query for p in to_process]
        results = ProfUtils.log_event(;
            name="AsyncInfer (batch: $(length(batch)))",
            cat="GPU", pid=0, tid=0) do
          f(batch)
        end

        for i in eachindex(to_process)
          put!(to_process[i].answer_channel, results[i])
        end

        pending = remaining
        last_process_time = time_ns()
      elseif length(pending) == 0 && num_active > 0
        # No pending queries, wait for next one (blocking)
        req = take!(channel)
        if req == :done
          num_active -= 1
          if num_active < effective_batch_size
            effective_batch_size = max(num_active, min_batch)
          end
        else
          push!(pending, req)
          last_process_time = time_ns()  # Reset timer when first query arrives
        end
      else
        # Have some queries but not ready to process yet
        # Sleep briefly to prevent busy-waiting while allowing timeout to progress
        sleep(0.0001)  # 0.1ms sleep
      end
    end
  end

  return channel
end

"""
    client_done!(reqc)

Signal that a worker is done and won't send more queries.
"""
client_done!(reqc) = put!(reqc, :done)

"""
    AsyncBatchedOracle(reqc, preprocess=(x->x))

Oracle that sends queries to an async inference server.

Unlike the synchronous BatchedOracle, this oracle's queries are processed
more eagerly, reducing wait time for workers.
"""
struct AsyncBatchedOracle{F}
  preprocess :: F
  reqchan :: Channel
  anschan :: Channel
  function AsyncBatchedOracle(reqchan, preprocess=(x->x))
    return new{typeof(preprocess)}(preprocess, reqchan, Channel(1))
  end
end

function (oracle::AsyncBatchedOracle)(state)
  query = oracle.preprocess(state)
  ProfUtils.instant_event(
    name="AsyncQuery", cat="Query", pid=0, tid=Threads.threadid())
  put!(oracle.reqchan, (query=query, answer_channel=oracle.anschan))
  answer = take!(oracle.anschan)
  return answer
end

end # module
