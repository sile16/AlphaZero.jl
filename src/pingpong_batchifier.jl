#####
##### Double-buffered (Ping-Pong) Batchifier
##### Overlaps GPU inference with worker submission for better utilization
#####

"""
Double-buffered batchifier that overlaps GPU inference with CPU work.

The key idea:
- Two buffers: "filling" and "processing"
- Workers submit to the filling buffer
- When filling buffer is ready, swap and start GPU processing
- While GPU processes one buffer, workers fill the other
- This creates overlap between GPU work and CPU work (tree traversal)
"""
module PingPongBatchifier

using ..AlphaZero: MCTS, Util, ProfUtils

export PingPongOracle, launch_pingpong_server

# Request sent by workers
struct Request
  query::Any
  answer_channel::Channel
end

"""
    launch_pingpong_server(f; num_workers, batch_size, min_batch=1)

Launch a double-buffered inference server.

# Arguments
- `f`: Neural network inference function (takes batch of inputs, returns batch of outputs)
- `num_workers`: Total number of workers that will submit queries
- `batch_size`: Target batch size for GPU inference
- `min_batch`: Minimum batch size before processing (default 1)

# Returns
A channel for submitting requests.
"""
function launch_pingpong_server(f; num_workers, batch_size, min_batch=1)
  @assert batch_size <= num_workers
  @assert min_batch <= batch_size

  request_channel = Channel{Union{Symbol, Request}}(num_workers * 2)

  Util.@tspawn_main Util.@printing_errors begin
    num_active = num_workers
    current_batch_size = batch_size

    # Two buffers for ping-pong
    buffer_a = Vector{Request}()
    buffer_b = Vector{Request}()

    # Which buffer is currently being filled
    filling = buffer_a
    processing = buffer_b

    # GPU task for async processing
    gpu_task = nothing
    gpu_results = nothing
    processing_buffer_snapshot = nothing

    while num_active > 0
      # Check if GPU task completed
      if gpu_task !== nothing && istaskdone(gpu_task)
        # Send results to workers that were waiting
        results = gpu_results
        for i in eachindex(processing_buffer_snapshot)
          put!(processing_buffer_snapshot[i].answer_channel, results[i])
        end
        gpu_task = nothing
        gpu_results = nothing
        processing_buffer_snapshot = nothing
      end

      # Try to get a request (non-blocking if we have work to do)
      req = nothing
      if isready(request_channel)
        req = take!(request_channel)
      elseif gpu_task === nothing && length(filling) >= min_batch
        # No GPU work and we have enough requests - process immediately
        req = nothing  # Skip waiting, process what we have
      elseif length(filling) > 0 && gpu_task === nothing
        # We have some pending requests and GPU is idle - wait briefly then process
        sleep(0.001)  # 1ms timeout
        if isready(request_channel)
          req = take!(request_channel)
        end
      else
        # Wait for a request
        req = take!(request_channel)
      end

      # Handle request
      if req == :done
        num_active -= 1
        if num_active < current_batch_size
          current_batch_size = max(num_active, 1)
        end
      elseif req !== nothing
        push!(filling, req)
      end

      # Decide if we should process the filling buffer
      should_process = false
      if gpu_task === nothing  # GPU is free
        if length(filling) >= current_batch_size
          should_process = true
        elseif length(filling) >= min_batch && num_active <= length(filling)
          # All active workers have submitted - process what we have
          should_process = true
        end
      end

      if should_process && length(filling) > 0
        # Swap buffers
        processing_buffer_snapshot = copy(filling)
        empty!(filling)

        # Start GPU processing asynchronously
        batch = [r.query for r in processing_buffer_snapshot]

        # Launch GPU inference
        gpu_task = @async begin
          ProfUtils.log_event(;
              name="Infer (batch size: $(length(batch)))",
              cat="GPU", pid=0, tid=0) do
            f(batch)
          end
        end

        # Store future results location
        gpu_results = nothing
        @async begin
          gpu_results = fetch(gpu_task)
        end
      end
    end

    # Process any remaining requests
    if length(filling) > 0
      batch = [r.query for r in filling]
      results = f(batch)
      for i in eachindex(filling)
        put!(filling[i].answer_channel, results[i])
      end
    end

    # Wait for final GPU task
    if gpu_task !== nothing
      results = fetch(gpu_task)
      for i in eachindex(processing_buffer_snapshot)
        put!(processing_buffer_snapshot[i].answer_channel, results[i])
      end
    end
  end

  return request_channel
end

"""
    client_done!(reqc)

Signal that a worker is done and won't send more queries.
"""
client_done!(reqc) = put!(reqc, :done)

"""
    PingPongOracle(reqc, preprocess=(x->x))

Create an oracle that uses the ping-pong batchifier.
"""
struct PingPongOracle{F}
  preprocess::F
  reqchan::Channel
  anschan::Channel

  function PingPongOracle(reqchan, preprocess=(x->x))
    return new{typeof(preprocess)}(preprocess, reqchan, Channel(1))
  end
end

function (oracle::PingPongOracle)(state)
  query = oracle.preprocess(state)
  ProfUtils.instant_event(
    name="Query", cat="Query", pid=0, tid=Threads.threadid())
  put!(oracle.reqchan, Request(query, oracle.anschan))
  answer = take!(oracle.anschan)
  return answer
end

end
