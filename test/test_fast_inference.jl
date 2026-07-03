#####
##### Regression tests for FastInference (src/inference/fast_weights.jl)
#####
# Guards the allocation-free CPU forward pass used by selfplay_client.jl:
# - _gemm_bias! against a Float64 reference on shapes hitting every remainder
#   path (k-tile tail, 4-step k-unroll tail, 4-column tail, single columns)
# - fast_forward_normalized! against the Flux FCResNetMultiHead forward
# - refresh_fast_weights! equivalence with a fresh extraction
# - zero allocations (per-move allocations destroy threaded selfplay throughput)

using Test
using Random
using AlphaZero
using AlphaZero.GI
import AlphaZero.Network
import AlphaZero.FluxLib
import Flux

using AlphaZero.FastInference

module FastInfTestGame
  import AlphaZero.GI

  struct GameSpec <: GI.AbstractGameSpec end

  const NUM_ACTIONS = 11
  const STATE_DIM = 9

  mutable struct GameEnv <: GI.AbstractGameEnv
    state :: Vector{Float32}
    terminated :: Bool
  end

  GI.init(::GameSpec) = GameEnv(zeros(Float32, STATE_DIM), false)
  GI.spec(::GameEnv) = GameSpec()
  GI.two_players(::GameSpec) = true
  GI.actions(::GameSpec) = collect(1:NUM_ACTIONS)
  GI.current_state(g::GameEnv) = g.state
  GI.game_terminated(g::GameEnv) = g.terminated
  GI.white_playing(g::GameEnv) = true
  GI.actions_mask(g::GameEnv) = trues(NUM_ACTIONS)
  GI.vectorize_state(::GameSpec, state) = Float32.(state)
end

# Randomize LayerNorm scale/bias everywhere so extraction bugs cannot hide
# behind the default scale=1, bias=0.
function randomize_layernorms!(rng, parts...)
  for part in parts
    part === identity && continue
    for l in Flux.modules(part)
      if l isa Flux.LayerNorm
        l.diag.scale .= 0.5f0 .+ rand(rng, Float32, size(l.diag.scale))
        l.diag.bias .= 0.2f0 .* randn(rng, Float32, size(l.diag.bias))
      end
    end
  end
end

function make_net(rng)
  gspec = FastInfTestGame.GameSpec()
  # depth_vhead = 2 with a shared trunk yields one Dense+LN in vhead_trunk —
  # the only shape extract_fast_weights supports (and what production uses).
  hyper = FluxLib.FCResNetMultiHeadHP(
    width = 32, num_blocks = 2, depth_phead = 2, depth_vhead = 2)
  nn = FluxLib.FCResNetMultiHead(gspec, hyper)
  randomize_layernorms!(rng, nn.common, nn.vhead_trunk, nn.phead)
  return nn
end

# Masked renormalization of the Flux policy output (phead ends with softmax,
# unmasked) — mathematically identical to FastInference's masked softmax.
function flux_reference(nn, X, A)
  p, v = Network.forward(nn, X)
  pm = p .* A
  pm = pm ./ sum(pm; dims = 1)
  return pm, vec(v)
end

function random_mask(rng, nact, n)
  A = Float32.(rand(rng, nact, n) .> 0.4)
  for j in 1:n
    A[rand(rng, 1:nact), j] = 1.0f0  # at least one legal action per column
  end
  return A
end

gemm_alloc(C, W, X, b, n) = @allocated FastInference._gemm_bias!(C, W, X, b, n)
fwd_alloc(fw, fb, X, A, n) = @allocated fast_forward_normalized!(fw, fb, X, A, n)

@testset "FastInference regressions" begin
  rng = MersenneTwister(0)

  @testset "_gemm_bias! matches Float64 reference on remainder shapes" begin
    for m in (1, 5, 32, 129), k in (1, 3, 63, 64, 65, 67, 128), n in (1, 3, 4, 5, 8)
      W = randn(rng, Float32, m, k) .* 0.1f0
      X = randn(rng, Float32, k, n) .* 0.5f0
      b = randn(rng, Float32, m) .* 0.1f0
      R = Float64.(W) * Float64.(X) .+ Float64.(b)
      C = fill(NaN32, m, n)  # catch any unwritten element
      FastInference._gemm_bias!(C, W, X, b, n)
      @test !any(isnan, C)
      scale = max(maximum(abs.(R)), 1.0)
      @test maximum(abs.(Float64.(C) .- R)) / scale < 1e-5
    end
  end

  @testset "_gemm_bias! is allocation-free" begin
    W = randn(rng, Float32, 128, 344)
    X = randn(rng, Float32, 344, 50)
    b = randn(rng, Float32, 128)
    C = zeros(Float32, 128, 50)
    gemm_alloc(C, W, X, b, 50)  # compile
    @test gemm_alloc(C, W, X, b, 50) == 0
  end

  @testset "fast_forward_normalized! matches Flux forward" begin
    nn = make_net(rng)
    fw = extract_fast_weights(nn)
    nact = FastInfTestGame.NUM_ACTIONS
    fb = FastBuffers(32, nact, 8)
    for n in (1, 4, 7)  # hit the 4-column kernel path and both tails
      X = randn(rng, Float32, FastInfTestGame.STATE_DIM, n)
      A = random_mask(rng, nact, n)
      P, V, _ = fast_forward_normalized!(fw, fb, X, A, n)
      P_ref, V_ref = flux_reference(nn, X, A)
      @test maximum(abs.(P[:, 1:n] .- P_ref)) < 1e-4
      @test maximum(abs.(V[1:n] .- V_ref)) < 1e-4
    end
  end

  @testset "policy invariants" begin
    nn = make_net(rng)
    fw = extract_fast_weights(nn)
    nact = FastInfTestGame.NUM_ACTIONS
    fb = FastBuffers(32, nact, 8)
    n = 7
    X = randn(rng, Float32, FastInfTestGame.STATE_DIM, n)
    A = random_mask(rng, nact, n)
    P, V, _ = fast_forward_normalized!(fw, fb, X, A, n)
    for j in 1:n
      @test sum(P[:, j]) ≈ 1.0f0 atol = 1e-4
      @test all(P[i, j] == 0.0f0 for i in 1:nact if A[i, j] == 0.0f0)
      @test -1.0f0 <= V[j] <= 1.0f0
    end
  end

  @testset "fast_forward_normalized! is allocation-free" begin
    nn = make_net(rng)
    fw = extract_fast_weights(nn)
    fb = FastBuffers(32, FastInfTestGame.NUM_ACTIONS, 8)
    X = randn(rng, Float32, FastInfTestGame.STATE_DIM, 7)
    A = random_mask(rng, FastInfTestGame.NUM_ACTIONS, 7)
    fwd_alloc(fw, fb, X, A, 7)  # compile
    @test fwd_alloc(fw, fb, X, A, 7) == 0
  end

  @testset "refresh_fast_weights! matches fresh extraction" begin
    nn1 = make_net(rng)
    nn2 = make_net(rng)
    fw = extract_fast_weights(nn1)
    refresh_fast_weights!(fw, nn2)
    fw2 = extract_fast_weights(nn2)
    fb_a = FastBuffers(32, FastInfTestGame.NUM_ACTIONS, 8)
    fb_b = FastBuffers(32, FastInfTestGame.NUM_ACTIONS, 8)
    X = randn(rng, Float32, FastInfTestGame.STATE_DIM, 5)
    A = random_mask(rng, FastInfTestGame.NUM_ACTIONS, 5)
    P_a, V_a, _ = fast_forward_normalized!(fw, fb_a, X, A, 5)
    P_b, V_b, _ = fast_forward_normalized!(fw2, fb_b, X, A, 5)
    @test P_a[:, 1:5] == P_b[:, 1:5]
    @test V_a[1:5] == V_b[1:5]
  end
end
