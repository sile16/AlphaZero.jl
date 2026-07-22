#!/usr/bin/env julia
"""
Rungs 6-7 of the MCTS validation staircase: the ERROR-RESPONSE CURVE.

Rungs 1-3 (test/test_mcts_identity_staircase.jl) proved the MCTS arithmetic is
EXACT (Q == input values to machine epsilon). A convergence sweep showed
finite-budget allocation noise with EXACT values (400 iters = knee). So any move error measured here is
attributable to INJECTED input error interacting with search — the machinery is
clean.

THEORY UNDER TEST (designed to falsify):
  1. FRESH value noise (re-drawn every evaluator call) is VARIANCE -> MCTS
     averages it out; regret decreases as sims rise (~1/sqrt(visits) per action).
  2. FROZEN value noise (deterministic per state, hash-seeded) is BIAS — like a
     trained NN's systematic error. At a depth-1 evaluator frontier, iterations
     CANNOT fix it: argmax over consistently-shifted constants stays shifted.
     Regret plateaus vs sims at a level set by eps and the local move-gap.
  3. POLICY-only error (bad priors, exact values) is cheap: corrected by a few
     hundred sims (uniform-prior sweep = the extreme datapoint, corrected by 400).
  4. Therefore for a real NN: sims buy correction of value VARIANCE and policy
     error, but only DEPTH past the frontier corrects value BIAS.

Rung 6: measure regret(injector, eps, sims) on ~400 fixed non-trivial bearoff
decision states. Rung 7: place a real race NN on the frozen-value curve.

Use a freshly validated, immutable race-position file and `--threads 8`.
"""

using ArgParse
function parse_cli()
    s = ArgParseSettings(description="MCTS error-response curves (rungs 6-7)", autofix_names=true)
    @add_arg_table! s begin
        "--num-states";      arg_type=Int;    default=400;  help="Non-trivial bearoff decision states in the fixed eval set"
        "--states-1600";     arg_type=Int;    default=200;  help="Trim to this many states for the 1600-sim rows (budget)"
        "--seed";            arg_type=Int;    default=42
        "--tau";             arg_type=Float64; default=0.02; help="Policy softmax temperature (raw points); noiseless prior sharpness"
        "--gap-max";         arg_type=Float64; default=0.5; help="Keep states with best-second exact gap < this (raw pts)"
        "--min-legal";       arg_type=Int;    default=3;    help="Keep states with >= this many legal moves"
        "--states-cache";    arg_type=String; default="/tmp/error_response_states.jls"
        "--positions-file";  arg_type=String; default=""
        "--checkpoint";      arg_type=String; default=get(ENV, "ALPHAZERO_CHECKPOINT",
            joinpath(dirname(@__DIR__), "sessions", "alphazero-server", "checkpoints", "race_iter_50.data"))
        "--width";           arg_type=Int;    default=256
        "--blocks";          arg_type=Int;    default=5
        "--nn-mcts-iters";   arg_type=Int;    default=600
        "--skip-nn";         action=:store_true
        "--out";             arg_type=String; default=joinpath(dirname(@__DIR__), "sessions", "validation", "error_response_results.jls")
    end
    return ArgParse.parse_args(s)
end

const ARGS_D = parse_cli()

using Random
using Serialization
using Statistics
using StaticArrays
using Printf

using AlphaZero
using AlphaZero: GI, MCTS, BatchedMCTS
import BackgammonNet
using BackgammonNet: BackgammonGame, BearoffK7, bearoff_best_move_value, bearoff_turn_value

ENV["BACKGAMMON_OBS_TYPE"] = "minimal_flat"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const GSPEC = GameSpec()

const TABLE = let
    dir = BackgammonNet.default_bearoff_k7_dir()
    isdir(dir) && isfile(joinpath(dir, "bearoff_k7_c14.bin")) ||
        error("k=7 table not found at $dir")
    println("Loading k=7 table from $dir ...")
    BearoffK7.BearoffTable(dir)
end

# ── Deterministic frozen-noise pool (hash -> standard normal) ────────────
# A fixed pool of N(0,1) draws indexed by a state/action hash gives a
# reproducible, allocation-free "g(hash)" per node. Collisions are negligible
# (2^20 slots, few distinct keys per state) and, if they occur, merely make two
# nodes share a frozen draw — still deterministic/frozen. Document: the hash
# includes p0,p1,dice,remaining_actions,current_player so distinct nodes (and
# distinct dice) get independent noise.
const NOISE_POOL = randn(MersenneTwister(0xBEEF5EED), 1 << 20)
const NOISE_MASK = UInt(length(NOISE_POOL) - 1)
@inline gnoise(x)::Float64 = @inbounds NOISE_POOL[(hash(x) & NOISE_MASK) + 1]

@inline state_key(g::BackgammonGame) =
    (g.p0, g.p1, g.dice[1], g.dice[2], g.remaining_actions, g.current_player)

# ── Exact evaluator (white-relative, /3 normalized), mirrors the MCTS
#    production bearoff evaluator in selfplay_client.jl ───────────────────
@inline function exact_white(game)::Union{Float64, Nothing}
    BearoffK7.is_bearoff_position(game.p0, game.p1) || return nothing
    if BackgammonNet.is_chance_node(game)
        eq = Float64(BackgammonNet.bearoff_equity(BearoffK7.lookup(TABLE, game))) / 3.0
        return game.current_player == 0 ? eq : -eq
    end
    acts = BackgammonNet.legal_actions(game)
    isempty(acts) && return nothing
    best = bearoff_best_move_value(TABLE, game) / 3.0
    return game.current_player == 0 ? best : -best
end

# ── Per-state precomputed data ───────────────────────────────────────────
struct StateData
    root::BackgammonGame
    mover::Int
    root_key::NTuple{6, Any}
    actions::Vector{Int}
    child_keys::Vector{NTuple{6, Any}}    # frontier-leaf key per action (immediate child)
    exact_raw::Vector{Float64}            # mover-perspective raw points per action
    best_raw::Float64
    gap::Float64                          # best - second-best (raw pts)
    optimal::Vector{Int}                  # action indices tying best (1e-9)
    is_doubles::Bool
    s15::Int                              # which side has all 15 on board (gammon-live), -1 none
end

function build_state_data(g::BackgammonGame)::StateData
    mover = Int(g.current_player)
    acts = BackgammonNet.legal_actions(g)
    exact = Float64[]
    ckeys = NTuple{6, Any}[]
    work = BackgammonNet.clone(g)
    for a in acts
        BackgammonNet.copy_state!(work, g)
        BackgammonNet.apply_action!(work, a)
        push!(ckeys, state_key(work))
        push!(exact, bearoff_turn_value(TABLE, work, mover))
    end
    best = maximum(exact)
    tol = 1e-9
    optimal = [i for i in eachindex(exact) if best - exact[i] <= tol]
    nonbest = [v for v in exact if best - v > tol]
    second = isempty(nonbest) ? best : maximum(nonbest)
    n_on0 = sum(Int((g.p0 >> (i * 4)) & 0xF) for i in 1:24)
    n_on1 = sum(Int((g.p1 >> (i * 4)) & 0xF) for i in 1:24)
    s15 = n_on0 == 15 ? (n_on1 == 15 ? -1 : 0) : (n_on1 == 15 ? 1 : -1)
    return StateData(BackgammonNet.clone(g), mover, state_key(g), collect(acts), ckeys,
                     exact, best, best - second, optimal, g.dice[1] == g.dice[2], s15)
end

# ── Fixed eval-state generation (rollout race starts to bearoff decisions) ─
function find_positions_file()
    path = ARGS_D["positions_file"]
    isempty(path) && error("--positions-file is required; use a validated immutable race artifact")
    isfile(path) || error("race positions file does not exist: $path")
    return path
end

function generate_states()
    cache = ARGS_D["states_cache"]
    want = ARGS_D["num_states"]
    if isfile(cache)
        sd = deserialize(cache)
        if length(sd) >= want
            println("Loaded $(length(sd)) cached eval states from $cache")
            return sd[1:want]
        end
    end
    src = find_positions_file()
    println("Generating eval states by rolling out race starts: $src")
    tuples = deserialize(src)
    rng = MersenneTwister(ARGS_D["seed"])
    seen = Set{NTuple{6, Any}}()
    out = StateData[]
    gapmax = ARGS_D["gap_max"]; minlegal = ARGS_D["min_legal"]
    for tup in shuffle(rng, tuples)
        length(out) >= want && break
        p0, p1, cp = tup[1], tup[2], tup[3]
        g = backgammon_game(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(cp), false, 0.0f0;
                            observation_type=:minimal_flat)
        for _ in 1:400
            g.terminated && break
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
                continue
            end
            if g.phase == BackgammonNet.PHASE_CHECKER_PLAY &&
               BearoffK7.is_bearoff_position(g.p0, g.p1)
                acts = BackgammonNet.legal_actions(g)
                if length(acts) >= minlegal
                    k = state_key(g)
                    if !(k in seen)
                        sd = build_state_data(g)
                        if sd.gap < gapmax && sd.gap > 1e-9   # non-trivial, non-forced-tie
                            push!(seen, k); push!(out, sd)
                            length(out) >= want && break
                        end
                    end
                end
            end
            acts = BackgammonNet.legal_actions(g)
            isempty(acts) && break
            BackgammonNet.apply_action!(g, rand(rng, acts))
        end
    end
    length(out) >= want || @warn "Only found $(length(out)) / $want states"
    mkpath(dirname(cache))
    serialize(cache, out)
    println("Cached $(length(out)) eval states to $cache")
    return out
end

# ── Injectable evaluator + oracle factories ──────────────────────────────
# injector ∈ (:exact, :vfrozen, :vfresh, :pfrozen, :both)
function make_evaluator(sd::StateData, injector::Symbol, eps::Float64)
    rk = sd.root_key
    scale = eps / 3.0
    policy_mode = injector === :pfrozen || injector === :both
    value_mode  = injector === :vfrozen || injector === :both
    fresh_mode  = injector === :vfresh
    return function(game_env)
        game = game_env.game
        BearoffK7.is_bearoff_position(game.p0, game.p1) || return nothing
        k = state_key(game)
        if k == rk
            # Root: policy modes defer to the oracle (so noisy priors take effect);
            # value/exact modes score the root exactly (no self-noise) — reproduces
            # the convergence-sweep baseline at eps=0.
            policy_mode && return nothing
            return exact_white(game)
        end
        base = exact_white(game)
        base === nothing && return nothing
        if value_mode
            return base + scale * gnoise(k)
        elseif fresh_mode
            return base + scale * randn()
        else
            return base
        end
    end
end

@inline function _softmax!(logits::Vector{Float32})
    m = maximum(logits)
    s = 0.0f0
    @inbounds for i in eachindex(logits); logits[i] = exp(logits[i] - m); s += logits[i]; end
    @inbounds for i in eachindex(logits); logits[i] /= s; end
    return logits
end

function make_oracle(sd::StateData, injector::Symbol, eps::Float64, tau::Float64)
    rk = sd.root_key
    policy_mode = injector === :pfrozen || injector === :both
    # Precompute noisy logits over the root's legal actions (raw points).
    act2logit = Dict{Int, Float64}()
    if policy_mode
        for (i, a) in enumerate(sd.actions)
            act2logit[a] = (sd.exact_raw[i] + eps * gnoise((rk, a))) / tau
        end
    end
    return function(state)
        acts = GI.available_actions(GSPEC, state)
        n = max(1, length(acts))
        if policy_mode && state_key(state) == rk
            logits = Float32[get(act2logit, a, -1f9) for a in acts]
            _softmax!(logits)
            return (logits, 0.0f0)
        end
        return (fill(Float32(1.0 / n), n), 0.0f0)
    end
end

const BASE_PARAMS = (cpuct=2.0, gamma=1.0)

function run_mcts_choice(sd::StateData, injector::Symbol, eps::Float64, sims::Int, tau::Float64)
    ev = make_evaluator(sd, injector, eps)
    oracle = make_oracle(sd, injector, eps, tau)
    params = MctsParams(num_iters_per_turn=sims, cpuct=BASE_PARAMS.cpuct, gamma=BASE_PARAMS.gamma,
        temperature=ConstSchedule(0.0), dirichlet_noise_ϵ=0.0, dirichlet_noise_α=0.3,
        prior_temperature=1.0, chance_mode=:passthrough)
    player = BatchedMCTS.BatchedMctsPlayer(GSPEC, oracle, params; batch_size=8, bearoff_evaluator=ev)
    genv = GameEnv(BackgammonNet.clone(sd.root), MersenneTwister(hash(sd.root_key) % typemax(Int)))
    BatchedMCTS.batched_explore!(player.benv, genv, sims)
    return _choice_regret(player, sd)
end

# Read a player's tree and return regret by BOTH selection rules:
#   visit-argmax (standard AlphaZero temperature-0 selection)  -> primary
#   Q-argmax (what the tree's VALUES say)                       -> isolates whether
#     residual regret is a value/search-value error vs a visit-allocation/prior artifact
function _choice_regret(pl, sd::StateData)
    info = pl.benv.env.tree[BackgammonNet.clone(sd.root)]
    vk = 0; vN = -1; vQ = -Inf     # visit-argmax (ties -> higher Q)
    qk = 0; qQ = -Inf              # Q-argmax
    for k in eachindex(info.stats)
        s = info.stats[k]; s.N < 1 && continue
        q = s.W / s.N
        if s.N > vN || (s.N == vN && q > vQ); vN = s.N; vQ = q; vk = k; end
        if q > qQ; qQ = q; qk = k; end
    end
    vk == 0 && return (0.0, false, 0.0, false)
    _reg(k) = begin
        ai = findfirst(==(info.actions[k]), sd.actions)
        ai === nothing ? 0.0 : sd.best_raw - sd.exact_raw[ai]
    end
    rv = _reg(vk); rq = _reg(qk)
    return (max(0.0, rv), rv > 1e-9, max(0.0, rq), rq > 1e-9)
end

# Zero-sim (no search) anchor for a frozen-VALUE injector: 1-ply argmax over the
# corrupted evaluator values. Converged MCTS Q(a) = exact_norm[a] + s*(eps/3)*g(childkey),
# s = (mover==0 ? +1 : -1) (derived: white->mover sign is constant across actions).
function zero_sim_frozen(sd::StateData, eps::Float64)
    s = sd.mover == 0 ? 1.0 : -1.0
    scale = eps / 3.0
    bestv = -Inf; bestai = 1
    for i in eachindex(sd.actions)
        # terminal children get NO evaluator noise (scored via reward path)
        corrupt = sd.exact_raw[i] / 3.0 + s * scale * gnoise(sd.child_keys[i])
        if corrupt > bestv; bestv = corrupt; bestai = i; end
    end
    regret = sd.best_raw - sd.exact_raw[bestai]
    return max(0.0, regret)
end

function run_cell(states::Vector{StateData}, injector::Symbol, eps::Float64, sims::Int, tau::Float64)
    n = length(states)
    reg = zeros(Float64, n); wrong = falses(n)
    qreg = zeros(Float64, n); qwrong = falses(n)
    Threads.@threads for i in 1:n
        rv, wv, rq, wq = run_mcts_choice(states[i], injector, eps, sims, tau)
        reg[i] = rv; wrong[i] = wv; qreg[i] = rq; qwrong[i] = wq
    end
    return (mean_regret=mean(reg), p_wrong=mean(wrong),
            q_regret=mean(qreg), q_wrong=mean(qwrong))
end

# ════════════════════════════════════════════════════════════════════════
#  RUNG 6 — the curves
# ════════════════════════════════════════════════════════════════════════
function rung6(states)
    tau = ARGS_D["tau"]
    epsgrid = [0.01, 0.03, 0.1, 0.3]
    simsgrid = [30, 100, 400, 1600]
    injectors = [:vfrozen, :vfresh, :pfrozen, :both]
    n1600 = min(ARGS_D["states_1600"], length(states))

    # Gap distribution report
    gaps = [sd.gap for sd in states]
    nacts = [length(sd.actions) for sd in states]
    ndoub = count(sd -> sd.is_doubles, states)
    ng0 = count(sd -> sd.mover == 0, states)
    n15 = count(sd -> sd.s15 >= 0, states)
    println("\n" * "="^72)
    println("EVAL SET: $(length(states)) non-trivial bearoff decision states")
    @printf("  gap (raw pts): min=%.4f  p25=%.4f  median=%.4f  p75=%.4f  max=%.4f  mean=%.4f\n",
            minimum(gaps), quantile(gaps, 0.25), median(gaps), quantile(gaps, 0.75), maximum(gaps), mean(gaps))
    @printf("  legal moves: min=%d median=%.1f max=%d | doubles=%d (%.0f%%) | mover=0: %d | gammon-live: %d\n",
            minimum(nacts), median(nacts), maximum(nacts), ndoub, 100ndoub/length(states), ng0, n15)

    # Noiseless policy sharpness (document tau): mean top-1 prior of the exact softmax
    top1p = Float64[]
    for sd in states
        logits = Float32[sd.exact_raw[i] / tau for i in eachindex(sd.actions)]
        _softmax!(logits)
        push!(top1p, maximum(logits))
    end
    @printf("  policy tau=%.3f -> noiseless softmax mean top-1 prior = %.3f (sharpness)\n", tau, mean(top1p))

    # eps=0 baseline (exact evaluator, uniform priors) — should reproduce the sweep shape
    println("\n--- BASELINE eps=0 (exact evaluator, uniform priors) ---")
    baseline = Dict{Int, NamedTuple}()
    for sims in simsgrid
        ss = sims == 1600 ? states[1:n1600] : states
        c = run_cell(ss, :exact, 0.0, sims, tau)
        baseline[sims] = c
        @printf("  sims=%-4d  mean_regret=%.5f  P(wrong)=%.3f  (n=%d)\n", sims, c.mean_regret, c.p_wrong, length(ss))
    end

    # Zero-sim (no-search) frozen-value anchors per eps
    println("\n--- ZERO-SIM anchors: 1-ply argmax over corrupted values (frozen value noise) ---")
    zerosim = Dict{Float64, Float64}()
    for eps in epsgrid
        z = mean(zero_sim_frozen(sd, eps) for sd in states)
        zerosim[eps] = z
        @printf("  eps=%.2f  zero-sim mean_regret=%.5f\n", eps, z)
    end

    # Full grid: visit-argmax regret (primary), plus Q-argmax regret (isolates
    # whether residual regret is a value error or a visit-allocation/prior artifact).
    results = Dict{Tuple{Symbol, Float64, Int}, NamedTuple}()
    for injector in injectors
        println("\n--- injector=$(injector)  [visit-argmax regret] ---")
        @printf("  %-8s", "eps\\sims")
        for sims in simsgrid; @printf("  %10d", sims); end
        println("   | zero-sim")
        for eps in epsgrid
            for sims in simsgrid
                ss = sims == 1600 ? states[1:n1600] : states
                results[(injector, eps, sims)] = run_cell(ss, injector, eps, sims, tau)
            end
            @printf("  %-8.2f", eps)
            for sims in simsgrid; @printf("  %10.5f", results[(injector, eps, sims)].mean_regret); end
            za = (injector in (:vfrozen, :both)) ? @sprintf("%.5f", zerosim[eps]) : "   n/a"
            println("   | $za")
        end
        # Q-argmax companion (same cells, selection by max Q instead of max visits)
        println("  ($(injector) Q-argmax regret)")
        for eps in epsgrid
            @printf("  %-8.2f", eps)
            for sims in simsgrid; @printf("  %10.5f", results[(injector, eps, sims)].q_regret); end
            println()
        end
    end
    return (baseline=baseline, zerosim=zerosim, results=results,
            epsgrid=epsgrid, simsgrid=simsgrid, injectors=injectors, tau=tau,
            gaps=gaps, n1600=n1600)
end

# ════════════════════════════════════════════════════════════════════════
#  RUNG 7 — place the real NN on the curve
# ════════════════════════════════════════════════════════════════════════
function rung7(states, rung6res)
    ckpt = ARGS_D["checkpoint"]
    if ARGS_D["skip_nn"] || !isfile(ckpt)
        @warn "Skipping Rung 7 (checkpoint missing or --skip-nn): $ckpt"
        return nothing
    end
    println("\n" * "="^72)
    println("RUNG 7 — placing race NN on the frozen-value curve: $ckpt")

    # NN oracle (value)
    @eval begin
        using AlphaZero: Network, FluxLib
        import Flux
        using AlphaZero.BackgammonInference
    end
    Base.invokelatest(rung7_impl, states, rung6res, ckpt)
end

function rung7_impl(states, rung6res, ckpt)
    width = ARGS_D["width"]; blocks = ARGS_D["blocks"]
    STATE_DIM = GI.state_dim(GSPEC)[1]; NUM_ACTIONS = GI.num_actions(GSPEC)
    cfg = AlphaZero.BackgammonInference.OracleConfig(STATE_DIM, NUM_ACTIONS, GSPEC;
        vectorize_state! = vectorize_state_into!)
    network = AlphaZero.FluxLib.FCResNetMultiHead(GSPEC,
        AlphaZero.FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    AlphaZero.FluxLib.load_weights(ckpt, network)
    network = AlphaZero.Flux.cpu(network)
    backend = AlphaZero.BackgammonInference.resolve_cpu_backend("auto")
    so, vo = AlphaZero.BackgammonInference.make_cpu_oracles(backend, network, cfg;
        batch_size=64, nslots=Threads.nthreads())

    # NN white-relative /3 value at a bearoff node (child.cp-perspective from vo, -> white)
    nn_white(g) = let v = Float64(vo([g])[1][2]); g.current_player == 0 ? v : -v; end

    # 1) NN value error vs exact on the FRONTIER children (the injection points)
    #    and on the root states. mover-perspective, /3 normalized -> report raw pts.
    child_err = Float64[]      # NN - exact, mover perspective /3
    common = Float64[]         # per-root mean sibling error (correlated component)
    diffvar = Float64[]        # per-root within-sibling error variance (differential)
    sib_all = Float64[]        # all sibling errors pooled (for total variance)
    root_err = Float64[]
    for sd in states
        # root NN value error
        rv = nn_white(sd.root)                # white-rel /3
        rexact = (sd.mover == 0 ? 1.0 : -1.0) * bearoff_best_move_value(TABLE, sd.root) / 3.0
        push!(root_err, rv - rexact)
        # per-child errors (mover perspective /3)
        errs = Float64[]
        work = BackgammonNet.clone(sd.root)
        for (i, a) in enumerate(sd.actions)
            BackgammonNet.copy_state!(work, sd.root)
            BackgammonNet.apply_action!(work, a)
            # NN value of child in MOVER perspective /3:
            # nn_white(child) is white-rel; mover-rel = (mover==0? +: -) white-rel
            cw = nn_white(work)
            cm = (sd.mover == 0 ? 1.0 : -1.0) * cw
            e = cm - sd.exact_raw[i] / 3.0
            push!(errs, e); push!(child_err, e); push!(sib_all, e)
        end
        push!(common, mean(errs))
        push!(diffvar, length(errs) > 1 ? var(errs) : 0.0)
    end
    rmse_norm = sqrt(mean(child_err .^ 2))
    mae_norm = mean(abs.(child_err))
    bias_norm = mean(child_err)
    rmse_raw = rmse_norm * 3; mae_raw = mae_norm * 3; bias_raw = bias_norm * 3
    # sibling-error decomposition: e_i = c_root + d_i
    var_total = var(sib_all)
    var_common = var(common)
    diff_rmse_norm = sqrt(mean(diffvar))     # RMS of within-root deviation
    diff_rmse_raw = diff_rmse_norm * 3
    frac_common = var_total > 0 ? var_common / var_total : NaN
    # mean pairwise sibling correlation ~ fraction of variance that is common
    println("\nNN value error vs exact table (frontier children, /3 scale -> raw pts):")
    @printf("  n_children=%d  MAE=%.4f  RMSE=%.4f  bias=%+.4f  (raw pts)\n",
            length(child_err), mae_raw, rmse_raw, bias_raw)
    @printf("  root-state value error: MAE=%.4f RMSE=%.4f bias=%+.4f (raw pts)\n",
            mean(abs.(root_err))*3, sqrt(mean(root_err.^2))*3, mean(root_err)*3)
    @printf("  SIBLING correlation: total err var=%.5f  common(across-root) var=%.5f  frac_common=%.3f\n",
            var_total, var_common, frac_common)
    @printf("  differential (within-root) RMSE=%.4f raw pts  (this is the regret-driving component)\n",
            diff_rmse_raw)

    # 2) Predict regret from the frozen-value curve at eps ~ NN RMSE (and eps ~ diff RMSE)
    #    Interpolate the frozen-value curve at sims=600 (between 400 and 1600).
    function curve_regret_at(injector, eps_target, sims)
        egrid = rung6res.epsgrid
        # nearest-eps and linear interp in log-eps
        cell(e) = rung6res.results[(injector, e, sims)].mean_regret
        if eps_target <= egrid[1]; return cell(egrid[1]); end
        if eps_target >= egrid[end]; return cell(egrid[end]); end
        j = findlast(e -> e <= eps_target, egrid)
        e0, e1 = egrid[j], egrid[j+1]
        t = (log(eps_target) - log(e0)) / (log(e1) - log(e0))
        return cell(e0) * (1 - t) + cell(e1) * t
    end
    # sims=600 not on grid; interpolate frozen-value regret between 400 and 1600 (log-sims)
    function frozen_regret_600(eps_target)
        r400 = curve_regret_at(:vfrozen, eps_target, 400)
        r1600 = curve_regret_at(:vfrozen, eps_target, 1600)
        t = (log(600) - log(400)) / (log(1600) - log(400))
        return r400 * (1 - t) + r1600 * t
    end
    pred_total = frozen_regret_600(rmse_raw)
    pred_diff = frozen_regret_600(diff_rmse_raw)
    println("\nPREDICTED regret @600 sims from frozen-value curve:")
    @printf("  using eps=RMSE=%.4f  -> predicted regret=%.5f\n", rmse_raw, pred_total)
    @printf("  using eps=diffRMSE=%.4f -> predicted regret=%.5f (iid-sibling-corrected)\n", diff_rmse_raw, pred_diff)

    # 3) Measure reality
    #  (a) NN-as-evaluator at bearoff frontier (apples-to-apples with frozen curve),
    #      uniform root priors, sims=600.
    sims = ARGS_D["nn_mcts_iters"]
    function make_nn_evaluator(rk)
        return function(game_env)
            game = game_env.game
            BearoffK7.is_bearoff_position(game.p0, game.p1) || return nothing
            state_key(game) == rk && return nothing   # root -> oracle uniform priors
            BackgammonNet.is_chance_node(game) && return nn_white(game)
            isempty(BackgammonNet.legal_actions(game)) && return nothing
            return nn_white(game)
        end
    end
    uniform_oracle(state) = let n = max(1, length(GI.available_actions(GSPEC, state)))
        (fill(Float32(1.0 / n), n), 0.0f0)
    end
    n = length(states)
    reg_nneval = zeros(Float64, n); wrong_nneval = falses(n)
    reg_nnoracle = zeros(Float64, n); wrong_nnoracle = falses(n)
    reg_1ply = zeros(Float64, n); wrong_1ply = falses(n)
    params = MctsParams(num_iters_per_turn=sims, cpuct=BASE_PARAMS.cpuct, gamma=BASE_PARAMS.gamma,
        temperature=ConstSchedule(0.0), dirichlet_noise_ϵ=0.0, dirichlet_noise_α=0.3,
        prior_temperature=1.0, chance_mode=:passthrough)
    Threads.@threads for i in 1:n
        sd = states[i]
        # (a) NN-as-evaluator, uniform root priors
        ev = make_nn_evaluator(sd.root_key)
        pl = BatchedMCTS.BatchedMctsPlayer(GSPEC, uniform_oracle, params; batch_size=8, bearoff_evaluator=ev)
        genv = GameEnv(BackgammonNet.clone(sd.root), MersenneTwister(hash(sd.root_key) % typemax(Int)))
        BatchedMCTS.batched_explore!(pl.benv, genv, sims)
        reg_nneval[i], wrong_nneval[i], _, _ = _choice_regret(pl, sd)
        # (b) NN-as-oracle (no bearoff evaluator) — real self-play usage
        pl2 = BatchedMCTS.BatchedMctsPlayer(GSPEC, so, params; batch_size=8, bearoff_evaluator=nothing)
        genv2 = GameEnv(BackgammonNet.clone(sd.root), MersenneTwister(hash(sd.root_key) % typemax(Int)))
        BatchedMCTS.batched_explore!(pl2.benv, genv2, sims)
        reg_nnoracle[i], wrong_nnoracle[i], _, _ = _choice_regret(pl2, sd)
        # (c) raw NN 1-ply: argmax over NN child values (mover perspective), zero search
        bestv = -Inf; bestai = 1
        work = BackgammonNet.clone(sd.root)
        for (j, a) in enumerate(sd.actions)
            BackgammonNet.copy_state!(work, sd.root)
            BackgammonNet.apply_action!(work, a)
            cm = (sd.mover == 0 ? 1.0 : -1.0) * nn_white(work)
            if cm > bestv; bestv = cm; bestai = j; end
        end
        reg_1ply[i] = max(0.0, sd.best_raw - sd.exact_raw[bestai])
        wrong_1ply[i] = (sd.best_raw - sd.exact_raw[bestai]) > 1e-9
    end
    println("\nMEASURED regret (real NN):")
    @printf("  NN-as-evaluator @%d sims (uniform priors, frontier=NN):  regret=%.5f  P(wrong)=%.3f\n",
            sims, mean(reg_nneval), mean(wrong_nneval))
    @printf("  NN-as-oracle    @%d sims (no bearoff eval, real usage):   regret=%.5f  P(wrong)=%.3f\n",
            sims, mean(reg_nnoracle), mean(wrong_nnoracle))
    @printf("  raw NN 1-ply (zero search):                              regret=%.5f  P(wrong)=%.3f\n",
            mean(reg_1ply), mean(wrong_1ply))

    println("\nPREDICTED vs ACTUAL @600 sims (NN-as-evaluator is the apples-to-apples row):")
    @printf("  predicted (eps=RMSE=%.4f):      %.5f\n", rmse_raw, pred_total)
    @printf("  predicted (eps=diffRMSE=%.4f):  %.5f\n", diff_rmse_raw, pred_diff)
    @printf("  actual   (NN-as-evaluator):        %.5f\n", mean(reg_nneval))
    ratio_t = mean(reg_nneval) / max(pred_total, 1e-9)
    ratio_d = mean(reg_nneval) / max(pred_diff, 1e-9)
    @printf("  actual/pred(total)=%.2fx   actual/pred(diff)=%.2fx\n", ratio_t, ratio_d)

    return (rmse_raw=rmse_raw, mae_raw=mae_raw, bias_raw=bias_raw,
            diff_rmse_raw=diff_rmse_raw, frac_common=frac_common,
            pred_total=pred_total, pred_diff=pred_diff,
            actual_nneval=mean(reg_nneval), actual_nnoracle=mean(reg_nnoracle),
            actual_1ply=mean(reg_1ply))
end

function main()
    println("MCTS error-response curves (rungs 6-7)")
    println("  threads=$(Threads.nthreads())  seed=$(ARGS_D["seed"])")
    states = generate_states()
    r6 = rung6(states)
    r7 = rung7(states, r6)
    out = ARGS_D["out"]
    mkpath(dirname(out))
    serialize(out, (rung6=r6, rung7=r7))
    println("\nSerialized results to $out")
end

main()
