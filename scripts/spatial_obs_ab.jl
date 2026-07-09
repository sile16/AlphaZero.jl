#!/usr/bin/env julia
"""
spatial_obs_ab.jl — DEFINITIVE supervised A/B: does a CONV/SPATIAL architecture over
a spatial board observation learn a better raw POLICY than the flat MLP over a flat
observation, on IDENTICAL supervised data?

Two arms, trained from the SAME 30k contact bootstrap dataset with the SAME optimiser/
epochs/LR/loss, differing ONLY in (observation, architecture):

  FLAT arm (control):  obs = :min_plus_flat (350),  arch = FC pre-act ResNet (MLP)
  CONV arm (test):     obs = :full (76x1x26),        arch = AlphaGo-Zero conv ResNet

Both nets: shared trunk -> policy head (softmax/logits over CHECKER_ACTIONS=676) + single tanh
value head. Policy loss = soft cross-entropy vs the teacher's soft 676-dim checker target.
Value loss = MSE vs equity/3. Identical for both arms.

Then measure RAW-policy contact PR of BOTH on the SAME fixed benchmark position set
(seed 1, gnubg-0ply generator; gnubg-ply1 native move-list grading — reused verbatim
from scripts/benchmark_pr.jl). Raw policy = policy-head argmax over legal actions, NO
search. The FLAT control MUST reproduce ~79 contact PR to validate the harness.

This is self-contained: it builds the Flux nets directly (mirroring
src/networks/architectures/{fc_resnet.jl,resnet.jl}) and drives game mechanics through
BackgammonNet directly, so it needs NO game.jl include and is unaffected by the
precompile-baked BACKGAMMON_OBS_TYPE. No existing src/ or game files are modified.

Usage:
    julia --threads auto --project scripts/spatial_obs_ab.jl \
        --epochs 300 --batch 512 --n-contact 1000 --seed 1
"""

using ArgParse

function parse_args_ab()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--data";        arg_type=String; default="/home/sile/github/BackgammonNet.jl/data/bootstrap/contact_bootstrap_obs_test_30k.jls"
        "--epochs";      arg_type=Int;    default=300
        "--batch";       arg_type=Int;    default=512
        "--lr";          arg_type=Float64; default=1e-3
        "--lr-final";    arg_type=Float64; default=1e-4
        "--val-frac";    arg_type=Float64; default=0.1
        # flat arch
        "--flat-width";  arg_type=Int;    default=256
        "--flat-blocks"; arg_type=Int;    default=5
        # conv arch
        "--conv-filters"; arg_type=Int;   default=64
        "--conv-blocks";  arg_type=Int;   default=5
        "--conv-kernel";  arg_type=Int;   default=3    # kernel along the 76 board dim (width dim = 1)
        # value loss weight
        "--vloss";       arg_type=Float64; default=1.0
        # benchmark
        "--n-contact";   arg_type=Int;    default=1000
        "--seed";        arg_type=Int;    default=1
        "--num-workers"; arg_type=Int;    default=12
        "--max-games";   arg_type=Int;    default=100_000
        "--arms";        arg_type=String; default="flat,conv"
        "--out-dir";     arg_type=String; default="sessions/spatial_obs_ab"
        "--pos-cache";   arg_type=String; default=""   # serialize/reuse benchmark positions
        "--eval-every";  arg_type=Int;    default=10    # val-CE eval / early-stop check cadence
        "--wildbg-lib";  arg_type=String; default=""    # libwildbg.so (default ~/github/wildbg/...)
        "--skip-refs";   action=:store_true             # skip the reference-agent ladder
        "--smoke";       action=:store_true            # tiny run to validate the pipeline
    end
    return ArgParse.parse_args(s)
end

const A = parse_args_ab()

using BackgammonNet
using Serialization
using Random
using Statistics
using Printf
using Dates
import Flux
using Flux: Chain, Dense, Conv, BatchNorm, LayerNorm, SkipConnection, relu, flatten, softmax
const Optimisers = Flux.Optimisers
import CUDA

const DEV = CUDA.functional() ? Flux.gpu : identity
const NACT = Int(BackgammonNet.MAX_ACTIONS)

# ─────────────────────────────────────────────────────────────────────────────
# Networks (pure Flux, mirroring src/networks/architectures/{fc_resnet,resnet}.jl)
# Policy head outputs LOGITS (softmax applied at eval); value head outputs tanh.
# ─────────────────────────────────────────────────────────────────────────────

# --- FLAT: FC pre-activation ResNet (mirrors fc_resnet.jl) ---
struct PreActBlock; ln1; d1; ln2; d2; end
Flux.@layer PreActBlock
PreActBlock(w::Int) = PreActBlock(LayerNorm(w), Dense(w,w), LayerNorm(w), Dense(w,w))
function (b::PreActBlock)(x)
    h = b.d1(relu.(b.ln1(x)))
    h = b.d2(relu.(b.ln2(h)))
    return x .+ h
end

struct FlatNet; trunk; phead; vhead; end
Flux.@layer FlatNet
function FlatNet(indim::Int, width::Int, nblocks::Int)
    input = Chain(Dense(indim, width), LayerNorm(width), x->relu.(x))
    blocks = [PreActBlock(width) for _ in 1:nblocks]
    trunk = Chain(input, blocks..., LayerNorm(width), x->relu.(x))
    phead = Chain(Dense(width,width), LayerNorm(width), x->relu.(x),
                  Dense(width,width), LayerNorm(width), x->relu.(x),
                  Dense(width, NACT))                       # logits
    vhead = Chain(Dense(width,width), LayerNorm(width), x->relu.(x),
                  Dense(width,width), LayerNorm(width), x->relu.(x),
                  Dense(width, 1))                          # pre-tanh
    return FlatNet(trunk, phead, vhead)
end
function (m::FlatNet)(x)
    c = m.trunk(x)
    return (m.phead(c), tanh.(m.vhead(c)))
end

# --- CONV: AlphaGo-Zero conv ResNet (mirrors resnet.jl) ---
function ConvResBlock(ksize, n, pad, bnmom)
    layers = Chain(
        Conv(ksize, n=>n, pad=pad),
        BatchNorm(n, relu, momentum=bnmom),
        Conv(ksize, n=>n, pad=pad),
        BatchNorm(n, momentum=bnmom))
    return Chain(SkipConnection(layers, +), x->relu.(x))
end

struct ConvNet; common; phead; vhead; end
Flux.@layer ConvNet
function ConvNet(H::Int, W::Int, C::Int, nf::Int, nblocks::Int, kh::Int)
    ksize = (kh, 1)
    pad = (kh ÷ 2, 0)
    bnmom = 0.6f0
    common = Chain(
        Conv(ksize, C=>nf, pad=pad),
        BatchNorm(nf, relu, momentum=bnmom),
        [ConvResBlock(ksize, nf, pad, bnmom) for _ in 1:nblocks]...)
    npf = 2; nvf = 1
    phead = Chain(
        Conv((1,1), nf=>npf), BatchNorm(npf, relu, momentum=bnmom),
        flatten, Dense(H*W*npf, NACT))                      # logits
    vhead = Chain(
        Conv((1,1), nf=>nvf), BatchNorm(nvf, relu, momentum=bnmom),
        flatten, Dense(H*W*nvf, nf, relu), Dense(nf, 1))    # pre-tanh
    return ConvNet(common, phead, vhead)
end
function (m::ConvNet)(x)
    c = m.common(x)
    return (m.phead(c), tanh.(m.vhead(c)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Observations (re-observe any BackgammonGame with the requested obs type)
# ─────────────────────────────────────────────────────────────────────────────
function obs_flat(g::BackgammonNet.BackgammonGame)
    gg = BackgammonNet.clone(g); gg.obs_type = :min_plus_flat
    return Float32.(vec(BackgammonNet.observe(gg)))          # (350,)
end
function obs_conv(g::BackgammonNet.BackgammonGame)
    gg = BackgammonNet.clone(g); gg.obs_type = :full
    o = Float32.(BackgammonNet.observe(gg))                  # (76,1,26) = (feature-channel, 1, boardloc)
    # PERMUTE to (26,1,76) so the SPATIAL board axis (26 locations: MyBar,P1..P24,OppBar)
    # is the Flux spatial dim and the 76 features are channels. The conv kernel then
    # slides over ADJACENT BOARD POINTS — the genuine spatial-locality test.
    return permutedims(o, (3, 2, 1))                         # (26,1,76)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark machinery — reused VERBATIM from scripts/benchmark_pr.jl
# (depends only on BackgammonNet; obs-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
struct MoveDecision
    res_p0::UInt128
    res_p1::UInt128
    is_contact::Bool
    player::Int
    forced_move::Bool     # whole turn had <=1 option at its native root (marked by grading)
end
struct BenchPosition
    state::BackgammonNet.BackgammonGame
    is_contact::Bool
    player::Int
end

function native_regret(gnubg, pre::BackgammonNet.BackgammonGame, res_p0, res_p1)
    g = pre
    (g.phase == BackgammonNet.PHASE_CHECKER_PLAY) || return (:forced, 0.0)
    BackgammonNet.open!(gnubg)
    md = lock(BackgammonNet._GNUBG_CLIB_LOCK) do
        BackgammonNet._gnubg_clib_move_data(gnubg, g)
    end
    n = length(md)
    n <= 1 && return (:forced, 0.0)
    player = Int(g.current_player)
    best_eq = -Inf; our_eq = nothing
    @inbounds for (tsimple, probs) in md
        eq = Float64(BackgammonNet.compute_cubeless_equity(g, probs))
        eq > best_eq && (best_eq = eq)
        tp0, tp1 = BackgammonNet.from_gnubg_simple(tsimple, player)
        if tp0 == res_p0 && tp1 == res_p1
            our_eq = eq
        end
    end
    our_eq === nothing && return (:unmatched, 0.0)
    return (:ok, max(best_eq - our_eq, 0.0))
end

function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end

function play_turn_with_engine!(g, engine, fail_counter::Threads.Atomic{Int})
    start_player = g.current_player
    while true
        att = BackgammonNet.action_type(g)
        (att == BackgammonNet.ACTION_TYPE_TERMINAL || att == BackgammonNet.ACTION_TYPE_CHANCE) && break
        g.current_player != start_player && break
        acts = BackgammonNet.legal_actions(g)
        isempty(acts) && break
        a = length(acts) == 1 ? acts[1] :
            try BackgammonNet.best_move(engine, g) catch
                Threads.atomic_add!(fail_counter, 1); acts[1]
            end
        BackgammonNet.apply_action!(g, a)
        g.terminated && break
    end
    return g
end

function generate_common_positions(gen_engine, n_contact::Int, n_race::Int, base_seed::Int, max_games::Int)
    contact = BenchPosition[]; race = BenchPosition[]
    fail = Threads.Atomic{Int}(0); gi = 0
    while (length(contact) < n_contact || length(race) < n_race) && gi < max_games
        gi += 1
        rng = MersenneTwister(base_seed + gi)
        g = BackgammonNet.initial_state()
        while true
            at = BackgammonNet.action_type(g)
            at == BackgammonNet.ACTION_TYPE_TERMINAL && break
            if at == BackgammonNet.ACTION_TYPE_CHANCE
                BackgammonNet.apply_chance!(g, _sample_dice(rng)); continue
            end
            pre = BackgammonNet.clone(g)
            is_contact = BackgammonNet.is_contact_position(g)
            player = Int(g.current_player)
            if is_contact
                length(contact) < n_contact && push!(contact, BenchPosition(pre, true, player))
            else
                length(race) < n_race && push!(race, BenchPosition(pre, false, player))
            end
            play_turn_with_engine!(g, gen_engine, fail)
            g.terminated && break
            (length(contact) >= n_contact && length(race) >= n_race) && break
        end
    end
    return contact, race, gi, fail[]
end

# ─────────────────────────────────────────────────────────────────────────────
# NET raw-policy full-turn move (argmax over legal actions, NO search).
# Sequential per position, using a CPU net. Returns MoveDecision per position.
# ─────────────────────────────────────────────────────────────────────────────
"""obsfn: game -> Array; netfwd: obs-batch -> policy-logits matrix (NACT x B)."""
function net_decisions(positions::Vector{BenchPosition}, obsfn, policy_logits)
    out = Vector{MoveDecision}(undef, length(positions))
    for (i, bp) in enumerate(positions)
        g = BackgammonNet.clone(bp.state)
        start_player = g.current_player
        while true
            att = BackgammonNet.action_type(g)
            (att == BackgammonNet.ACTION_TYPE_TERMINAL || att == BackgammonNet.ACTION_TYPE_CHANCE) && break
            g.current_player != start_player && break
            acts = BackgammonNet.legal_actions(g)
            isempty(acts) && break
            a = if length(acts) == 1
                acts[1]
            else
                logits = policy_logits(obsfn(g))          # NACT-vector
                # argmax over legal actions only
                best = acts[1]; bestv = -Inf32
                @inbounds for act in acts
                    v = logits[act]
                    v > bestv && (bestv = v; best = act)
                end
                best
            end
            BackgammonNet.apply_action!(g, a)
            g.terminated && break
        end
        out[i] = MoveDecision(g.p0, g.p1, bp.is_contact, bp.player, false)
    end
    return out
end

# ENGINE agent (gnubg0/gnubg1/wildbg): choose full-turn move per position (serial).
function engine_decisions(engine, positions::Vector{BenchPosition})
    out = Vector{MoveDecision}(undef, length(positions))
    fail = Threads.Atomic{Int}(0)
    for (i, bp) in enumerate(positions)
        g = BackgammonNet.clone(bp.state)
        play_turn_with_engine!(g, engine, fail)
        out[i] = MoveDecision(g.p0, g.p1, bp.is_contact, bp.player, false)
    end
    return out, fail[]
end

function score_moves(gnubg_backends, positions, decisions, num_workers)
    n = length(decisions)
    errors = fill(NaN, n); contact_flag = Vector{Bool}(undef, n)
    status = Vector{Symbol}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        gb = gnubg_backends[w]
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            d = decisions[i]; contact_flag[i] = d.is_contact
            local res
            try
                res = native_regret(gb, positions[i].state, d.res_p0, d.res_p1)
            catch e
                status[i] = :error; continue
            end
            tag, err = res; status[i] = tag
            tag == :ok && (errors[i] = err)
        end
    end
    return errors, contact_flag, status
end

function summarize(errors, contact_flag, status)
    ok = status .== :ok
    n_ok = count(ok); n_ok == 0 && return nothing
    e_ok = errors[ok]
    subset(sel) = begin
        e = errors[ok .& sel]
        isempty(e) ? (n=0, PR=NaN) : (n=length(e), PR=500.0*sum(e)/length(e))
    end
    return (n_ok=n_ok, n_forced=count(==(:forced),status), n_unmatched=count(==(:unmatched),status),
            n_error=count(==(:error),status), PR=500.0*mean(e_ok),
            contact=subset(contact_flag), race=subset(.!contact_flag))
end

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
function build_targets(states, policies, values)
    n = length(states)
    P = Array{Float32}(undef, NACT, n)
    V = Array{Float32}(undef, 1, n)
    @inbounds for i in 1:n
        P[:, i] .= policies[i]
        V[1, i] = clamp(Float32(values[i]) / 3f0, -1f0, 1f0)
    end
    return P, V
end

function build_obs_flat(states)
    n = length(states); X = Array{Float32}(undef, 350, n)
    @inbounds for i in 1:n; X[:, i] .= obs_flat(states[i]); end
    return X
end
function build_obs_conv(states)
    n = length(states); X = Array{Float32}(undef, 26, 1, 76, n)  # (boardloc,1,feature,N)
    @inbounds for i in 1:n; X[:, :, :, i] .= obs_conv(states[i]); end
    return X
end

lastdim_index(X::AbstractMatrix, idx) = X[:, idx]
lastdim_index(X::AbstractArray{<:Any,4}, idx) = X[:, :, :, idx]

# top-1 agreement of predicted logits vs teacher argmax, restricted to legal (nonzero) support
function top1_agree(logits::AbstractMatrix, Ptar::AbstractMatrix)
    n = size(Ptar, 2); agree = 0
    for i in 1:n
        legal = findall(>(0f0), view(Ptar,:,i))
        isempty(legal) && continue
        tgt = legal[argmax(view(Ptar, legal, i))]
        prd = legal[argmax(view(logits, legal, i))]
        agree += (tgt == prd)
    end
    return agree / n
end

# legal mask -> additive offset: 0 on legal (target>0), -1e9 on illegal, so
# softmax is confined to legal actions (matches production forward_normalized).
neg_from_target(P) = Float32[p > 0f0 ? 0f0 : -1f9 for p in P]

function train_arm!(name, model, Xtr, Ptr, Vtr, Xval, Pval, Vval; epochs, batch, lr0, lrf, vloss, eval_every)
    model = model |> DEV
    Xtr_d = Xtr |> DEV; Ptr_d = Ptr |> DEV; Vtr_d = Vtr |> DEV
    NEGtr_d = neg_from_target(Ptr) |> DEV
    NEGval = neg_from_target(Pval)
    ntr = size(Ptr, 2)
    opt = Optimisers.setup(Optimisers.Adam(lr0), model)
    nbatch = cld(ntr, batch)
    # masked policy CE (softmax over legal actions only) + value MSE
    lossfn(m, xb, pb, vb, neg) = begin
        pl, v = m(xb)
        pce = Flux.logitcrossentropy(pl .+ neg, pb; dims=1)
        vm  = Flux.mse(v, vb)
        return pce + Float32(vloss) * vm
    end
    best_ce = Inf; best_model = deepcopy(model |> Flux.cpu); best_ep = 0
    t0 = time()
    for ep in 1:epochs
        frac = epochs <= 1 ? 1.0 : (ep-1)/(epochs-1)
        lr = lrf + 0.5*(lr0-lrf)*(1+cos(pi*frac))       # cosine DECAY: lr0 -> lrf
        Optimisers.adjust!(opt, lr)
        perm = randperm(ntr)
        Flux.trainmode!(model)
        for b in 1:nbatch
            lo = (b-1)*batch + 1; hi = min(b*batch, ntr)
            bi = perm[lo:hi]
            xb = lastdim_index(Xtr_d, bi); pb = Ptr_d[:, bi]; vb = Vtr_d[:, bi]; neg = NEGtr_d[:, bi]
            val, gs = Flux.withgradient(m -> lossfn(m, xb, pb, vb, neg), model)
            Optimisers.update!(opt, model, gs[1])
        end
        if ep == 1 || ep % eval_every == 0 || ep == epochs
            Flux.testmode!(model)
            cpum = model |> Flux.cpu
            plv, vv = cpum(Xval)
            vce = Flux.logitcrossentropy(plv .+ NEGval, Pval; dims=1)
            vmse = Flux.mse(vv, Vval)
            ag = top1_agree(plv .+ NEGval, Pval)
            marker = ""
            if vce < best_ce
                best_ce = vce; best_ep = ep; best_model = deepcopy(cpum); marker = " *best*"
            end
            @printf("  [%s] ep %3d/%d  lr=%.2e  val CE=%.4f  val vMSE=%.4f  top1-agree=%.3f  (%.0fs)%s\n",
                    name, ep, epochs, lr, vce, vmse, ag, time()-t0, marker)
            flush(stdout)
        end
    end
    @printf("  [%s] EARLY-STOP: best val CE=%.4f at ep %d (reporting THAT checkpoint)\n", name, best_ce, best_ep)
    Flux.testmode!(best_model)
    return best_model
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
function main()
    if A["smoke"]
        A["epochs"] = 3; A["n_contact"] = 40
    end
    arms = split(A["arms"], ",")
    mkpath(A["out_dir"])
    println("="^78)
    println("SPATIAL-OBS A/B  —  does conv/spatial beat flat MLP on raw POLICY?")
    println("="^78)
    println("Data:        ", A["data"])
    println("Arms:        ", join(arms, ", "))
    println("Epochs=$(A["epochs"])  batch=$(A["batch"])  lr=$(A["lr"])->$(A["lr_final"])  vloss=$(A["vloss"])")
    println("Flat arch:   FC-ResNet  width=$(A["flat_width"]) blocks=$(A["flat_blocks"])  obs=min_plus_flat(350)")
    println("Conv arch:   ResNet     filters=$(A["conv_filters"]) blocks=$(A["conv_blocks"]) kernel=($(A["conv_kernel"]),1)  obs=full permuted to (26 boardloc,1,76 feat) — kernel slides over BOARD POINTS")
    println("Benchmark:   $(A["n_contact"]) contact positions, seed $(A["seed"]), gnubg-0ply gen + gnubg-1ply native grading")
    println("Device:      ", CUDA.functional() ? "GPU ("*CUDA.name(CUDA.device())*")" : "CPU")
    println("Threads:     ", Threads.nthreads())
    println("="^78); flush(stdout)

    # ── Load dataset ──
    println("\n[1] Loading dataset..."); flush(stdout)
    d = Serialization.deserialize(A["data"])
    states = d.states; policies = d.policies; values = d.values
    n = length(states)
    println("    $n samples (all contact=$(all(BackgammonNet.is_contact_position, states[1:min(n,3000)])))")

    # fixed train/val split
    Random.seed!(12345)
    perm = randperm(n)
    nval = max(1, round(Int, A["val_frac"]*n))
    val_idx = perm[1:nval]; tr_idx = perm[nval+1:end]
    println("    train=$(length(tr_idx))  val=$(length(val_idx))")

    P, V = build_targets(states, policies, values)
    Ptr = P[:, tr_idx]; Vtr = V[:, tr_idx]; Pval = P[:, val_idx]; Vval = V[:, val_idx]

    # ── Build / load benchmark positions (deterministic in seed) ──
    println("\n[2] Building common benchmark position set..."); flush(stdout)
    cache = A["pos_cache"]
    positions = nothing
    if !isempty(cache) && isfile(cache)
        positions = Serialization.deserialize(cache)
        println("    loaded $(length(positions)) cached positions from $cache")
    else
        tgen = time()
        gen = BackgammonNet.GnubgCLibBackend(ply=0, threads=1); BackgammonNet.open!(gen)
        contact_ps, race_ps, ngames, gfail = generate_common_positions(gen, A["n_contact"], 0, A["seed"], A["max_games"])
        try BackgammonNet.close(gen) catch end
        positions = contact_ps
        println("    $(length(positions)) contact positions from $ngames games ($(round(time()-tgen,digits=1))s, gen fallbacks=$gfail)")
        if !isempty(cache); Serialization.serialize(cache, positions); println("    cached -> $cache"); end
    end
    flush(stdout)

    # shared gnubg-ply1 grading backends
    nw = A["num_workers"]
    println("\n[3] Opening $nw gnubg ply-1 grading backends..."); flush(stdout)
    gbs = [begin gb=BackgammonNet.GnubgCLibBackend(ply=1, threads=1); BackgammonNet.open!(gb); gb end for _ in 1:nw]

    results = Dict{String,Any}()

    # ── Reference-agent LADDER: anchors the PR scale on THESE positions ──
    #   gnubg1 (grades its own ply-1 moves) MUST be ~1 PR  → pipeline validated
    #   gnubg0 ~2 PR ; wildbg-large ~13.5 PR (the teacher-class reference)
    # If these land on their known values, the flat/conv PR numbers are trustworthy
    # on an ABSOLUTE scale regardless of whether flat reproduces exactly 79.
    ref_ladder = Tuple{String,Any}[]
    if !A["skip_refs"]
        wildbg_lib = A["wildbg_lib"]
        isempty(wildbg_lib) && (wildbg_lib = joinpath(homedir(),"github","wildbg","target","release","libwildbg.so"))
        ref_specs = [(:gnubg1, "gnubg-1ply  [FLOOR/self-consistency]"),
                     (:gnubg0, "gnubg-0ply"),
                     (:wildbg, "wildbg-large [teacher-class ref]")]
        for (sym, lbl) in ref_specs
            sym === :wildbg && !isfile(wildbg_lib) && (println("  (wildbg lib not found, skipping wildbg ref)"); continue)
            println("\n[REF] $lbl : choosing moves + grading..."); flush(stdout)
            local eng
            try
                if sym === :gnubg0
                    eng = BackgammonNet.GnubgCLibBackend(ply=0, threads=1); BackgammonNet.open!(eng)
                elseif sym === :gnubg1
                    eng = BackgammonNet.GnubgCLibBackend(ply=1, threads=1); BackgammonNet.open!(eng)
                else
                    lsz = filesize(wildbg_lib)
                    if lsz > 10_000_000; BackgammonNet.wildbg_set_lib_path!(large=wildbg_lib)
                    else; BackgammonNet.wildbg_set_lib_path!(small=wildbg_lib); end
                    eng = BackgammonNet.WildbgBackend(nets = lsz>10_000_000 ? :large : :small); BackgammonNet.open!(eng)
                end
            catch e
                println("  (failed to open $sym: $e — skipping)"); continue
            end
            dec, ef = engine_decisions(eng, positions)
            try BackgammonNet.close(eng) catch end
            errors, cflag, status = score_moves(gbs, positions, dec, nw)
            s = summarize(errors, cflag, status)
            push!(ref_ladder, (String(sym), s))
            @printf("  %-14s contact PR = %6.2f  (scored=%d forced=%d unmatched=%d fallbacks=%d)\n",
                    String(sym), s.contact.PR, s.n_ok, s.n_forced, s.n_unmatched, ef)
            flush(stdout)
        end
    end

    for arm in arms
        println("\n" * "="^78)
        println("ARM: $arm")
        println("="^78); flush(stdout)

        if arm == "flat"
            println("[$arm] building observations (min_plus_flat)..."); flush(stdout)
            X = build_obs_flat(states)
            Xtr = X[:, tr_idx]; Xval = X[:, val_idx]
            model = FlatNet(350, A["flat_width"], A["flat_blocks"])
            obsfn = obs_flat
        elseif arm == "conv"
            println("[$arm] building observations (full, permuted to (26 boardloc,1,76 feat))..."); flush(stdout)
            X = build_obs_conv(states)
            Xtr = X[:,:,:,tr_idx]; Xval = X[:,:,:,val_idx]
            # spatial board axis = 26 locations, feature channels = 76
            model = ConvNet(26, 1, 76, A["conv_filters"], A["conv_blocks"], A["conv_kernel"])
            obsfn = obs_conv
        else
            error("unknown arm $arm")
        end

        np = sum(length, Flux.trainables(model))
        println("[$arm] params ≈ $(np)"); flush(stdout)

        println("[$arm] training..."); flush(stdout)
        model = train_arm!(arm, model, Xtr, Ptr, Vtr, Xval, Pval, Vval;
                           epochs=A["epochs"], batch=A["batch"], lr0=A["lr"], lrf=A["lr_final"],
                           vloss=A["vloss"], eval_every=A["eval_every"])

        # save weights (Flux state) for record
        wpath = joinpath(A["out_dir"], "$(arm)_net.jls")
        Serialization.serialize(wpath, Flux.cpu(Flux.state(model)))
        println("[$arm] saved -> $wpath"); flush(stdout)

        # ── raw-policy decisions on the benchmark set ──
        println("[$arm] computing raw-policy full-turn decisions on $(length(positions)) positions..."); flush(stdout)
        Flux.testmode!(model)
        # closure: obs -> policy logits vector (CPU net, single forward)
        function plogits(o)
            xb = arm == "flat" ? reshape(o, :, 1) : reshape(o, size(o)..., 1)
            pl, _ = model(xb)
            return vec(pl)
        end
        t0 = time()
        decisions = net_decisions(positions, obsfn, plogits)
        println("    move selection done ($(round(time()-t0,digits=1))s)"); flush(stdout)

        println("[$arm] gnubg ply-1 native grading..."); flush(stdout)
        t1 = time()
        errors, cflag, status = score_moves(gbs, positions, decisions, nw)
        s = summarize(errors, cflag, status)
        println("    grading done ($(round(time()-t1,digits=1))s)")
        results[arm] = s
        @printf("[%s] RAW-POLICY contact PR = %.2f   (scored=%d forced=%d unmatched=%d err=%d)\n",
                arm, s.contact.PR, s.n_ok, s.n_forced, s.n_unmatched, s.n_error)
        flush(stdout)
    end

    for gb in gbs; try BackgammonNet.close(gb) catch end; end

    # ── Verdict ──
    println("\n" * "="^78)
    println("VERDICT — raw-policy CONTACT PR on identical benchmark positions")
    println("="^78)
    if !isempty(ref_ladder)
        println("Reference ladder (anchors the PR scale on THESE positions):")
        for (nm, s) in ref_ladder
            @printf("  %-16s contact PR = %6.2f  (n=%d)\n", nm, s.contact.PR, s.contact.n)
        end
        g1 = findfirst(x->x[1]=="gnubg1", ref_ladder)
        if g1 !== nothing
            fpr = ref_ladder[g1][2].contact.PR
            println(fpr < 3.0 ? "  ✓ gnubg1 FLOOR ≈ $(round(fpr,digits=1)) PR → grading pipeline VALIDATED (scale is correct)." :
                                "  ✗ gnubg1 FLOOR = $(round(fpr,digits=1)) (expected ~1) → grading scale suspect.")
        end
        println()
    end
    flatpr = haskey(results,"flat") ? results["flat"].contact.PR : nothing
    convpr = haskey(results,"conv") ? results["conv"].contact.PR : nothing
    flatpr !== nothing && @printf("  FLAT  (min_plus_flat + FC-ResNet):  PR = %.2f\n", flatpr)
    convpr !== nothing && @printf("  CONV  (full spatial + conv ResNet): PR = %.2f\n", convpr)
    println("  (teacher wildbg ≈ 15 ; production flat baseline ≈ 79 = harness sanity target)")
    if flatpr !== nothing
        if 65 <= flatpr <= 95
            println("  ✓ FLAT control reproduces ~79 → supervised+eval harness VALIDATED.")
        else
            println("  ✗ FLAT control = $(round(flatpr,digits=1)), NOT ~79 → INSPECT HARNESS before trusting conv.")
        end
    end
    if flatpr !== nothing && convpr !== nothing
        gap = flatpr - convpr
        @printf("  Δ(flat-conv) = %.2f PR\n", gap)
        if convpr < flatpr - 15
            println("  >>> CONV's raw policy is MUCH BETTER than flat → the flat obs/arch was the ceiling.")
            println("      The cubeful net SHOULD use a spatial/conv architecture + spatial obs.")
        elseif abs(gap) <= 15
            println("  >>> CONV ≈ FLAT → spatial obs/arch is NOT the fix. The policy ceiling is DEEPER")
            println("      (policy-vs-value optimization asymmetry or a more fundamental issue).")
        else
            println("  >>> CONV is WORSE than flat → spatial arch did not help here.")
        end
    end
    println("="^78); flush(stdout)
end

main()
