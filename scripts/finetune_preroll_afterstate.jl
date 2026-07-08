#!/usr/bin/env julia
"""
finetune_preroll_afterstate.jl — Exp 1 training.

Fine-tune the EXISTING i140 contact value net into a DIRECT PRE-ROLL AFTERSTATE
evaluator, trained on wildbg per-move sibling labels
(generate_sibling_afterstates.jl output).

Target contract (per afterstate A, opponent to move, encoded PRE-ROLL i.e. dice
planes all zero): net(A) 5 heads ≈ wildbg evaluate_probs(A), both in
current_player(A) = OPPONENT perspective. At selection time the mover scores
Score(move) = -compute_cubeless_equity(A, net(A)) = MY equity.

Loss: BCEWithLogits on the 5 joint-cumulative heads vs wildbg soft probs
(identical to the training pipeline's bce_logits_wmean). Policy head is NOT in the
value loss graph, so it is left untouched.

Retention: 25-30% of every batch is the ORIGINAL POST-ROLL bootstrap data (decision
states + their wildbg 5-head), preventing post-roll feature forgetting.

Split BY DECISION (all siblings of a decision on the same side).
Early-stop on the DECISION-LEVEL ranking proxy (held-out): mean equity-loss of the
net-argmax move vs the wildbg-label-argmax move, and argmax-match rate.

Validation (printed): opening pre-roll equity (net vs wildbg teacher), mirror
symmetry V(s) = -V(mirror(s)) (wildbg-verified construction), old-vs-new proxy.

Usage:
  julia --threads 8 --project scripts/finetune_preroll_afterstate.jl \\
      --data=sessions/preroll-afterstate/sibling_afterstates_30k.jls \\
      --net-ckpt=sessions/contact-flywheel/checkpoints/contact_iter_140.data \\
      --out=sessions/preroll-afterstate/preroll_afterstate_ft.data
"""

using ArgParse
function parse_args_ft()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--data";        arg_type=String; default="sessions/preroll-afterstate/sibling_afterstates_30k.jls"
        "--net-ckpt";    arg_type=String; default="sessions/contact-flywheel/checkpoints/contact_iter_140.data"
        "--out";         arg_type=String; default="sessions/preroll-afterstate/preroll_afterstate_ft.data"
        "--obs-type";    arg_type=String; default="min_plus_flat"
        "--width";       arg_type=Int; default=256
        "--blocks";      arg_type=Int; default=5
        "--epochs";      arg_type=Int; default=30
        "--batch-size";  arg_type=Int; default=1024
        "--lr";          arg_type=Float64; default=1.0e-4
        "--retention-frac"; arg_type=Float64; default=0.28
        "--val-frac";    arg_type=Float64; default=0.10
        "--patience";    arg_type=Int; default=4
        "--seed";        arg_type=Int; default=1
        "--ranking-weight"; arg_type=Float64; default=0.0   # 0 = regression only; A/B uses 0.15
        "--ranking-gap";    arg_type=Float64; default=0.02  # only pairs with wildbg equity gap > this
        "--ranking-margin-cap"; arg_type=Float64; default=0.2
        "--gpu";         action=:store_true
        "--no-gpu";      action=:store_true
    end
    return ArgParse.parse_args(s)
end
const A = parse_args_ft()
ENV["BACKGAMMON_OBS_TYPE"] = A["obs_type"]

using AlphaZero
using AlphaZero: GI, FluxLib
import Flux
import CUDA
using BackgammonNet
using Serialization, Random, Statistics, Printf
import LinearAlgebra
try; LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()); catch; end

include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const STATE_DIM = GI.state_dim(gspec)[1]
const OBS_SYM = Symbol(A["obs_type"])

@inline eqjoint5(pw,gw,bgw,gl,bgl) = (2f0*pw-1f0)+(gw-gl)+(bgw-bgl)

# ── net IO ────────────────────────────────────────────────────────────────
function load_net(path, width, blocks)
    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(path, net)
    return Flux.cpu(net)
end

# raw value logits (skips the policy head entirely → phead never touched)
function value_logits(net, X)
    c = net.common(X)
    v = net.vhead_trunk(c)
    return net.vhead_win(v), net.vhead_gw(v), net.vhead_bgw(v), net.vhead_gl(v), net.vhead_bgl(v)
end

bcelogits(l, y) = mean(max.(l, 0f0) .- l .* y .+ log.(1f0 .+ exp.(.-abs.(l))))
function loss_fn(net, X, Y)
    lw, lgw, lbgw, lgl, lbgl = value_logits(net, X)
    return bcelogits(lw, view(Y,1:1,:)) + bcelogits(lgw, view(Y,2:2,:)) +
           bcelogits(lbgw, view(Y,3:3,:)) + bcelogits(lgl, view(Y,4:4,:)) +
           bcelogits(lbgl, view(Y,5:5,:))
end

# combined loss: regression BCE (all samples) + optional pairwise ranking hinge on
# head-DERIVED my-equity over sibling pairs (pa=better move, pb=worse move by wildbg
# label, margin=capped wildbg equity gap). pa/pb index the FIRST nsib columns of Xb.
function loss_fn_rank(net, Xb, Yb, nsib::Int, pa, pb, marg, rw::Float32)
    lw, lgw, lbgw, lgl, lbgl = value_logits(net, Xb)
    reg = bcelogits(lw, view(Yb,1:1,:)) + bcelogits(lgw, view(Yb,2:2,:)) +
          bcelogits(lbgw, view(Yb,3:3,:)) + bcelogits(lgl, view(Yb,4:4,:)) +
          bcelogits(lbgl, view(Yb,5:5,:))
    (rw == 0f0 || isempty(marg)) && return reg
    pw = Flux.sigmoid.(vec(lw)); pgw = Flux.sigmoid.(vec(lgw)); pbgw = Flux.sigmoid.(vec(lbgw))
    pgl = Flux.sigmoid.(vec(lgl)); pbgl = Flux.sigmoid.(vec(lbgl))
    opp = (2f0 .* pw .- 1f0) .+ (pgw .- pgl) .+ (pbgw .- pbgl)   # opponent equity per sample
    myeq = (.-opp)[1:nsib]                                       # my equity (sibling block)
    rank = Statistics.mean(max.(0f0, marg .- (myeq[pa] .- myeq[pb])))
    return reg + rw * rank
end

# probabilities (5 x N) for a batch of states, batched on device
function net_probs(net, X; chunk=8192)
    N = size(X, 2)
    P = Matrix{Float32}(undef, 5, N)
    p = 1
    while p <= N
        q = min(p+chunk-1, N)
        lw,lgw,lbgw,lgl,lbgl = value_logits(net, X[:, p:q])
        P[1,p:q] = Flux.cpu(vec(Flux.sigmoid.(lw)))
        P[2,p:q] = Flux.cpu(vec(Flux.sigmoid.(lgw)))
        P[3,p:q] = Flux.cpu(vec(Flux.sigmoid.(lbgw)))
        P[4,p:q] = Flux.cpu(vec(Flux.sigmoid.(lgl)))
        P[5,p:q] = Flux.cpu(vec(Flux.sigmoid.(lbgl)))
        p = q + 1
    end
    return P
end

# ── decision-level ranking proxy on held-out val decisions ──────────────────
# my_equity(A) = -opp_equity(A); best move = min opp_equity. Compare net pick to
# wildbg-label pick; equity-loss = opp_eq_wb[net_pick] - opp_eq_wb[true_pick] (>=0).
function ranking_proxy(net, Xdev, labels5::Matrix{Float32}, groups::Vector{Vector{Int}})
    P = net_probs(net, Xdev)                       # 5 x N (net probs, opp perspective)
    opp_eq_net = [eqjoint5(P[1,i],P[2,i],P[3,i],P[4,i],P[5,i]) for i in 1:size(P,2)]
    opp_eq_wb  = [eqjoint5(labels5[1,i],labels5[2,i],labels5[3,i],labels5[4,i],labels5[5,i]) for i in 1:size(labels5,2)]
    nmatch = 0; sumloss = 0.0; ndec = 0
    for g in groups
        length(g) < 2 && continue
        ndec += 1
        net_pick = g[argmin(@view opp_eq_net[g])]
        true_pick = g[argmin(@view opp_eq_wb[g])]
        nmatch += (net_pick == true_pick) ? 1 : 0
        sumloss += opp_eq_wb[net_pick] - opp_eq_wb[true_pick]
    end
    return (argmax_match = nmatch/ndec, equity_loss = sumloss/ndec, ndec = ndec)
end

# ── perspective-relabel mirror: swap pieces + flip on-roll player so the SAME
# on-roll board is presented → wildbg equity is UNCHANGED (verified Δ=0). This is
# the satisfiable symmetry for an on-roll/pre-roll evaluator; the literal
# V(s)=-V(mirror(s)) is ill-posed here (on-roll tempo is inherently asymmetric).
function relabel_mirror(g::BackgammonNet.BackgammonGame)
    cp = Int(g.current_player)
    s = Vector{Int}(undef, 26)
    BackgammonNet.to_gnubg_simple!(s, g; perspective=cp)      # cp's on-roll view
    p0n, p1n = BackgammonNet.from_gnubg_simple(s, 1-cp)        # hand that exact board to 1-cp
    m = BackgammonNet.clone(g)
    m.p0 = p0n; m.p1 = p1n; m.current_player = Int8(1-cp)
    return m
end

function net_equity_one(net, g::BackgammonNet.BackgammonGame)
    x = Matrix{Float32}(undef, STATE_DIM, 1)
    vectorize_state_into!(view(x,:,1), gspec, g)
    P = net_probs(net, _to_dev(x))
    return eqjoint5(P[1,1],P[2,1],P[3,1],P[4,1],P[5,1])
end

# ── device helpers ──────────────────────────────────────────────────────────
const USE_GPU = Ref(false)
_to_dev(x) = USE_GPU[] ? Flux.gpu(x) : x

function main()
    Random.seed!(A["seed"])
    use_gpu = A["gpu"] || (!A["no_gpu"])
    if use_gpu
        try
            use_gpu = CUDA.functional()
        catch; use_gpu = false; end
    end
    USE_GPU[] = use_gpu
    println("="^80)
    println("PRE-ROLL AFTERSTATE FINE-TUNE (Exp 1)")
    println("="^80)
    println("device: ", use_gpu ? "GPU" : "CPU", "  obs=$(A["obs_type"])  state_dim=$STATE_DIM")
    println("net:    $(A["net_ckpt"])  ($(A["width"])x$(A["blocks"]))")
    println("data:   $(A["data"])")
    flush(stdout)

    D = deserialize(A["data"])
    sib_states = D.sib_states; sib_labels = D.sib_labels; sib_dec = D.sib_dec
    ret_states = D.ret_states; ret_labels = D.ret_labels
    Ns = length(sib_states); Nr = length(ret_states)
    println("siblings=$Ns  retention=$Nr  meta=", D.meta); flush(stdout)

    # vectorize
    println("vectorizing..."); flush(stdout); t0 = time()
    Xs = Matrix{Float32}(undef, STATE_DIM, Ns)
    Ys = Matrix{Float32}(undef, 5, Ns)
    Threads.@threads for i in 1:Ns
        vectorize_state_into!(view(Xs,:,i), gspec, sib_states[i])
        l = sib_labels[i]; @inbounds for k in 1:5; Ys[k,i] = l[k]; end
    end
    Xr = Matrix{Float32}(undef, STATE_DIM, Nr)
    Yr = Matrix{Float32}(undef, 5, Nr)
    Threads.@threads for i in 1:Nr
        vectorize_state_into!(view(Xr,:,i), gspec, ret_states[i])
        l = ret_labels[i]; @inbounds for k in 1:5; Yr[k,i] = l[k]; end
    end
    println("  vectorized ($(round(time()-t0,digits=1))s)"); flush(stdout)

    # split BY DECISION
    udec = unique(sib_dec)
    shuffle!(udec)
    nval = round(Int, A["val_frac"]*length(udec))
    val_dec = Set(udec[1:nval]); train_dec = Set(udec[nval+1:end])
    train_sib = [i for i in 1:Ns if sib_dec[i] in train_dec]
    val_sib   = [i for i in 1:Ns if sib_dec[i] in val_dec]
    # val decision groups, indexed by LOCAL position within the val subset arrays
    local_of = Dict{Int,Int}(val_sib[k] => k for k in 1:length(val_sib))
    val_groups_full = Dict{Int32,Vector{Int}}()
    for i in val_sib; push!(get!(val_groups_full, sib_dec[i], Int[]), local_of[i]); end
    val_groups = collect(values(val_groups_full))
    @printf("decisions: train=%d val=%d | sibling: train=%d val=%d | val groups=%d\n",
            length(train_dec), length(val_dec), length(train_sib), length(val_sib), length(val_groups))
    flush(stdout)

    # move full arrays to device (fits on 4090)
    Xs_d = _to_dev(Xs); Xr_d = _to_dev(Xr)
    Ys_d = _to_dev(Ys); Yr_d = _to_dev(Yr)
    # val features on device for proxy
    Xval_d = _to_dev(Xs[:, val_sib])
    Yval = Ys[:, val_sib]                      # cpu labels for proxy (wildbg)

    net = _to_dev(load_net(A["net_ckpt"], A["width"], A["blocks"]))

    # ── baseline (OLD net) proxy + opening/mirror sanity ──
    base = ranking_proxy(net, Xval_d, Yval, val_groups)
    @printf("\nBASELINE (i140, pre-finetune)  val argmax-match=%.3f  equity-loss=%.4f  (ndec=%d)\n",
            base.argmax_match, base.equity_loss, base.ndec); flush(stdout)

    # opening probe
    open_state = BackgammonNet.initial_state(short_game=false, cube_enabled=false,
                                             jacoby_enabled=false, obs_type=OBS_SYM)
    open_eq_old = net_equity_one(net, open_state)
    @printf("OPENING pre-roll equity: i140=%+.4f  (wildbg teacher≈+0.11)\n", open_eq_old); flush(stdout)

    # ── ranking setup (Task 2 A/B): decision→sibling map + per-sibling wildbg my-equity ──
    rw = Float32(A["ranking_weight"]); rgap = Float32(A["ranking_gap"]); rcap = Float32(A["ranking_margin_cap"])
    wb_myeq = Float32[-eqjoint5(Ys[1,i],Ys[2,i],Ys[3,i],Ys[4,i],Ys[5,i]) for i in 1:Ns]
    dec2sibs = Dict{Int32,Vector{Int}}()
    if rw > 0f0
        for i in train_sib; push!(get!(dec2sibs, sib_dec[i], Int[]), i); end
        println("RANKING loss ENABLED: weight=$rw gap>$rgap margin-cap=$rcap  (grouped sampler, $(length(dec2sibs)) train decisions)")
    else
        println("RANKING loss DISABLED (regression only)")
    end
    flush(stdout)

    # ── training ──
    opt = Flux.setup(Flux.Adam(A["lr"]), net)
    bs = A["batch_size"]; rf = A["retention_frac"]
    n_ret_per = max(1, round(Int, bs*rf)); n_sib_per = bs - n_ret_per
    best_loss = Inf; best_state = deepcopy(Flux.cpu(net)); best_epoch = 0; since_improve = 0
    train_decs = collect(keys(dec2sibs))

    for epoch in 1:A["epochs"]
        te = time()
        running = 0.0; nb = 0
        if rw > 0f0
            # decision-grouped sampler: accumulate whole groups to ~n_sib_per siblings
            shuffle!(train_decs)
            gi = 1; ndec = length(train_decs)
            while gi <= ndec
                sidx = Int[]; group_bounds = Tuple{Int,Int}[]   # (start,stop) within sidx per group
                while gi <= ndec && length(sidx) < n_sib_per
                    g = dec2sibs[train_decs[gi]]; gi += 1
                    length(g) < 2 && (continue)
                    st = length(sidx) + 1
                    append!(sidx, g)
                    push!(group_bounds, (st, length(sidx)))
                end
                isempty(sidx) && continue
                # build pairs (better,worse) with wildbg gap>rgap, margin=min(gap,cap), batch-local positions
                pa = Int[]; pb = Int[]; marg = Float32[]
                for (st, sp) in group_bounds
                    for x in st:sp, y in (x+1):sp
                        ex = wb_myeq[sidx[x]]; ey = wb_myeq[sidx[y]]
                        d = ex - ey
                        if d > rgap; push!(pa,x); push!(pb,y); push!(marg, min(d, rcap))
                        elseif -d > rgap; push!(pa,y); push!(pb,x); push!(marg, min(-d, rcap)); end
                    end
                end
                ridx = rand(1:Nr, n_ret_per)
                Xb = hcat(Xs_d[:, sidx], Xr_d[:, ridx])
                Yb = hcat(Ys_d[:, sidx], Yr_d[:, ridx])
                nsib = length(sidx)
                pa_d = _to_dev(pa); pb_d = _to_dev(pb); marg_d = _to_dev(marg)
                l, gs = Flux.withgradient(m -> loss_fn_rank(m, Xb, Yb, nsib, pa_d, pb_d, marg_d, rw), net)
                Flux.update!(opt, net, gs[1]); running += l; nb += 1
            end
        else
        perm = shuffle(train_sib)
        nb = cld(length(perm), n_sib_per)
        for b in 1:nb
            lo = (b-1)*n_sib_per+1; hi = min(b*n_sib_per, length(perm))
            sidx = perm[lo:hi]
            ridx = rand(1:Nr, n_ret_per)
            Xb = hcat(Xs_d[:, sidx], Xr_d[:, ridx])
            Yb = hcat(Ys_d[:, sidx], Yr_d[:, ridx])
            l, gs = Flux.withgradient(m -> loss_fn(m, Xb, Yb), net)
            Flux.update!(opt, net, gs[1])
            running += l
        end
        end
        # val proxy + val BCE
        prox = ranking_proxy(net, Xval_d, Yval, val_groups)
        vbce = loss_fn(net, Xval_d, _to_dev(Yval))
        @printf("epoch %2d  train_loss=%.4f  val_BCE=%.4f  val argmax-match=%.3f  equity-loss=%.4f  (%.1fs)\n",
                epoch, running/nb, vbce, prox.argmax_match, prox.equity_loss, time()-te); flush(stdout)
        if prox.equity_loss < best_loss - 1e-5
            best_loss = prox.equity_loss; best_state = deepcopy(Flux.cpu(net))
            best_epoch = epoch; since_improve = 0
        else
            since_improve += 1
            if since_improve >= A["patience"]
                println("early stop (no proxy improvement for $(A["patience"]) epochs)"); break
            end
        end
    end

    # restore best (best_state is a standalone cpu copy captured at its epoch)
    cpu_net = best_state
    net = _to_dev(cpu_net)
    @printf("\nBEST epoch=%d  val equity-loss=%.4f (baseline %.4f)\n", best_epoch, best_loss, base.equity_loss)

    # ── final validation ──
    fin = ranking_proxy(net, Xval_d, Yval, val_groups)
    @printf("FINAL (best)  val argmax-match=%.3f (base %.3f)  equity-loss=%.4f (base %.4f)\n",
            fin.argmax_match, base.argmax_match, fin.equity_loss, base.equity_loss)

    open_eq_new = net_equity_one(net, open_state)
    @printf("OPENING pre-roll equity: NEW=%+.4f  OLD=%+.4f  wildbg teacher=+0.11\n", open_eq_new, open_eq_old)

    # SIGN/MAGNITUDE check — net vs wildbg teacher on held-out val afterstates
    # (both opponent-perspective). This is the substantive perspective validation.
    Pv = net_probs(net, Xval_d)
    net_eq_v = [eqjoint5(Pv[1,i],Pv[2,i],Pv[3,i],Pv[4,i],Pv[5,i]) for i in 1:size(Pv,2)]
    wb_eq_v  = [eqjoint5(Yval[1,i],Yval[2,i],Yval[3,i],Yval[4,i],Yval[5,i]) for i in 1:size(Yval,2)]
    dev = abs.(net_eq_v .- wb_eq_v)
    corr = Statistics.cor(net_eq_v, wb_eq_v)
    @printf("NET vs WILDBG (val afterstates, opp-perspective equity): mean|Δ|=%.4f  max=%.4f  corr=%.4f\n",
            mean(dev), maximum(dev), corr)
    # baseline (old net) agreement for contrast
    net_old = _to_dev(load_net(A["net_ckpt"], A["width"], A["blocks"]))
    Pv0 = net_probs(net_old, Xval_d)
    net_eq_v0 = [eqjoint5(Pv0[1,i],Pv0[2,i],Pv0[3,i],Pv0[4,i],Pv0[5,i]) for i in 1:size(Pv0,2)]
    @printf("  (OLD i140 direct pre-roll: mean|Δ|=%.4f  corr=%.4f)\n",
            mean(abs.(net_eq_v0 .- wb_eq_v)), Statistics.cor(net_eq_v0, wb_eq_v))

    # structural relabel-invariance (encoding perspective-clean): net(A)==net(relabel(A))
    rb = [relabel_mirror(sib_states[val_sib[k]]) for k in 1:min(64, length(val_sib))]
    Xrb = Matrix{Float32}(undef, STATE_DIM, length(rb))
    for (j,g) in enumerate(rb); vectorize_state_into!(view(Xrb,:,j), gspec, g); end
    Prb = net_probs(net, _to_dev(Xrb))
    releq = [eqjoint5(Prb[1,j],Prb[2,j],Prb[3,j],Prb[4,j],Prb[5,j]) for j in 1:length(rb)]
    @printf("RELABEL invariance net(A)==net(swap+flip A): mean|Δ|=%.5f (expect ~0; obs is perspective-clean)\n",
            mean(abs.(releq .- net_eq_v[1:length(rb)])))

    # save
    outp = A["out"]; mkpath(dirname(outp))
    FluxLib.save_weights(outp, cpu_net)
    println("\nSaved fine-tuned net → $outp")
    println("="^80)
end

main()
