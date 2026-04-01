#####
##### Allocation-free forward pass for FCResNetMultiHead
#####
# Extracts network weights into flat structures for thread-safe,
# allocation-free CPU inference. Uses pure Julia GEMM + LayerNorm
# to avoid BLAS thread contention in multi-worker self-play.
#####

module FastInference

export FastWeights, FastBuffers
export fast_forward_normalized!, extract_fast_weights, refresh_fast_weights!

#####
##### Weight extraction
#####

"""Pre-extracted weights from FCResNetMultiHead for allocation-free forward."""
struct FastWeights
    W_in::Matrix{Float32}
    b_in::Vector{Float32}
    ln_in_s::Vector{Float32}
    ln_in_b::Vector{Float32}
    res_W1::Vector{Matrix{Float32}}
    res_b1::Vector{Vector{Float32}}
    res_ln1_s::Vector{Vector{Float32}}
    res_ln1_b::Vector{Vector{Float32}}
    res_W2::Vector{Matrix{Float32}}
    res_b2::Vector{Vector{Float32}}
    res_ln2_s::Vector{Vector{Float32}}
    res_ln2_b::Vector{Vector{Float32}}
    ln_post_s::Vector{Float32}
    ln_post_b::Vector{Float32}
    W_vt::Matrix{Float32}
    b_vt::Vector{Float32}
    ln_vt_s::Vector{Float32}
    ln_vt_b::Vector{Float32}
    W_vh::Vector{Vector{Float32}}
    b_vh::Vector{Float32}
    W_p::Vector{Matrix{Float32}}
    b_p::Vector{Vector{Float32}}
    ln_p_s::Vector{Vector{Float32}}
    ln_p_b::Vector{Vector{Float32}}
    W_pout::Matrix{Float32}
    b_pout::Vector{Float32}
    num_blocks::Int
    num_policy_layers::Int
end

function _extract_ln(ln)
    return (Vector{Float32}(ln.diag.scale), Vector{Float32}(ln.diag.bias))
end

function _extract_dense(d)
    return (Matrix{Float32}(d.weight), Vector{Float32}(d.bias))
end

"""
    extract_fast_weights(nn) -> FastWeights

Extract weights from an FCResNetMultiHead network into a FastWeights struct.
The network must have: .common.layers, .vhead_trunk, .vhead_win/gw/bgw/gl/bgl,
.phead, .hyper.num_blocks, .hyper.depth_phead fields.
"""
function extract_fast_weights(nn)
    common = nn.common
    layers = common.layers
    input_chain = layers[1]
    W_in, b_in = _extract_dense(input_chain.layers[2])
    ln_in_s, ln_in_b = _extract_ln(input_chain.layers[3])

    num_blocks = nn.hyper.num_blocks
    res_W1 = Matrix{Float32}[]
    res_b1 = Vector{Float32}[]
    res_ln1_s = Vector{Float32}[]
    res_ln1_b = Vector{Float32}[]
    res_W2 = Matrix{Float32}[]
    res_b2 = Vector{Float32}[]
    res_ln2_s = Vector{Float32}[]
    res_ln2_b = Vector{Float32}[]

    for i in 1:num_blocks
        block = layers[1 + i]
        w1, b1_ = _extract_dense(block.dense1)
        s1, b1_ln = _extract_ln(block.ln1)
        w2, b2_ = _extract_dense(block.dense2)
        s2, b2_ln = _extract_ln(block.ln2)
        push!(res_W1, w1); push!(res_b1, b1_)
        push!(res_ln1_s, s1); push!(res_ln1_b, b1_ln)
        push!(res_W2, w2); push!(res_b2, b2_)
        push!(res_ln2_s, s2); push!(res_ln2_b, b2_ln)
    end

    ln_post_s, ln_post_b = _extract_ln(layers[1 + num_blocks + 1])

    vt = nn.vhead_trunk
    W_vt, b_vt = _extract_dense(vt.layers[1])
    ln_vt_s, ln_vt_b = _extract_ln(vt.layers[2])

    W_vh = Vector{Float32}[]
    b_vh = Float32[]
    for head in (nn.vhead_win, nn.vhead_gw, nn.vhead_bgw, nn.vhead_gl, nn.vhead_bgl)
        d = head.layers[1]
        push!(W_vh, vec(d.weight))
        push!(b_vh, d.bias[1])
    end

    phead = nn.phead
    n_policy = nn.hyper.depth_phead
    W_p = Matrix{Float32}[]
    b_p = Vector{Float32}[]
    ln_p_s = Vector{Float32}[]
    ln_p_b = Vector{Float32}[]
    for i in 1:n_policy
        base = (i - 1) * 3
        w, b = _extract_dense(phead.layers[base + 1])
        s, lb = _extract_ln(phead.layers[base + 2])
        push!(W_p, w); push!(b_p, b)
        push!(ln_p_s, s); push!(ln_p_b, lb)
    end
    W_pout, b_pout = _extract_dense(phead.layers[n_policy * 3 + 1])

    FastWeights(W_in, b_in, ln_in_s, ln_in_b,
                res_W1, res_b1, res_ln1_s, res_ln1_b,
                res_W2, res_b2, res_ln2_s, res_ln2_b,
                ln_post_s, ln_post_b,
                W_vt, b_vt, ln_vt_s, ln_vt_b,
                W_vh, b_vh,
                W_p, b_p, ln_p_s, ln_p_b, W_pout, b_pout,
                num_blocks, n_policy)
end

"""Refresh FastWeights in-place from a network (avoids reallocation)."""
function refresh_fast_weights!(fw::FastWeights, nn)
    fw_new = extract_fast_weights(nn)
    copyto!(fw.W_in, fw_new.W_in)
    copyto!(fw.b_in, fw_new.b_in)
    copyto!(fw.ln_in_s, fw_new.ln_in_s)
    copyto!(fw.ln_in_b, fw_new.ln_in_b)
    for i in 1:fw.num_blocks
        copyto!(fw.res_W1[i], fw_new.res_W1[i])
        copyto!(fw.res_b1[i], fw_new.res_b1[i])
        copyto!(fw.res_ln1_s[i], fw_new.res_ln1_s[i])
        copyto!(fw.res_ln1_b[i], fw_new.res_ln1_b[i])
        copyto!(fw.res_W2[i], fw_new.res_W2[i])
        copyto!(fw.res_b2[i], fw_new.res_b2[i])
        copyto!(fw.res_ln2_s[i], fw_new.res_ln2_s[i])
        copyto!(fw.res_ln2_b[i], fw_new.res_ln2_b[i])
    end
    copyto!(fw.ln_post_s, fw_new.ln_post_s)
    copyto!(fw.ln_post_b, fw_new.ln_post_b)
    copyto!(fw.W_vt, fw_new.W_vt)
    copyto!(fw.b_vt, fw_new.b_vt)
    copyto!(fw.ln_vt_s, fw_new.ln_vt_s)
    copyto!(fw.ln_vt_b, fw_new.ln_vt_b)
    for i in 1:5
        copyto!(fw.W_vh[i], fw_new.W_vh[i])
    end
    copyto!(fw.b_vh, fw_new.b_vh)
    for i in 1:fw.num_policy_layers
        copyto!(fw.W_p[i], fw_new.W_p[i])
        copyto!(fw.b_p[i], fw_new.b_p[i])
        copyto!(fw.ln_p_s[i], fw_new.ln_p_s[i])
        copyto!(fw.ln_p_b[i], fw_new.ln_p_b[i])
    end
    copyto!(fw.W_pout, fw_new.W_pout)
    copyto!(fw.b_pout, fw_new.b_pout)
end

#####
##### Pre-allocated buffers
#####

"""Pre-allocated buffers for allocation-free forward pass."""
struct FastBuffers
    h1::Matrix{Float32}
    h2::Matrix{Float32}
    skip::Matrix{Float32}
    vt::Matrix{Float32}
    p::Matrix{Float32}
    ln_mean::Vector{Float32}
    ln_rstd::Vector{Float32}
    result_vecs::Vector{Vector{Float32}}
    results::Vector{Tuple{Vector{Float32}, Float32}}
end

function FastBuffers(width::Int, nactions::Int, max_batch::Int)
    FastBuffers(
        zeros(Float32, width, max_batch),
        zeros(Float32, width, max_batch),
        zeros(Float32, width, max_batch),
        zeros(Float32, width, max_batch),
        zeros(Float32, nactions, max_batch),
        zeros(Float32, max_batch),
        zeros(Float32, max_batch),
        [Vector{Float32}(undef, nactions) for _ in 1:max_batch],
        Vector{Tuple{Vector{Float32}, Float32}}(undef, max_batch))
end

#####
##### Pure Julia GEMM + LayerNorm (thread-safe, no BLAS contention)
#####

function layernorm_relu!(out::AbstractMatrix, x::AbstractMatrix, scale::Vector, bias::Vector,
                         mean_buf::Vector, rstd_buf::Vector, n::Int)
    d = size(x, 1)
    inv_d = 1.0f0 / d
    @inbounds for j in 1:n
        m = 0.0f0
        @simd for i in 1:d
            m += x[i, j]
        end
        m *= inv_d

        v = 0.0f0
        @simd for i in 1:d
            diff = x[i, j] - m
            v += diff * diff
        end
        rs = 1.0f0 / sqrt(v * inv_d + 1.0f-5)

        @simd for i in 1:d
            val = scale[i] * (x[i, j] - m) * rs + bias[i]
            out[i, j] = max(0.0f0, val)
        end
    end
end

function _gemm_bias!(C::AbstractMatrix{Float32}, A::Matrix{Float32},
                     B::AbstractMatrix{Float32}, bias::Vector{Float32}, n::Int)
    m, k = size(A)
    @inbounds for j in 1:n
        @simd for i in 1:m
            C[i, j] = bias[i]
        end
    end

    tile_k = 64
    @inbounds for pk in 1:tile_k:k
        pk_end = min(pk + tile_k - 1, k)
        j = 1
        while j + 3 <= n
            for p in pk:pk_end
                b1 = B[p, j]; b2 = B[p, j+1]; b3 = B[p, j+2]; b4 = B[p, j+3]
                @simd for i in 1:m
                    a = A[i, p]
                    C[i, j]   += a * b1
                    C[i, j+1] += a * b2
                    C[i, j+2] += a * b3
                    C[i, j+3] += a * b4
                end
            end
            j += 4
        end
        while j <= n
            for p in pk:pk_end
                bp = B[p, j]
                @simd for i in 1:m
                    C[i, j] += A[i, p] * bp
                end
            end
            j += 1
        end
    end
end

# Pure Julia GEMM is used on all platforms. Benchmarks show:
# - ARM: 1.3x faster than Apple BLAS (AMX auto-vectorization via LLVM)
# - x86: BLAS is 1.6x faster single-threaded, but view allocations (96 bytes/call)
#   cause GC pressure under multi-threading that negates the GEMM speedup.
# Net result: pure Julia GEMM wins for production multi-worker selfplay on both platforms.

function dense!(out::AbstractMatrix, W::Matrix{Float32}, x::AbstractMatrix, b::Vector{Float32}, n::Int)
    _gemm_bias!(out, W, x, b, n)
end

function dense_relu!(out::AbstractMatrix, W::Matrix{Float32}, x::AbstractMatrix, b::Vector{Float32}, n::Int)
    d_out = size(W, 1)
    _gemm_bias!(out, W, x, b, n)
    @inbounds for j in 1:n
        @simd for i in 1:d_out
            out[i, j] = max(0.0f0, out[i, j])
        end
    end
end

#####
##### Forward pass
#####

"""
    fast_forward_normalized!(fw, fb, X, A, n) -> (P_masked, V_equity, n)

Allocation-free forward pass through FCResNetMultiHead.
Returns masked+normalized policy, scalar equity values, and batch size.

- `fw`: FastWeights extracted from network
- `fb`: FastBuffers pre-allocated for this worker
- `X`: Input states matrix (state_dim × n)
- `A`: Action mask matrix (num_actions × n), 1.0 for legal actions
- `n`: Batch size (number of active columns in X and A)
"""
function fast_forward_normalized!(fw::FastWeights, fb::FastBuffers,
                                   X::AbstractMatrix, A::AbstractMatrix, n::Int)
    w = size(fw.W_in, 1)

    # Input layer: Dense + LayerNorm + ReLU
    dense!(fb.h1, fw.W_in, X, fw.b_in, n)
    layernorm_relu!(fb.h2, fb.h1, fw.ln_in_s, fw.ln_in_b, fb.ln_mean, fb.ln_rstd, n)

    # Residual blocks: LN+ReLU → Dense → LN+ReLU → Dense + skip
    for blk in 1:fw.num_blocks
        @inbounds for j in 1:n
            @simd for i in 1:w
                fb.skip[i, j] = fb.h2[i, j]
            end
        end
        layernorm_relu!(fb.h1, fb.h2, fw.res_ln1_s[blk], fw.res_ln1_b[blk], fb.ln_mean, fb.ln_rstd, n)
        dense!(fb.h2, fw.res_W1[blk], fb.h1, fw.res_b1[blk], n)
        layernorm_relu!(fb.h1, fb.h2, fw.res_ln2_s[blk], fw.res_ln2_b[blk], fb.ln_mean, fb.ln_rstd, n)
        dense!(fb.h2, fw.res_W2[blk], fb.h1, fw.res_b2[blk], n)
        @inbounds for j in 1:n
            @simd for i in 1:w
                fb.h2[i, j] += fb.skip[i, j]
            end
        end
    end

    # Post-residual LayerNorm + ReLU
    layernorm_relu!(fb.h1, fb.h2, fw.ln_post_s, fw.ln_post_b, fb.ln_mean, fb.ln_rstd, n)

    # Value trunk: Dense + LayerNorm + ReLU
    dense!(fb.vt, fw.W_vt, fb.h1, fw.b_vt, n)
    layernorm_relu!(fb.h2, fb.vt, fw.ln_vt_s, fw.ln_vt_b, fb.ln_mean, fb.ln_rstd, n)

    # 5-head value computation: sigmoid per head → joint equity formula
    local V_equity = fb.ln_mean  # Reuse buffer
    wvh1 = fw.W_vh[1]; wvh2 = fw.W_vh[2]; wvh3 = fw.W_vh[3]
    wvh4 = fw.W_vh[4]; wvh5 = fw.W_vh[5]
    bvh1 = fw.b_vh[1]; bvh2 = fw.b_vh[2]; bvh3 = fw.b_vh[3]
    bvh4 = fw.b_vh[4]; bvh5 = fw.b_vh[5]
    @inbounds for j in 1:n
        p_win = bvh1; p_wg = bvh2; p_wbg = bvh3; p_lg = bvh4; p_lbg = bvh5
        @simd for i in 1:w
            v = fb.h2[i, j]
            p_win += wvh1[i] * v
            p_wg += wvh2[i] * v
            p_wbg += wvh3[i] * v
            p_lg += wvh4[i] * v
            p_lbg += wvh5[i] * v
        end
        # Apply sigmoid to raw logits
        p_win = 1.0f0 / (1.0f0 + exp(-p_win))
        p_wg = 1.0f0 / (1.0f0 + exp(-p_wg))
        p_wbg = 1.0f0 / (1.0f0 + exp(-p_wbg))
        p_lg = 1.0f0 / (1.0f0 + exp(-p_lg))
        p_lbg = 1.0f0 / (1.0f0 + exp(-p_lbg))
        # Joint equity formula: (2pw-1) + (wg-lg) + (wbg-lbg), normalized to [-1,1]
        V_equity[j] = ((2.0f0 * p_win - 1.0f0) + (p_wg - p_lg) + (p_wbg - p_lbg)) / 3.0f0
    end

    # Policy head: Dense+LN+ReLU layers → final Dense → masked softmax
    @views for i in 1:fw.num_policy_layers
        dense!(fb.vt, fw.W_p[i], fb.h1, fw.b_p[i], n)
        layernorm_relu!(fb.h1, fb.vt, fw.ln_p_s[i], fw.ln_p_b[i], fb.ln_mean, fb.ln_rstd, n)
    end
    nact = size(fw.W_pout, 1)
    dense!(fb.p, fw.W_pout, fb.h1, fw.b_pout, n)

    # Masked softmax over legal actions
    @inbounds for j in 1:n
        max_val = -Inf32
        for i in 1:nact
            if A[i, j] > 0.0f0 && fb.p[i, j] > max_val
                max_val = fb.p[i, j]
            end
        end
        s = 0.0f0
        for i in 1:nact
            if A[i, j] > 0.0f0
                fb.p[i, j] = exp(fb.p[i, j] - max_val)
                s += fb.p[i, j]
            else
                fb.p[i, j] = 0.0f0
            end
        end
        inv_s = 1.0f0 / (s + 1.0f-7)
        @simd for i in 1:nact
            fb.p[i, j] *= inv_s
        end
    end

    return fb.p, V_equity, n
end

end  # module FastInference
