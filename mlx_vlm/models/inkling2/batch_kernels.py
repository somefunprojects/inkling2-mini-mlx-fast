# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""Kernels for B>1 (M-row) decode. INKLING_BATCHFAST tokens; see fast_decode.FastDecoder.bf.

head_rows: LM head, BF16 weight x FP32 activation, FP32 accumulate, for M = 2..32 rows. Same per-lane summation order
as fast_kernels.lm_head_gemv (lane owns d8 = lane + 32k, k ascending; acc += dot(lo) + dot(hi); simd_sum), so every
row is bitwise equal to lm_head_gemv (and to its M=1 call). Difference: the activation chunk for all M rows is staged
once per threadgroup in threadgroup memory (lm_head_gemv re-reads h from device for every row and lane, which made
it load-issue bound at M=16: 3.4 ms in situ vs a 0.84 ms byte floor).
"""
import mlx.core as mx

_HEAD_ROWS_SRC = r"""
    // h: [M, D] float32, w: [V, D] bfloat16, out: [M, VOUT] float32.
    // TG = NSG simdgroups; simdgroup s owns vocab rows row0..row0+ROWS-1.
    threadgroup float4 hs[M * 64];            // one 256-float chunk of every row (M*1 KB)
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32 + lane;
    const int row0 = (int)((threadgroup_position_in_grid.x * NSG + sg) * ROWS);
    constexpr int D8 = D / 8;
    constexpr int NCH = (D8 + 31) / 32;
    const device uint4* wv = (const device uint4*)w;
    const device float4* hv = (const device float4*)h;
    float acc[ROWS][M];
    for (int r = 0; r < ROWS; ++r)
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    int rows[ROWS];
    for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, VOUT - 1);
    for (int c = 0; c < NCH; ++c) {
        // stage h[:, 256c : 256c+256] (64 float4 per row)
        for (int j = (int)tid; j < M * 64; j += NSG * 32) {
            const int m = j / 64, q = j % 64;
            const int src = c * 64 + q;   // float4 index within the row
            hs[j] = (src < D / 4) ? hv[(size_t)m * (D / 4) + src] : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int i = c * 32 + (int)lane;   // d8 index, same as lm_head_gemv's i = lane + 32c
        if (i < D8 && row0 < VOUT) {
            float4 a[ROWS], b[ROWS];
            for (int r = 0; r < ROWS; ++r) {
                uint4 u = wv[(size_t)rows[r] * D8 + i];
                a[r] = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                              as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
                b[r] = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                              as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
            }
            for (int m = 0; m < M; ++m) {
                float4 h0 = hs[m * 64 + 2 * (int)lane];
                float4 h1 = hs[m * 64 + 2 * (int)lane + 1];
                for (int r = 0; r < ROWS; ++r)
                    acc[r][m] += dot(a[r], h0) + dot(b[r], h1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 >= VOUT) return;
    const float cp = cap[0];
    for (int r = 0; r < ROWS; ++r) {
        for (int m = 0; m < M; ++m) {
            float s = simd_sum(acc[r][m]);
            if (lane == 0 && row0 + r < VOUT) {
                if (cp > 0.0f) s = metal::precise::tanh(s / cp) * cp;
                out[(size_t)m * VOUT + row0 + r] = s;
            }
        }
    }
"""

_k = {}
NSG = 8


def _head_kernel():
    k = _k.get("head")
    if k is None:
        k = _k["head"] = mx.fast.metal_kernel(
            name="inkling2_batch_head_rows", input_names=["h", "w", "cap"], output_names=["out"],
            source=_HEAD_ROWS_SRC, compile_options={"math_mode": "safe"})
    return k


def head_rows_default(m):
    return 4 if m <= 8 else 2


def head_supported(m, d, weight):
    return 2 <= m <= 32 and d % 256 == 0 and weight.dtype == mx.bfloat16 and weight.shape[1] == d


def head_rows(h, weight, vout=None, cap=0.0, rows=None):
    """h [..., D] (promoted to FP32), weight [V, D] bf16 -> FP32 logits [..., vout] = softcap(h @ w[:vout].T)."""
    lead, D = h.shape[:-1], h.shape[-1]
    vout = weight.shape[0] if vout is None else int(vout)
    h2 = h.reshape(-1, D).astype(mx.float32)
    M = h2.shape[0]
    rows = rows or head_rows_default(M)
    n_tg = (vout + NSG * rows - 1) // (NSG * rows)
    if not isinstance(cap, mx.array):
        cap = mx.array([float(cap or 0.0)], mx.float32)
    out = _head_kernel()(
        inputs=[h2, weight, cap],
        template=[("M", M), ("D", D), ("ROWS", rows), ("NSG", NSG), ("VOUT", vout)],
        grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(M, vout)], output_dtypes=[mx.float32])[0]
    return out.reshape(*lead, vout)


# ==========================================================================
# B-row decode SDPA (L = 1): rel-bias mask + SDPA + sink + output gate in one launch per visit,
# derived from spec_attn.sdpa_rows (v2, specfast-attn 8bf01e9) with the L query rows replaced by
# B batch rows (each with its own position qpos[b], its own K/V buffer and slot_pos row).
# Threadgroup = (row b, q head h, position block blk); KS simdgroups walk the block's positions
# lo..qpos[b] (lo = max(q0 - W + 1, 0) sliding, 0 global): empty trailing slots are never read.
# Same arithmetic as sdpa_rows per row (NONBITWISE vs the stock mask + mx.fast SDPA: rel bias by
# butterfly sum, online softmax over KS slices merged in threadgroup memory, NB blocks merged by
# a combine launch). FP32-exact up to ordering.
# ==========================================================================
_BSDPA_SRC = r"""
    constexpr int QPT = DH / 32;
    constexpr int G = HQ / HK;
    constexpr int RP = 32 / DR;
    constexpr int NT = KS * 32;
    threadgroup float tm[KS];
    threadgroup float ts[KS];
    threadgroup float to[KS * DH];
    const int ks = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int tid = (int)thread_position_in_threadgroup.x;
    const int h = (int)threadgroup_position_in_grid.y;
    const int bz = (int)threadgroup_position_in_grid.z;
    const int b = bz / NB, blk = bz % NB;
    const int kvh = h / G;
    const int d0 = lane * QPT;
    const int q0 = qpos[b];
    const int lo = WINDOW > 0 ? metal::max(q0 - WINDOW + 1, 0) : 0;
    const int cnt = q0 - lo + 1;
    const int per = (cnt + NB - 1) / NB;
    const int blo = lo + blk * per;
    const int bhi = metal::min(q0, blo + per - 1);
    const float sc = scale[0];
    float qv[QPT], o[QPT];
    #pragma clang loop unroll(full)
    for (int i = 0; i < QPT; ++i) {
        qv[i] = sc * float(q[(size_t)b * q_strides[0] + (size_t)h * q_strides[2]
                             + (size_t)(d0 + i) * q_strides[3]]);
        o[i] = 0.0f;
    }
    const int rq = lane / DR, rd = lane % DR;
    const float relv = float(rel[(size_t)b * rel_strides[0] + (size_t)h * rel_strides[2]
                                 + (size_t)rd * rel_strides[3]]);
    float m = -3.402823466e+38f, s = 0.0f;
    const device T* kh = kb + ((size_t)b * HK + kvh) * S1 * DH + d0;
    const device T* vh = vb + ((size_t)b * HK + kvh) * S1 * DH + d0;
    const int nb = bhi - blo + 1;
    for (int i0 = ks; i0 < nb; i0 += KS * TK) {
        float kr[TK][QPT], vr[TK][QPT], pj[TK / RP];
        bool ok[TK];
        #pragma clang loop unroll(full)
        for (int t = 0; t < TK; ++t) {
            const int i = i0 + t * KS;
            const bool in = i < nb;
            const int p = blo + (in ? i : 0);
            const int slot = SLIDING ? p % RING : p;
            const int dist = q0 - p;
            ok[t] = in && dist >= 0 && (WINDOW == 0 || dist < WINDOW)
                    && (!SLIDING || slot_pos[(size_t)b * slot_pos_strides[0]
                                             + (size_t)slot * slot_pos_strides[1]] == p);
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; ++j) kr[t][j] = float(kh[(size_t)slot * DH + j]);
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; ++j) vr[t][j] = float(vh[(size_t)slot * DH + j]);
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < TK / RP; ++g) {
            const int i = i0 + (g * RP + rq) * KS;
            const int p = blo + (i < nb ? i : 0);
            const int dist = q0 - p;
            const int dc = metal::clamp(dist, 0, REXT - 1);
            const float pv = float(static_cast<T>(projT[(size_t)dc * DR + rd]));
            pj[g] = (dist >= 0 && dist < REXT) ? relv * pv : 0.0f;
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < TK / RP; ++g) {
            float prod = pj[g];
            #pragma clang loop unroll(full)
            for (ushort off = DR / 2; off >= 1; off >>= 1) prod += simd_shuffle_xor(prod, off);
            pj[g] = prod;
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < TK; ++t) {
            float a = 0.0f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; ++j) a += qv[j] * kr[t][j];
            a = simd_sum(a);
            const float bias = simd_shuffle(pj[t / RP], (ushort)((t % RP) * DR));
            if (!ok[t]) continue;
            const float score = a + float(static_cast<T>(bias));
            const float nm = metal::max(m, score);
            const float f = fast::exp(m - nm);
            const float e = fast::exp(score - nm);
            m = nm;
            s = s * f + e;
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; ++j) o[j] = o[j] * f + e * vr[t][j];
        }
    }
    if (lane == 0) { tm[ks] = m; ts[ks] = s; }
    #pragma clang loop unroll(full)
    for (int i = 0; i < QPT; ++i) to[ks * DH + d0 + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < DH; d += NT) {
        float M = -3.402823466e+38f;
        for (int j = 0; j < KS; ++j)
            if (ts[j] > 0.0f) M = metal::max(M, tm[j]);
        float snk = 0.0f;
        if (NB == 1) {
            snk = float(sinks[(size_t)b * sinks_strides[0] + (size_t)h * sinks_strides[2]]);
            M = metal::max(M, snk);
        }
        float S = 0.0f, O = 0.0f;
        for (int j = 0; j < KS; ++j) {
            if (ts[j] > 0.0f) {
                const float f = fast::exp(tm[j] - M);
                S += ts[j] * f;
                O += to[j * DH + d] * f;
            }
        }
/*EPI*/
    }
"""

_BEPI1 = r"""
        S += fast::exp(snk - M);
        const T ov = static_cast<T>(O / S);
        T res = ov;
        if (GATE) {
            const float g = float(gate[(size_t)b * gate_strides[0] + (size_t)(h * DH + d) * gate_strides[2]]);
            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
            const float sgm = g < 0.0f ? y : 1.0f - y;
            res = static_cast<T>(float(ov) * sgm);
        }
        out[((size_t)b * HQ + h) * DH + d] = res;
"""
_BEPIN = r"""
        const size_t pb = ((size_t)b * HQ + h) * NB + blk;
        po[pb * DH + d] = O;
        if (d == 0) { pm[pb] = M; ps[pb] = S; }
"""
_BCOMBINE_SRC = r"""
    const int tid = (int)thread_position_in_grid.x;
    const int bh = (int)thread_position_in_grid.y;
    const int b = bh / HQ, h = bh % HQ;
    const float snk = float(sinks[(size_t)b * sinks_strides[0] + (size_t)h * sinks_strides[2]]);
    float M = snk;
    for (int k = 0; k < NB; ++k) if (ps[bh * NB + k] > 0.0f) M = metal::max(M, pm[bh * NB + k]);
    float S = fast::exp(snk - M), O = 0.0f;
    for (int k = 0; k < NB; ++k) {
        if (ps[bh * NB + k] > 0.0f) {
            const float f = fast::exp(pm[bh * NB + k] - M);
            S += ps[bh * NB + k] * f;
            O += po[((size_t)bh * NB + k) * DH + tid] * f;
        }
    }
    const T ov = static_cast<T>(O / S);
    T res = ov;
    if (GATE) {
        const float g = float(gate[(size_t)b * gate_strides[0] + (size_t)(h * DH + tid) * gate_strides[2]]);
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
        const float sgm = g < 0.0f ? y : 1.0f - y;
        res = static_cast<T>(float(ov) * sgm);
    }
    out[(size_t)bh * DH + tid] = res;
"""

import os as _os

BSDPA_KS = int(_os.environ.get("INKLING_BSDPA_KS", "4"))
BSDPA_TK = int(_os.environ.get("INKLING_BSDPA_TK", "4"))
BSDPA_BLOCK = int(_os.environ.get("INKLING_BSDPA_BLOCK", "256"))
_FLOATS = (mx.bfloat16, mx.float16, mx.float32)


def _kern(name, src, inputs, outputs):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=src,
                                            ensure_row_contiguous=False)
    return k


def sdpa_supported(attn, q, kb, r, sinks):
    """q [B, 1, HQ, DH] rows (pre-transpose), kb [B, HK, S1, DH] contiguous state buffer."""
    if not (mx.metal.is_available() and mx.default_device() == mx.gpu) or q.ndim != 4:
        return False
    B, L, HQ, DH = q.shape
    HK = kb.shape[1]
    DR = r.shape[-1]
    return (
        L == 1 and B >= 1 and kb.shape[0] == B
        and q.dtype in _FLOATS and kb.dtype == q.dtype
        and DH % 32 == 0 and DH // 32 <= 8 and kb.shape[3] == DH
        and HQ % HK == 0
        and DR <= 32 and 32 % DR == 0
        and tuple(r.shape) == (B, 1, HQ, DR)
        and tuple(sinks.shape) == (B, 1, HQ)
        and "_sa_projT" in attn.__dict__
    )


def sdpa_brows(attn, q, kb, vb, r, slot_pos, qpos, sinks, gate, ring, scale, cap):
    """q [B,1,HQ,DH], kb/vb [B,HK,S1,DH] (this step's rows written), r [B,1,HQ,DR], slot_pos [B, ring]
    (sliding) or None, qpos [B] int32 positions of the query rows, sinks [B,1,HQ], gate [B,1,HQ*DH] or
    None. Returns [B, 1, HQ*DH] = gated attention output."""
    from . import spec_attn as sa
    B, _, HQ, DH = q.shape
    HK, S1 = kb.shape[1], kb.shape[2]
    DR = r.shape[-1]
    projT = sa.proj_t(attn)
    R = projT.shape[0]
    window = int(attn.sliding)
    sliding = bool(attn.is_sliding)
    dt = q.dtype
    if r.dtype != dt:
        r = r.astype(dt)
    nmax = min(window, cap + 1) if window > 0 else min(S1, cap + 1)
    NB = max(1, -(-nmax // BSDPA_BLOCK))
    KS = BSDPA_KS
    NT = 32 * KS
    TK = -(-max(BSDPA_TK, 32 // DR) // (32 // DR)) * (32 // DR)
    has_gate = gate is not None
    tmpl = [("T", dt), ("HQ", HQ), ("HK", HK), ("DH", DH), ("DR", DR), ("REXT", R), ("WINDOW", window),
            ("SLIDING", sliding), ("RING", int(ring)), ("S1", S1), ("NB", NB), ("KS", KS), ("TK", TK),
            ("GATE", has_gate)]
    qpos = qpos.reshape(-1).astype(mx.int32)
    sp = slot_pos.astype(mx.int32) if sliding else qpos.reshape(B, 1)
    g = gate if has_gate else q
    sc = mx.array([float(scale)], mx.float32)
    kb, vb = mx.contiguous(kb), mx.contiguous(vb)
    ins = [q, kb, vb, r, projT, sp, qpos, sinks, g, sc]
    names = ["q", "kb", "vb", "rel", "projT", "slot_pos", "qpos", "sinks", "gate", "scale"]
    if NB == 1:
        k = _kern("inkling2_b_sdpa1", _BSDPA_SRC.replace("/*EPI*/", _BEPI1), names, ["out"])
        return k(inputs=ins, template=tmpl, grid=(NT, HQ, B), threadgroup=(NT, 1, 1),
                 output_shapes=[(B, 1, HQ * DH)], output_dtypes=[dt])[0]
    k = _kern("inkling2_b_sdpaN", _BSDPA_SRC.replace("/*EPI*/", _BEPIN), names, ["po", "pm", "ps"])
    po, pm, ps = k(inputs=ins, template=tmpl, grid=(NT, HQ, NB * B), threadgroup=(NT, 1, 1),
                   output_shapes=[(B * HQ, NB, DH), (B * HQ, NB), (B * HQ, NB)], output_dtypes=[mx.float32] * 3)
    kc = _kern("inkling2_b_sdpa_combine", _BCOMBINE_SRC, ["po", "pm", "ps", "sinks", "gate"], ["out"])
    return kc(inputs=[po, pm, ps, sinks, g],
              template=[("T", dt), ("HQ", HQ), ("DH", DH), ("NB", NB), ("GATE", has_gate)],
              grid=(DH, B * HQ, 1), threadgroup=(DH, 1, 1),
              output_shapes=[(B, 1, HQ * DH)], output_dtypes=[dt])[0]


# ==========================================================================
# B-row tail (post / pre / depth attention) for L = 1 decode rows, from spec_attn._tail_source.
# Row l of the grid = batch row b with its OWN conv state [B, W-1, D] (spec's L-row version runs
# the rows causally through one shared state). Outputs as tail_rows, except ctx = the row's NEW
# conv state [B, W-1, D] (state[1:] + f32(x)) instead of the rollback context.
# Same per-row arithmetic as tail_rows (NONBITWISE vs the stock split path: depth-attention sums
# in a different order; sconv value serial over lags as stock).
# ==========================================================================
def _tail_brows_source(nk):
    from . import spec_attn as sa
    src = sa._tail_source(nk)
    reps = [
        ("            const int t = l + j;\n"
         "            const float xv = t < WS ? state[(size_t)t * D + c] : float(xin[(size_t)(t - WS) * D + c]);",
         "            const float xv = j < WS ? state[(size_t)l * WS * D + (size_t)j * D + c] : hv;"),
        ("        ctx[(size_t)(WS + l) * D + c] = hv;\n"
         "        if (l == 0)\n"
         "            for (int j = 0; j < WS; ++j) ctx[(size_t)j * D + c] = state[(size_t)j * D + c];",
         "        for (int j = 0; j + 1 < WS; ++j)\n"
         "            ctx[(size_t)l * WS * D + (size_t)j * D + c] = state[(size_t)l * WS * D + (size_t)(j + 1) * D + c];\n"
         "        ctx[(size_t)l * WS * D + (size_t)(WS - 1) * D + c] = hv;"),
    ]
    for a, b in reps:
        assert src.count(a) == 1, ("tail source drifted", a[:60])
        src = src.replace(a, b)
    return src


def tail_brows_supported(module, state, xin, n_keys, query):
    if not (mx.metal.is_available() and mx.default_device() == mx.gpu) or xin.ndim != 3:
        return False
    B, L, D = xin.shape
    if L != 1 or not (2 <= B <= 32) or xin.dtype not in _FLOATS or D % 128 or D // 4 > 1024:
        return False
    if not (query.shape[-1] == D and (query.ndim == 1 or (query.ndim == 2 and query.shape[0] <= 4))):
        return False
    if module is None:
        return n_keys + 14 <= 31
    W = getattr(module, "kernel_size", 0)
    w = getattr(getattr(module, "conv", None), "weight", None)
    return (W >= 2 and w is not None and w.size == D * W and tuple(state.shape) == (B, W - 1, D)
            and state.dtype == mx.float32 and n_keys + 14 <= 31)


def tail_brows(mode, module, state, xin, skip, keys, query, boundary=False, norm=None, want_hid=True,
               out_dtype=None):
    """As spec_attn.tail_rows for B rows of L=1 (xin [B,1,D], state [B,W-1,D] f32, keys [B,1,1,D]).
    Returns dict(ctx = new conv state [B,W-1,D] f32 (modes 0/1), res, hid [B,1,NQ,D], bkey, hn)."""
    import math
    from . import spec_attn as sa
    B, _, D = xin.shape
    W = module.kernel_size if mode != 2 else 4
    nk = len(keys)
    NQ = 1 if query.ndim == 1 else query.shape[0]
    if mode == 2:
        state = skip = xin
        weight = query
    else:
        weight = module.conv.weight
    T = xin.dtype if mode == 0 else (out_dtype or xin.dtype)
    RT = mx.float32 if mode == 0 else T
    has_norm = norm is not None
    bkey = bool(boundary) and mode == 0
    k = sa._kernel(f"inkling2_b_tail_{nk}", _tail_brows_source(nk),
                   [f"k{i}" for i in range(nk)]
                   + ["state", "weight", "xin", "skip", "query", "scale", "divisor", "nw", "eps"],
                   ["ctx", "res", "hid", "bkey", "hn"], contiguous=True)
    outs = k(
        inputs=[*keys, state, weight, xin, skip, query, sa._scalar(D ** -0.5),
                sa._scalar(math.sqrt(W + 1)), norm.weight if has_norm else query,
                sa._scalar(norm.eps if has_norm else 1e-6)],
        template=[("T", T), ("RT", RT), ("D", D), ("TG", D // 4), ("NK", nk), ("W", W),
                  ("MODE", int(mode)), ("BKEY", bkey), ("NORM", has_norm),
                  ("HID", bool(want_hid)), ("NQ", NQ)],
        grid=(D // 4, B, 1), threadgroup=(D // 4, 1, 1),
        output_shapes=[(B, W - 1, D) if mode != 2 else (1,), (B, 1, D) if mode != 2 else (1,),
                       (B, 1, NQ, D) if want_hid else (1,),
                       (B, 1, 1, D) if bkey else (1,), (B, 1, D) if has_norm else (1,)],
        output_dtypes=[mx.float32, RT, mx.float32, mx.float32, T],
    )
    res = {} if mode == 2 else {"ctx": outs[0], "res": outs[1]}
    if want_hid:
        res["hid"] = outs[2]
    if bkey:
        res["bkey"] = outs[3]
    if has_norm:
        res["hn"] = outs[4]
    return res
