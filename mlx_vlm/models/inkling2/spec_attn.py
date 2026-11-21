# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""specfast-attn: attention-side kernels for the L>1 verify step (B=1).

INKLING_SPECFAST tokens (FastDecoder.sf), active only in verify (``_prefixes is not None``):
  sdpa  one launch per visit for rel-bias mask + SDPA + sink + output gate (no [H, L, S+1]
        mask tensor, no output transpose, no output_gate launch). GQA: the G q heads of a kv
        head read the same K/V rows (L2-shared). Long buffers (> ~1.1k positions) split the
        positions over NB threadgroups per head, then a small combine launch.
  post  attn sconv tail (+ residual) + boundary key + mlp_input depth attention
        (+ post_attention_layernorm) per row in one launch; also emits the conv context
        [1, W-1+L, D] the commit gathers (replaces the separate concat).
  pre   mlp sconv tail + residual + the NEXT visit's attention_input depth attention
        (+ its input_layernorm) per row in one launch (the L-row mega_pre); also emits the
        conv context.
Class NONBITWISE-OK: every value is the stock expression; only summation orders differ
(FP32-exact up to ordering). Kernels are templated on dtype so the FP32 tiny model runs them.
"""
import math
import os

import mlx.core as mx

_k = {}


def _kernel(name, src, inputs, outputs, contiguous=False):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=outputs, source=src,
            ensure_row_contiguous=contiguous)
    return k


def _scalar(v):
    from . import fast_kernels as fk
    return fk._scalar(v)


def _on_gpu():
    return mx.metal.is_available() and mx.default_device() == mx.gpu


_FLOATS = (mx.bfloat16, mx.float16, mx.float32)

# ==========================================================================
# 1. sdpa: L query rows x all cached positions, rel-bias + sink + gate inline
# ==========================================================================
# v2 (register-light): grid (32*L*KS, HQ, NB); threadgroup = one q head h and one slice
# (blk) of the positions [lo, hi] (lo = max(q0 - W + 1, 0) sliding / 0 global,
# hi = q0 + L - 1). Simdgroup (l, ks) owns query row l and walks positions blo + ks,
# blo + ks + KS, ... (the L row-simdgroups of a slice read the same K/V rows: L1-shared;
# the G q heads of a kv head: L2-shared). Per position: slot (global: p; sliding: p % RING,
# skipped unless slot_pos[slot] == p, which is the stock mask's p < 0 test), the rel bias
# (lanes = (position, d_rel) pairs, butterfly-summed: ordering-only vs the serial mask loop;
# rounded to T like the stock mask), score = simd_sum(scale*q . k) + bias, the stock
# sdpa_vector online-softmax update; masked positions (dist < 0 or >= window; stock -1e30
# -> e = 0) skipped. TK positions' loads are issued before their updates.
# Epilogue: the KS partials of each row merged through threadgroup memory; NB == 1: the
# sink (score = sink logit, value 0) joins, o/sum -> T -> * sigmoid(gate) -> T, written
# [1, L, HQ*DH]; NB > 1: partials (o, max, sum) for the combine launch.
_SDPA_SRC = r"""
    constexpr int QPT = DH / 32;
    constexpr int G = HQ / HK;
    constexpr int RP = 32 / DR;   // positions per bias pass
    constexpr int NSG = L * KS;
    constexpr int NT = NSG * 32;
    threadgroup float tm[NSG];
    threadgroup float ts[NSG];
    threadgroup float to[NSG * DH];
    const int sgi = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int tid = (int)thread_position_in_threadgroup.x;
    const int h = (int)threadgroup_position_in_grid.y;
    const int blk = (int)threadgroup_position_in_grid.z;
    const int l = sgi / KS, ks = sgi % KS;
    const int kvh = h / G;
    const int d0 = lane * QPT;
    const int q0 = qpos[0];
    const int ql = q0 + l;
    const int hi = q0 + L - 1;
    const int lo = WINDOW > 0 ? metal::max(q0 - WINDOW + 1, 0) : 0;
    const int cnt = hi - lo + 1;
    const int per = (cnt + NB - 1) / NB;
    const int blo = lo + blk * per;
    const int bhi = metal::min(hi, blo + per - 1);
    const float sc = scale[0];
    float qv[QPT], o[QPT];
    #pragma clang loop unroll(full)
    for (int i = 0; i < QPT; ++i) {
        qv[i] = sc * float(q[(size_t)l * q_strides[1] + (size_t)h * q_strides[2]
                             + (size_t)(d0 + i) * q_strides[3]]);
        o[i] = 0.0f;
    }
    const int rq = lane / DR, rd = lane % DR;
    const float relv = float(rel[(size_t)l * rel_strides[1] + (size_t)h * rel_strides[2]
                                 + (size_t)rd * rel_strides[3]]);
    float m = -3.402823466e+38f, s = 0.0f;
    const device T* kh = kb + (size_t)kvh * S1 * DH + d0;
    const device T* vh = vb + (size_t)kvh * S1 * DH + d0;
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
            const int dist = ql - p;
            ok[t] = in && dist >= 0 && (WINDOW == 0 || dist < WINDOW)
                    && (!SLIDING || slot_pos[(size_t)slot * slot_pos_strides[1]] == p);
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; ++j) kr[t][j] = float(kh[(size_t)slot * DH + j]);
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; ++j) vr[t][j] = float(vh[(size_t)slot * DH + j]);
        }
        #pragma clang loop unroll(full)
        for (int g = 0; g < TK / RP; ++g) {
            // lane (rq, rd): position t = g*RP + rq of this batch, d_rel index rd
            const int i = i0 + (g * RP + rq) * KS;
            const int p = blo + (i < nb ? i : 0);
            const int dist = ql - p;
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
    if (lane == 0) { tm[sgi] = m; ts[sgi] = s; }
    #pragma clang loop unroll(full)
    for (int i = 0; i < QPT; ++i) to[sgi * DH + d0 + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < L * DH; idx += NT) {
        const int lr = idx / DH, d = idx % DH;
        float M = -3.402823466e+38f;
        for (int j = 0; j < KS; ++j)
            if (ts[lr * KS + j] > 0.0f) M = metal::max(M, tm[lr * KS + j]);
        float snk = 0.0f;
        if (NB == 1) {
            snk = float(sinks[(size_t)lr * sinks_strides[1] + (size_t)h * sinks_strides[2]]);
            M = metal::max(M, snk);
        }
        float S = 0.0f, O = 0.0f;
        for (int j = 0; j < KS; ++j) {
            const int sj = lr * KS + j;
            if (ts[sj] > 0.0f) {
                const float f = fast::exp(tm[sj] - M);
                S += ts[sj] * f;
                O += to[sj * DH + d] * f;
            }
        }
/*EPI*/
    }
"""

_EPI1 = r"""
        S += fast::exp(snk - M);
        const T ov = static_cast<T>(O / S);
        T res = ov;
        if (GATE) {
            const float g = float(gate[(size_t)lr * gate_strides[1]
                                       + (size_t)(h * DH + d) * gate_strides[2]]);
            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
            const float sgm = g < 0.0f ? y : 1.0f - y;
            res = static_cast<T>(float(ov) * sgm);
        }
        out[((size_t)lr * HQ + h) * DH + d] = res;
"""
_EPIN = r"""
        const size_t pb = ((size_t)lr * HQ + h) * NB + blk;
        po[pb * DH + d] = O;
        if (d == 0) { pm[pb] = M; ps[pb] = S; }
"""

_COMBINE_SRC = r"""
    const int tid = (int)thread_position_in_grid.x;
    const int lh = (int)thread_position_in_grid.y;
    const int l = lh / HQ, h = lh % HQ;
    const float snk = float(sinks[(size_t)l * sinks_strides[1] + (size_t)h * sinks_strides[2]]);
    float M = snk;
    for (int b = 0; b < NB; ++b) if (ps[lh * NB + b] > 0.0f) M = metal::max(M, pm[lh * NB + b]);
    float S = fast::exp(snk - M), O = 0.0f;
    for (int b = 0; b < NB; ++b) {
        if (ps[lh * NB + b] > 0.0f) {
            const float f = fast::exp(pm[lh * NB + b] - M);
            S += ps[lh * NB + b] * f;
            O += po[((size_t)lh * NB + b) * DH + tid] * f;
        }
    }
    const T ov = static_cast<T>(O / S);
    T res = ov;
    if (GATE) {
        const float g = float(gate[(size_t)l * gate_strides[1] + (size_t)(h * DH + tid) * gate_strides[2]]);
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
        const float sgm = g < 0.0f ? y : 1.0f - y;
        res = static_cast<T>(float(ov) * sgm);
    }
    out[(size_t)lh * DH + tid] = res;
"""

SDPA_KS = int(os.environ.get("INKLING_SA_KS", "4"))  # key slices per row
SDPA_TK = int(os.environ.get("INKLING_SA_TK", "4"))
SDPA_BLOCK = int(os.environ.get("INKLING_SA_BLOCK", "256"))  # positions per threadgroup (NB = ceil(positions / BLOCK))


def proj_t(attn):
    """rel_proj^T [R, d_rel] (contiguous, rel_proj's dtype), cached on the module."""
    p = attn.__dict__.get("_sa_projT")
    if p is None:
        p = attn.__dict__["_sa_projT"] = mx.contiguous(attn.rel_proj.T)
    return p


def prepare(model):
    """Arrays to build before tracing (FastDecoder._prepare)."""
    return [proj_t(layer.self_attn) for layer in model.layers]


def sdpa_supported(attn, q, kb, r, sinks):
    if not _on_gpu() or q.ndim != 4 or q.shape[0] != 1:
        return False
    _, L, HQ, DH = q.shape
    HK = kb.shape[1]
    DR = r.shape[-1]
    return (
        2 <= L <= 8
        and q.dtype in _FLOATS and kb.dtype == q.dtype
        and DH % 32 == 0 and DH // 32 <= 8 and kb.shape[3] == DH
        and HQ % HK == 0
        and DR <= 32 and 32 % DR == 0
        and tuple(r.shape) == (1, L, HQ, DR)
        and tuple(sinks.shape) == (1, L, HQ)
        and "_sa_projT" in attn.__dict__
    )


def sdpa_rows(attn, q, kb, vb, r, slot_pos, qp, sinks, gate, ring, scale, cap):
    """q [1, L, HQ, DH] (pre-transpose rows), kb/vb [1, HK, S1, DH] buffers with this
    call's rows already written, r [1, L, HQ, DR], slot_pos [1, ring] (sliding) or None
    (global: slot = position), qp [1, L] = q0 + arange(L), sinks [1, L, HQ], gate
    [1, L, HQ*DH] or None. Returns [1, L, HQ*DH] = gated attention output."""
    _, L, HQ, DH = q.shape
    HK, S1 = kb.shape[1], kb.shape[2]
    DR = r.shape[-1]
    projT = proj_t(attn)
    R = projT.shape[0]
    window = int(attn.sliding)
    sliding = bool(attn.is_sliding)
    dt = q.dtype
    if r.dtype != dt:
        r = r.astype(dt)
    # static bound on the positions one head sees -> blocks per head
    nmax = min(window + L - 1, cap + 1) if window > 0 else min(S1, cap + 1)
    NB = max(1, -(-nmax // SDPA_BLOCK))
    KS = max(1, min(SDPA_KS, 32 // L))
    NT = 32 * L * KS
    TK = -(-max(SDPA_TK, 32 // DR) // (32 // DR)) * (32 // DR)  # multiple of the bias pass
    has_gate = gate is not None
    tmpl = [("T", dt), ("L", L), ("HQ", HQ), ("HK", HK), ("DH", DH), ("DR", DR),
            ("REXT", R), ("WINDOW", window), ("SLIDING", sliding), ("RING", int(ring)),
            ("S1", S1), ("NB", NB), ("KS", KS), ("TK", TK), ("GATE", has_gate)]
    sp = slot_pos.astype(mx.int32) if sliding else qp
    qpos = qp.reshape(-1).astype(mx.int32)
    g = gate if has_gate else q
    sc = _scalar(scale)
    ins = [q, kb, vb, r, projT, sp, qpos, sinks, g, sc]
    names = ["q", "kb", "vb", "rel", "projT", "slot_pos", "qpos", "sinks", "gate", "scale"]
    if NB == 1:
        k = _kernel("inkling2_sa_sdpa1v2", _SDPA_SRC.replace("/*EPI*/", _EPI1), names, ["out"])
        return k(inputs=ins, template=tmpl, grid=(NT, HQ, 1), threadgroup=(NT, 1, 1),
                 output_shapes=[(1, L, HQ * DH)], output_dtypes=[dt])[0]
    k = _kernel("inkling2_sa_sdpaNv2", _SDPA_SRC.replace("/*EPI*/", _EPIN), names, ["po", "pm", "ps"])
    po, pm, ps = k(inputs=ins, template=tmpl, grid=(NT, HQ, NB), threadgroup=(NT, 1, 1),
                   output_shapes=[(L * HQ, NB, DH), (L * HQ, NB), (L * HQ, NB)],
                   output_dtypes=[mx.float32] * 3)
    kc = _kernel("inkling2_sa_combine", _COMBINE_SRC, ["po", "pm", "ps", "sinks", "gate"], ["out"])
    return kc(inputs=[po, pm, ps, sinks, g],
              template=[("T", dt), ("HQ", HQ), ("DH", DH), ("NB", NB), ("GATE", has_gate)],
              grid=(DH, L * HQ, 1), threadgroup=(DH, 1, 1),
              output_shapes=[(1, L, HQ * DH)], output_dtypes=[dt])[0]


# ==========================================================================
# 2. tail rows: sconv tail + residual + depth attention (+ norm), one row per threadgroup
# ==========================================================================
# Grid (TG, L), TG = D/4: thread t owns channels 4t..4t+3 of row l.
# MODE 0 (post): s = T((sum_j ctx[l+j]*w[j] + attn_l) / div); residual = s (+ x_l unless
#   boundary) [f32]; boundary key = rms_norm(f32(x_l)); depth attention over
#   [k_0..k_{NK-1}, (bkey), rms_norm(residual)] with the mlp_input query.
# MODE 1 (pre): s from the MoE output; x_next = T(residual_l + s); depth attention over
#   [k_0.., rms_norm(f32(x_next))] with the next visit's attention_input query.
# Both: ctx = concat([state, f32(xin)]) (the rollback context), hidden [f32] and/or
# hn = T(nw * rms_norm(hidden)).
def _tail_source(nk):
    loads = "\n".join(
        f"        if (r == {i}) return float(k{i}[ro + c]);" for i in range(nk))
    return r"""
    constexpr int NSG = TG / 32;
    constexpr int WS = W - 1;
    constexpr int N = NK + (BKEY ? 1 : 0) + 1;
    threadgroup float redA[2 * NSG];
    threadgroup float redC[NSG];
    threadgroup float part[NSG * N * NQ];
    threadgroup float sc[N * NQ];
    const int lid = (int)thread_position_in_threadgroup.x;
    const int sg = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int l = (int)threadgroup_position_in_grid.y;
    const size_t ro = (size_t)l * D;
    const int n0 = lid * 4;
    float pv[4], bx[4];
    float acc_p = 0.0f, acc_b = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int c = n0 + i;
        const float hv = float(xin[ro + c]);
        float p;
        if (MODE == 2) {
            p = hv;
            pv[i] = p;
            acc_p += p * p;
            continue;
        }
        float value = 0.0f;
        for (int j = 0; j < W; ++j) {
            const int t = l + j;
            const float xv = t < WS ? state[(size_t)t * D + c] : float(xin[(size_t)(t - WS) * D + c]);
            value = value + xv * float(weight[(size_t)c * W + j]);
        }
        const T s = static_cast<T>((value + hv) / divisor[0]);
        ctx[(size_t)(WS + l) * D + c] = hv;
        if (l == 0)
            for (int j = 0; j < WS; ++j) ctx[(size_t)j * D + c] = state[(size_t)j * D + c];
        if (MODE == 0) {
            p = BKEY ? float(s) : float(s) + float(skip[ro + c]);
            res[ro + c] = static_cast<RT>(p);
        } else {
            const RT xn = static_cast<RT>(float(skip[ro + c]) + float(s));
            res[ro + c] = xn;
            p = float(xn);
        }
        pv[i] = p;
        acc_p += p * p;
        if (BKEY) { bx[i] = float(skip[ro + c]); acc_b += bx[i] * bx[i]; }
    }
    // rms of the partial (and of the boundary block input), one reduction
    acc_p = simd_sum(acc_p);
    acc_b = simd_sum(acc_b);
    if (lane == 0) { redA[sg] = acc_p; redA[NSG + sg] = acc_b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        float a = 0.0f, b = 0.0f;
        for (int j = 0; j < NSG; ++j) { a += redA[j]; b += redA[NSG + j]; }
        acc_p = a; acc_b = b;
    }
    const float inv_p = metal::precise::rsqrt(acc_p / D + 1e-6f);
    const float inv_b = metal::precise::rsqrt(acc_b / D + 1e-6f);
    float pk[4], bk[4];
    for (int i = 0; i < 4; ++i) {
        pk[i] = pv[i] * inv_p;
        if (BKEY) {
            bk[i] = bx[i] * inv_b;
            bkey[ro + n0 + i] = bk[i];
        }
    }
    auto key = [&](int r, int i) -> float {
        const int c = n0 + i;
""" + loads + r"""
        if (BKEY && r == NK) return bk[i];
        return pk[i];
    };
    // scores (NQ queries)
    for (int qi = 0; qi < NQ; ++qi) {
        float qv[4];
        for (int i = 0; i < 4; ++i) qv[i] = float(query[(size_t)qi * D + n0 + i]);
        for (int r = 0; r < N; ++r) {
            float a = 0.0f;
            for (int i = 0; i < 4; ++i) a += key(r, i) * qv[i];
            a = simd_sum(a);
            if (lane == 0) part[(sg * NQ + qi) * N + r] = a;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < N * NQ) {
        float a = 0.0f;
        for (int j = 0; j < NSG; ++j) a += part[j * N * NQ + lid];
        sc[lid] = a * scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float hx[4];
    float acc_h = 0.0f;
    for (int qi = 0; qi < NQ; ++qi) {
        float w[N];
        float mx_ = -INFINITY;
        for (int r = 0; r < N; ++r) { w[r] = sc[qi * N + r]; mx_ = metal::max(mx_, w[r]); }
        float z = 0.0f;
        for (int r = 0; r < N; ++r) { w[r] = metal::precise::exp(w[r] - mx_); z = w[r] + z; }
        for (int r = 0; r < N; ++r) w[r] = w[r] / z;
        for (int i = 0; i < 4; ++i) {
            float a = 0.0f;
            for (int r = 0; r < N; ++r) a += w[r] * key(r, i);
            hx[i] = a;
            acc_h += a * a;
            if (HID) hid[((size_t)l * NQ + qi) * D + n0 + i] = a;
        }
    }
    if (NORM) {
        acc_h = simd_sum(acc_h);
        if (lane == 0) redC[sg] = acc_h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float a = 0.0f;
        for (int j = 0; j < NSG; ++j) a += redC[j];
        const float inv = metal::precise::rsqrt(a / D + eps[0]);
        for (int i = 0; i < 4; ++i)
            hn[ro + n0 + i] = static_cast<T>(float(nw[n0 + i]) * (hx[i] * inv));
    }
"""


def norm_ok(norm):
    w = getattr(norm, "weight", None)
    return w is not None and w.ndim == 1 and type(norm).__name__ == "RMSNorm"


def tail_supported(module, state, xin, n_keys, query):
    """n_keys: depth-attention keys besides the partial (incl. a boundary key)."""
    if not _on_gpu() or xin.ndim != 3 or xin.shape[0] != 1:
        return False
    L, D = xin.shape[1], xin.shape[2]
    if module is None:  # mode 2 (depth attention only)
        return (2 <= L <= 16 and xin.dtype in _FLOATS and D % 128 == 0 and D // 4 <= 1024
                and query.shape[-1] == D and (query.ndim == 1 or query.shape[0] <= 4)
                and n_keys + 14 <= 31)
    W = getattr(module, "kernel_size", 0)
    w = getattr(getattr(module, "conv", None), "weight", None)
    return (
        2 <= L <= 16
        and xin.dtype in _FLOATS
        and D % 128 == 0 and D // 4 <= 1024
        and W >= 2 and w is not None and w.size == D * W
        and tuple(state.shape) == (1, W - 1, D) and state.dtype == mx.float32
        and query.shape[-1] == D and (query.ndim == 1 or (query.ndim == 2 and query.shape[0] <= 4))
        # 31 Metal buffer slots: keys + 9 inputs + 5 outputs
        and n_keys + 14 <= 31
    )


def tail_rows(mode, module, state, xin, skip, keys, query, boundary=False, norm=None,
              want_hid=True, out_dtype=None):
    """mode 2: depth attention only (xin = partial [1,L,D]; module/state/skip unused):
    returns dict(hid [1,L,NQ,D] f32, hn if norm). query [D] or [NQ, D] (NQ > 1: no norm)."""
    """mode 0 (post): xin = attention output, skip = layer input x; returns dict(ctx, res
    [f32 residual], hid [1,L,1,D] f32, bkey [1,L,1,D] f32 if boundary, hn [1,L,D] if norm).
    mode 1 (pre): xin = MoE output, skip = residual f32; res = x_next [out_dtype]."""
    _, L, D = xin.shape
    W = module.kernel_size if mode != 2 else 4
    nk = len(keys)
    NQ = 1 if query.ndim == 1 else query.shape[0]
    if NQ > 1:
        assert norm is None and want_hid
    if mode == 2:
        state = skip = xin
        weight = query
    else:
        weight = module.conv.weight
    T = xin.dtype if mode == 0 else (out_dtype or xin.dtype)
    RT = mx.float32 if mode == 0 else T
    has_norm = norm is not None
    bkey = bool(boundary) and mode == 0
    k = _kernel(f"inkling2_sa_tail2_{nk}", _tail_source(nk),
                [f"k{i}" for i in range(nk)]
                + ["state", "weight", "xin", "skip", "query", "scale", "divisor", "nw", "eps"],
                ["ctx", "res", "hid", "bkey", "hn"], contiguous=True)
    from . import fast_kernels as fk
    outs = k(
        inputs=[*keys, state, weight, xin, skip, query, _scalar(D ** -0.5),
                _scalar(math.sqrt(W + 1)), norm.weight if has_norm else query,
                _scalar(norm.eps if has_norm else 1e-6)],
        template=[("T", T), ("RT", RT), ("D", D), ("TG", D // 4), ("NK", nk), ("W", W),
                  ("MODE", int(mode)), ("BKEY", bkey), ("NORM", has_norm),
                  ("HID", bool(want_hid)), ("NQ", NQ)],
        grid=(D // 4, L, 1), threadgroup=(D // 4, 1, 1),
        output_shapes=[(1, W - 1 + L, D) if mode != 2 else (1,), (1, L, D) if mode != 2 else (1,),
                       (1, L, NQ, D) if want_hid else (1,),
                       (1, L, 1, D) if bkey else (1,), (1, L, D) if has_norm else (1,)],
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


# ==========================================================================
# 3. kvsc: sliding-layer k/v sconv folded into the qkv clip kernel (fast_kernels.qkv_clip)
# ==========================================================================
# The k and v rows of fk._QKV_SRC read the raw projection and first apply the stock
# ShortConvolution (state + this call's rows, serial lags, (sum + x) / sqrt(W + 1) -> T),
# then clip / value-norm exactly as qkv_clip. The rollback contexts concat([state, f32(x)])
# are written as two extra outputs (row l's threadgroups write row W-1+l; l == 0 also the
# state rows). One launch instead of sconv k + sconv v + qkv_clip (+ 2 concats).
_KV_SCONV = r"""{
                const int C_ = HK * HD;
                const float hv_ = float(@X@[l * @X@_strides[1] + c * @X@_strides[2]]);
                float val_ = 0.0f;
                for (int j_ = 0; j_ < W; ++j_) {
                    const int t_ = l + j_;
                    const float xx_ = t_ < W - 1 ? @X@st[t_ * C_ + c]
                        : float(@X@[(t_ - (W - 1)) * @X@_strides[1] + c * @X@_strides[2]]);
                    val_ = val_ + xx_ * float(@X@w[c * W + j_]);
                }
                @DST@ = float(static_cast<T>((val_ + hv_) / divisor[0]));
                @X@ctx[(W - 1 + l) * C_ + c] = hv_;
                if (l == 0)
                    for (int j_ = 0; j_ < W - 1; ++j_) @X@ctx[j_ * C_ + c] = @X@st[j_ * C_ + c];
            }"""


def _kvsc_source():
    from . import fast_kernels as fk
    src = fk._QKV_SRC
    ko = "                xv = float(k[b * k_strides[0] + l * k_strides[1] + c * k_strides[2]]);"
    vo = "            xv[i] = float(v[b * v_strides[0] + l * v_strides[1] + c * v_strides[2]]);"
    assert src.count(ko) == 1 and src.count(vo) == 1, "qkv_clip source changed"
    src = src.replace(ko, "                " + _KV_SCONV.replace("@X@", "k").replace("@DST@", "xv"))
    src = src.replace(vo, "            " + _KV_SCONV.replace("@X@", "v").replace("@DST@", "xv[i]"))
    return src


def kvsc_supported(attn, cfg, x, ks, vs, kstate, vstate, B):
    from . import fast_kernels as fk
    C = attn.n_kv * attn.head_dim
    W = getattr(ks, "kernel_size", 0)
    return (
        B == 1 and attn.is_sliding and fk.qkv_clip_supported(attn, cfg, x)
        and 2 <= x.shape[1] <= 16
        and W >= 2 and getattr(vs, "kernel_size", 0) == W
        and ks.conv.weight.size == C * W and vs.conv.weight.size == C * W
        and tuple(kstate.shape) == (1, W - 1, C) and tuple(vstate.shape) == (1, W - 1, C)
        and kstate.dtype == mx.float32 and vstate.dtype == mx.float32
    )


def kvsc_clip(q, k, v, r, attn, cfg, ks, vs, kstate, vstate):
    """q, r, raw k, v projections [1, L, *] (views ok) -> (q, k, v, r) as fk.qkv_clip after
    the k/v ShortConvolutions, plus the k and v rollback contexts [1, W-1+L, C] f32."""
    from . import fast_kernels as fk
    k_ = _k.get("kvsc")
    if k_ is None:
        k_ = _k["kvsc"] = mx.fast.metal_kernel(
            name="inkling2_sa_kvsc",
            input_names=["q", "k", "v", "r", "q_scale", "rel_scale", "thr", "rthr", "eps",
                         "tau_in", "kst", "vst", "kw", "vw", "divisor"],
            output_names=["qo", "ko", "vo", "ro", "kctx", "vctx"],
            header=fk._QKV_HEADER, source=_kvsc_source(),
            ensure_row_contiguous=False, compile_options={"math_mode": "safe"})
    B, L = q.shape[0], q.shape[1]
    hq, hk, hd, dr = attn.n_heads, attn.n_kv, attn.head_dim, cfg.d_rel
    W = ks.kernel_size
    C = hk * hd
    dt = q.dtype
    outs = k_(
        inputs=[q, k, v, r, attn.q_scale, attn.rel_scale, _scalar(cfg.qk_l2_clip_threshold),
                _scalar(cfg.rel_l2_clip_threshold), _scalar(cfg.rms_norm_eps), _scalar(1.0),
                kstate, vstate, ks.conv.weight, vs.conv.weight, _scalar(math.sqrt(W + 1))],
        template=[("T", dt), ("HQ", hq), ("HK", hk), ("HD", hd), ("DR", dr),
                  ("HAS_TAU", False), ("VNORM", bool(cfg.use_value_norm_no_gamma)),
                  ("ZROW", False), ("W", W)],
        grid=(32, hq + 2 * hk + hq, B * L), threadgroup=(32, 1, 1),
        output_shapes=[(B, L, hq, hd), (B, L, hk, hd), (B, L, hk, hd), (B, L, hq, dr),
                       (1, W - 1 + L, C), (1, W - 1 + L, C)],
        output_dtypes=[dt] * 4 + [mx.float32] * 2,
    )
    return outs
