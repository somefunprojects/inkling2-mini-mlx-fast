# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""Coarse block kernels for the FastDecoder L=1 step (INKLING_FAST=mega_*).

Goal: fewer GPU dispatches per decode token (CPU
encode is ~2.2-2.9 us per launch at B=1). Each block reproduces the MLX
kernels it replaces in the same reduction order, so results are bitwise vs
the op chain it replaces (unit tests in tests_fast/test_mega.py).

Flags (each independent; all need B*L == 1, GPU, BF16 weights):
* ``mega_ctrl``  MessageController(s): rms_norm(x), rms_norm(message), concat,
  BF16 cast, down GEMV [r, 2D]  -> kernel 1 (ctrl_down)
  forget/gain GEMVs [D, r] (FP32), tanh, blend -> kernel 2 (ctrl_blend).
  Replaces ~15 launches (3 batched controllers) / ~12 (mlp controller).
"""

import mlx.core as mx

# --------------------------------------------------------------------------
# MessageController
# --------------------------------------------------------------------------
# Stock (B=1):
#   pair   = concat([rms_norm(x), rms_norm(m)], -1)                 f32 [C, 2D]
#   hidden = (pair.astype(bf16) @ dw.T).astype(f32)    gemv bm1 bn8 sm1 sn32 tm4 tn4
#                                                      (K=2D >= 16*r: split-K variant)
#   forget = tanh(hidden @ fw.T)   fw promoted to f32  gemv bm1 bn1 sm8 sn4 tm4 tn4 (K=r<=64)
#   gain   = tanh(hidden @ gw.T)
#   out    = (1 - forget) * x + gain * m
# rms_norm: MLX rms_single_row, D/4 threads x 4 reads, simd_sum, zero-padded
# threadgroup simd_sum, precise::rsqrt(acc / D + eps).

_CTRL_DOWN_SRC = r"""
    // grid: x = tile (4 output rows) * 256 threads, y = controller c.
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int tile = (int)threadgroup_position_in_grid.x;
    const int c = (int)threadgroup_position_in_grid.y;
    constexpr int K = 2 * D;
    constexpr int NVS = D / 128;          // virtual simdgroups of rms_single_row
    threadgroup float pv[K];
    threadgroup float red[2][32];
    threadgroup float tgp[8 * 8];
    const int xrow = c < NQ ? c : NQ - 1;
    const float epsv = eps[0];
    for (int w = 0; w < 2; ++w) {
        const device float* src = (w == 0) ? (x + (size_t)xrow * D) : (m + (size_t)(M0 + c) * D);
        if (lid < 32) red[w][lid] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int vs = (int)sg; vs < NVS; vs += 8) {
            const int n0 = (vs * 32 + (int)lane) * 4;
            float acc = 0;
            for (int i = 0; i < 4; i++) { float t = src[n0 + i]; acc += t * t; }
            acc = simd_sum(acc);
            if (lane == 0) red[w][vs] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = simd_sum(red[w][lane]);
        const float inv = metal::precise::rsqrt(tot / D + epsv);
        for (int i = (int)lid; i < D; i += 256) {
            float v = 1.0f * static_cast<float>(src[i] * inv);
            pv[w * D + i] = static_cast<float>(static_cast<bfloat16_t>(v));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // gemv bm1 bn8 sm1 sn32 tm4 tn4 over K; rows tile*4 .. +3 of dw[c].
    const int out_row = tile * 4;
    const device bfloat16_t* mat = dw + ((size_t)c * R + out_row) * K;
    float result[4] = {0, 0, 0, 0};
    int bn = ((int)sg * 32 + (int)lane) * 4;
    constexpr int NIT = K / 1024;
    for (int it = 0; it < NIT; ++it) {
        float vc[4];
        for (int t = 0; t < 4; t++) vc[t] = pv[bn + t];
        for (int tm = 0; tm < 4; tm++) {
            bfloat16_t inter[4];
            for (int t = 0; t < 4; t++) inter[t] = mat[tm * K + bn + t];
            for (int t = 0; t < 4; t++) result[tm] += inter[t] * vc[t];
        }
        bn += 1024;
    }
    if (K % 1024 > 0) {
        float vc[4];
        bool full = bn + 4 <= K;
        for (int t = 0; t < 4; t++) vc[t] = (full || bn + t < K) ? pv[bn + t] : 0.0f;
        for (int tm = 0; tm < 4; tm++) {
            bfloat16_t inter[4];
            for (int t = 0; t < 4; t++) inter[t] = (full || bn + t < K) ? mat[tm * K + bn + t] : bfloat16_t(0);
            for (int t = 0; t < 4; t++) result[tm] += inter[t] * vc[t];
        }
    }
    for (int tm = 0; tm < 4; tm++)
        for (ushort s = 16; s >= 1; s >>= 1) result[tm] += simd_shuffle_down(result[tm], s);
    if (lane == 0)
        for (int tm = 0; tm < 4; tm++) tgp[sg * 8 + tm] = result[tm];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (int sgn = 1; sgn < 8; sgn++)
            for (int tm = 0; tm < 4; tm++) result[tm] += tgp[sgn * 8 + tm];
        for (int tm = 0; tm < 4; tm++)
            hidden[(size_t)c * R + out_row + tm] = static_cast<float>(static_cast<bfloat16_t>(result[tm]));
    }
"""

_CTRL_BLEND_SRC = r"""
    // gemv bm1 bn1 sm8 sn4 tm4 tn4 (K = R <= 64): one simdgroup = 32 rows.
    // 8 simdgroups per threadgroup = 256 rows. grid y = controller c.
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int c = (int)threadgroup_position_in_grid.y;
    const int thrM = (int)lane / 4, thrN = (int)lane % 4;
    int out_row = ((int)threadgroup_position_in_grid.x * 8 + (int)sg) * 32 + thrM * 4;
    if (out_row >= D) return;
    out_row = out_row + 4 <= D ? out_row : D - 4;
    const device bfloat16_t* fm = fw + ((size_t)c * D + out_row) * R;
    const device bfloat16_t* gm = gw + ((size_t)c * D + out_row) * R;
    const device float* h = hidden + (size_t)c * R;
    float rf[4] = {0, 0, 0, 0}, rg[4] = {0, 0, 0, 0};
    int bn = thrN * 4;
    for (int it = 0; it < R / 16; ++it) {
        float vc[4];
        for (int t = 0; t < 4; t++) vc[t] = h[bn + t];
        for (int tm = 0; tm < 4; tm++) {
            float fi[4], gi[4];
            for (int t = 0; t < 4; t++) { fi[t] = static_cast<float>(fm[tm * R + bn + t]); gi[t] = static_cast<float>(gm[tm * R + bn + t]); }
            for (int t = 0; t < 4; t++) { rf[tm] += fi[t] * vc[t]; rg[tm] += gi[t] * vc[t]; }
        }
        bn += 16;
    }
    for (int tm = 0; tm < 4; tm++) {
        for (ushort s = 2; s >= 1; s >>= 1) {
            rf[tm] += simd_shuffle_down(rf[tm], s);
            rg[tm] += simd_shuffle_down(rg[tm], s);
        }
    }
    if (thrN == 0) {
        const int xrow = c < NQ ? c : NQ - 1;
        for (int tm = 0; tm < 4; tm++) {
            const int d = out_row + tm;
            const float f = metal::precise::tanh(rf[tm]);
            const float g = metal::precise::tanh(rg[tm]);
            const float xv = x[(size_t)xrow * D + d];
            const float mv = m[(size_t)(M0 + c) * D + d];
            const float a = (1.0f - f) * xv;
            const float b = g * mv;
            out[(size_t)c * D + d] = a + b;
        }
    }
"""

_kern = {}


def _dp_suffix():
    from . import fast_kernels as fk
    return "_dp" if fk.enabled("mega_dpar") else ""




def _get(name, src, inputs, outputs):
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name=f"inkling2_mega_{name}",
            input_names=inputs,
            output_names=outputs,
            source=src,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    return k


def ctrl_supported(x, dw, fw=None, gw=None, n_ctrl=None):
    """dw [n, R, 2D], fw/gw [n, D, R] (or one controller's 2-D weights,
    n = 1), all BF16: the kernels cast them to device bfloat16_t* and index
    them as dense [n, D, R]."""
    D = x.shape[-1]
    R = dw.shape[-2]
    n = dw.shape[0] if dw.ndim == 3 else 1
    shp = lambda w: tuple(w.shape) if w.ndim == 3 else (1, *w.shape)
    return (
        x.size // D == 1
        and dw.dtype == mx.bfloat16
        and dw.ndim in (2, 3)
        and (n_ctrl is None or n >= n_ctrl)
        and fw is not None and gw is not None
        and fw.dtype == mx.bfloat16 and gw.dtype == mx.bfloat16
        and fw.ndim == dw.ndim and gw.ndim == dw.ndim
        and shp(fw) == (n, D, R) and shp(gw) == (n, D, R)
        and D % 128 == 0
        and D // 128 <= 32
        and dw.shape[-1] == 2 * D
        and R % 16 == 0
        and R <= 64
        and R % 4 == 0
        and (2 * D) * 4 <= 24 * 1024
    )


def controllers(x, m, m0, dw, fw, gw, eps, n_ctrl, nq):
    """n_ctrl MessageControllers at B*L == 1.

    x: [nq, D] f32 (controller c reads row min(c, nq - 1));
    m: [Mrows, D] f32 (controller c reads row m0 + c);
    dw [n_ctrl, R, 2D], fw/gw [n_ctrl, D, R] (BF16). Returns [n_ctrl, D] f32."""
    D = x.shape[-1]
    R = dw.shape[-2]
    tmpl = [("D", D), ("R", R), ("NQ", nq), ("M0", m0)]
    eps_a = mx.array([eps], mx.float32)
    hidden = _get("ctrl_down", _CTRL_DOWN_SRC, ["x", "m", "dw", "eps"], ["hidden"])(
        inputs=[x, m, dw, eps_a],
        template=tmpl,
        grid=(256 * (R // 4), n_ctrl, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_ctrl, R)],
        output_dtypes=[mx.float32],
    )[0]
    ntg = (D + 255) // 256
    return _get("ctrl_blend", _CTRL_BLEND_SRC, ["x", "m", "hidden", "fw", "gw"], ["out"])(
        inputs=[x, m, hidden, fw, gw],
        template=tmpl,
        grid=(256 * ntg, n_ctrl, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_ctrl, D)],
        output_dtypes=[mx.float32],
    )[0]


# --------------------------------------------------------------------------
# RMSNorm(weight) + cast, n rows in one launch  (INKLING_FAST=mega_norm)
# --------------------------------------------------------------------------
# Stock: nn.RMSNorm(x f32) with a BF16 weight = weight.astype(f32) +
# RMSNorm (rms_single_row, D/4 threads) + .astype(bf16): 3 launches per norm.

def _norm_source(n):
    sel_x = "\n".join(
        f"    if (r == {i}) {{ xp = x{i}; xs0 = x{i}_strides[0]; xs1 = x{i}_strides[1]; xs2 = x{i}_strides[2]; wp = w{i}; ws = w{i}_strides[0]; }}"
        for i in range(n)
    )
    return r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int r = (int)threadgroup_position_in_grid.y;
    const device float* xp = x0; const device bfloat16_t* wp = w0;
    int64_t xs0 = 0, xs1 = 0, xs2 = 0, ws = 1;
""" + sel_x + r"""
    (void)xs0; (void)xs1;
    threadgroup float red[32];
    float tx[4];
    float acc = 0;
    const int n0 = (int)lid * 4;
    for (int i = 0; i < 4; i++) {
        tx[i] = xp[(size_t)(n0 + i) * xs2];
        acc += tx[i] * tx[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) red[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(red[lane]);
        if (lane == 0) red[0] = metal::precise::rsqrt(acc / D + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red[0];
    for (int i = 0; i < 4; i++) {
        const float wv = static_cast<float>(wp[(size_t)(n0 + i) * ws]);
        out[(size_t)r * D + n0 + i] = static_cast<bfloat16_t>(wv * static_cast<float>(tx[i] * inv));
    }
"""


def norm_supported(x, weight):
    D = x.shape[-1]
    return (
        x.dtype == mx.float32
        and weight.dtype == mx.bfloat16
        and weight.ndim == 1
        and weight.shape[0] == D
        and x.size == D
        and D % 128 == 0
        and 32 <= D // 4 <= 1024
    )


def rms_norm_cast(xs, ws, eps):
    """[RMSNorm(w_i)(x_i).astype(bf16) for i] in one launch. x_i: [..., D] f32
    with one row (any strides); w_i: [D] bf16 (any stride).
    Returns [n, D] bf16."""
    n = len(xs)
    D = xs[0].shape[-1]
    name = f"norm{n}"
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name=f"inkling2_mega_{name}",
            input_names=[f"x{i}" for i in range(n)] + [f"w{i}" for i in range(n)] + ["eps"],
            output_names=["out"],
            source=_norm_source(n),
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    xs = [x.reshape(1, 1, D) for x in xs]
    return k(
        inputs=[*xs, *ws, mx.array([eps], mx.float32)],
        template=[("D", D)],
        grid=(D // 4, n, 1),
        threadgroup=(D // 4, 1, 1),
        output_shapes=[(n, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


# --------------------------------------------------------------------------
# DynamicShortConvolution (global k/v sconv), both in one launch
# (INKLING_FAST=mega_dsconv)
# --------------------------------------------------------------------------
# Stock (L=1, mask None), per k and v:
#   dyn  = weight_proj(dyn_in)  BF16 GEMV [H*4, D], K >= 16*N -> gemv bm1 bn8
#          sm1 sn32 tm4 tn4 (split-K), out BF16 -> f32
#   w[h, j, lag] = static_weight[lag, h*CH + j] + dyn[h*4 + lag]
#   ctx = [state0, state1, state2, x];  out = 0; out = out + ctx[3-lag] * w[lag]
#   y = ((out + x) / 3.0).astype(bf16);  next state = [state1, state2, x]
# One threadgroup per (head, k|v): its 4 dynamic taps are exactly one gemv
# tile (rows h*4 .. h*4+3), reduced in MLX's order.

_DSCONV_SRC = r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int h = (int)threadgroup_position_in_grid.x;
    const int which = (int)threadgroup_position_in_grid.y;
    const device bfloat16_t* xin = which ? xv : xk;
    const device bfloat16_t* din = which ? dv : dk;
    const device bfloat16_t* wp = which ? wpv : wpk;
    const device bfloat16_t* sw = which ? swv : swk;
    const device float* st = which ? stv : stk;
    device bfloat16_t* o = which ? outv : outk;
    device float* ns = which ? nsv : nsk;
    threadgroup float tgp[8 * 8];
    threadgroup float dyn[4];
    // 1. dynamic taps: rows h*4 .. h*4+3 of wp @ din (gemv bm1 bn8 sm1 sn32 tm4 tn4)
    {
        const device bfloat16_t* mat = wp + (size_t)(h * 4) * D;
        float result[4] = {0, 0, 0, 0};
        int bn = ((int)sg * 32 + (int)lane) * 4;
        for (int it = 0; it < D / 1024; ++it) {
            float vc[4];
            for (int t = 0; t < 4; t++) vc[t] = static_cast<float>(din[bn + t]);
            for (int tm = 0; tm < 4; tm++) {
                bfloat16_t inter[4];
                for (int t = 0; t < 4; t++) inter[t] = mat[tm * D + bn + t];
                for (int t = 0; t < 4; t++) result[tm] += inter[t] * vc[t];
            }
            bn += 1024;
        }
        if (D % 1024 > 0) {
            float vc[4];
            bfloat16_t inter[4][4];
            bool full = bn + 4 <= D;
            for (int t = 0; t < 4; t++) vc[t] = (full || bn + t < D) ? static_cast<float>(din[bn + t]) : 0.0f;
            for (int tm = 0; tm < 4; tm++)
                for (int t = 0; t < 4; t++) inter[tm][t] = (full || bn + t < D) ? mat[tm * D + bn + t] : bfloat16_t(0);
            for (int tm = 0; tm < 4; tm++)
                for (int t = 0; t < 4; t++) result[tm] += inter[tm][t] * vc[t];
        }
        for (int tm = 0; tm < 4; tm++)
            for (ushort s = 16; s >= 1; s >>= 1) result[tm] += simd_shuffle_down(result[tm], s);
        if (lane == 0)
            for (int tm = 0; tm < 4; tm++) tgp[sg * 8 + tm] = result[tm];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            for (int sgn = 1; sgn < 8; sgn++)
                for (int tm = 0; tm < 4; tm++) result[tm] += tgp[sgn * 8 + tm];
            for (int tm = 0; tm < 4; tm++)
                dyn[tm] = static_cast<float>(static_cast<bfloat16_t>(result[tm]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 2. convolution for this head's CH channels
    for (int j = (int)lid; j < CH; j += 256) {
        const int c = h * CH + j;
        const float xf = static_cast<float>(xin[c]);
        const float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
        const float v[4] = {xf, s2, s1, s0};  // ctx[3 - lag]
        float acc = 0.0f;
        for (int lag = 0; lag < 4; ++lag) {
            const float w = static_cast<float>(sw[lag * C + c]) + dyn[lag];
            const float p = v[lag] * w;
            acc = acc + p;
        }
        const float r = (acc + xf) / div[0];
        o[c] = static_cast<bfloat16_t>(r);
        ns[c] = s1;
        ns[C + c] = s2;
        ns[2 * C + c] = xf;
    }
"""


def dsconv_supported(mod, x, dyn_in, state):
    w = mod.weight_proj.weight
    D = dyn_in.shape[-1]
    C = x.shape[-1]
    return (
        x.size == C
        and dyn_in.size == D
        and mod.kernel_size == 4
        and x.dtype == mx.bfloat16
        and dyn_in.dtype == mx.bfloat16
        and w.dtype == mx.bfloat16
        and mod.static_weight.dtype == mx.bfloat16
        and state is not None
        and state.dtype == mx.float32
        and state.shape[1] == 3
        and w.shape[0] == mod.num_heads * 4
        and w.ndim == 2 and w.shape[1] == D
        and tuple(mod.static_weight.shape) == (4, C)
        and C % mod.num_heads == 0
        and D % 4 == 0
        # the kernel hard-codes MLX's split-K gemv tile (bm1 bn8): stock picks
        # it only for K > 64 and K >= 16 * out rows
        and D > 64
        and D >= 16 * w.shape[0]
    )


def dsconv_pair(kmod, vmod, xk, xv, dk, dv, stk, stv):
    """k and v DynamicShortConvolution (L=1). Returns (yk, yv, nstate_k, nstate_v)."""
    C = xk.shape[-1]
    H = kmod.num_heads
    D = dk.shape[-1]
    k = _kern.get("dsconv")
    if k is None:
        k = _kern["dsconv"] = mx.fast.metal_kernel(
            name="inkling2_mega_dsconv",
            input_names=["xk", "xv", "dk", "dv", "wpk", "wpv", "swk", "swv", "stk", "stv", "div"],
            output_names=["outk", "outv", "nsk", "nsv"],
            source=_DSCONV_SRC,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    yk, yv, nk, nv = k(
        inputs=[xk, xv, dk, dv, kmod.weight_proj.weight, vmod.weight_proj.weight,
                kmod.static_weight, vmod.static_weight, stk, stv,
                mx.array([3.0], mx.float32)],
        template=[("D", D), ("C", C), ("CH", C // H)],
        grid=(256 * H, 2, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[xk.shape, xv.shape, stk.shape, stv.shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )
    return yk, yv, nk, nv


# --------------------------------------------------------------------------
# MoE glue at B*L == 1 (INKLING_FAST=mega_moe)
# --------------------------------------------------------------------------
# * router logits: x.astype(f32) @ W_f32.T is AsType + gemv; one kernel reads
#   the BF16 x directly (gemv bm1 bn8 sm1 sn32 tm4 tn4, K >= 16 * E).
# * gather_qmm without lhs_indices builds an arange (one launch each for
#   gate/up and down): pass cached constant lhs indices instead.

_ROUTER_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const int out_row = (int)threadgroup_position_in_grid.x * 4;
    threadgroup float tgp[8 * 8];
    const device float* mat = w + (size_t)out_row * K;
    float result[4] = {0, 0, 0, 0};
    int bn = ((int)sg * 32 + (int)lane) * 4;
    for (int it = 0; it < K / 1024; ++it) {
        float vc[4];
        for (int t = 0; t < 4; t++) vc[t] = static_cast<float>(x[bn + t]);
        for (int tm = 0; tm < 4; tm++) {
            float inter[4];
            for (int t = 0; t < 4; t++) inter[t] = mat[tm * K + bn + t];
            for (int t = 0; t < 4; t++) result[tm] += inter[t] * vc[t];
        }
        bn += 1024;
    }
    if (K % 1024 > 0) {
        float vc[4];
        float inter[4][4];
        bool full = bn + 4 <= K;
        for (int t = 0; t < 4; t++) vc[t] = (full || bn + t < K) ? static_cast<float>(x[bn + t]) : 0.0f;
        for (int tm = 0; tm < 4; tm++)
            for (int t = 0; t < 4; t++) inter[tm][t] = (full || bn + t < K) ? mat[tm * K + bn + t] : 0.0f;
        for (int tm = 0; tm < 4; tm++)
            for (int t = 0; t < 4; t++) result[tm] += inter[tm][t] * vc[t];
    }
    for (int tm = 0; tm < 4; tm++)
        for (ushort s = 16; s >= 1; s >>= 1) result[tm] += simd_shuffle_down(result[tm], s);
    if (lane == 0)
        for (int tm = 0; tm < 4; tm++) tgp[sg * 8 + tm] = result[tm];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        for (int sgn = 1; sgn < 8; sgn++)
            for (int tm = 0; tm < 4; tm++) result[tm] += tgp[sgn * 8 + tm];
        for (int tm = 0; tm < 4; tm++) logits[out_row + tm] = result[tm];
    }
"""


def router_logits(x, w_f32):
    """x [..., K] bf16 (one row), w [E, K] f32 -> [1, E] f32 == x.astype(f32) @ w.T."""
    E, K = w_f32.shape
    k = _kern.get("router")
    if k is None:
        k = _kern["router"] = mx.fast.metal_kernel(
            name="inkling2_mega_router",
            input_names=["x", "w"],
            output_names=["logits"],
            source=_ROUTER_SRC,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    return k(
        inputs=[x.reshape(K), w_f32],
        template=[("K", K)],
        grid=(256 * (E // 4), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, E)],
        output_dtypes=[mx.float32],
    )[0]


def router_supported(x, w_f32):
    E, K = w_f32.shape
    return (x.size == K and x.dtype == mx.bfloat16 and w_f32.dtype == mx.float32
            and E % 4 == 0 and K >= 16 * E and K % 4 == 0)


_consts = {}


def lhs_const(n):
    """Cached [1, 1, n] uint32 arange (what gather_qmm builds when lhs_indices
    is omitted for a [1, 1, n, ...] input). Built outside any trace."""
    a = _consts.get(n)
    if a is None:
        a = _consts[n] = mx.arange(n, dtype=mx.uint32).reshape(1, 1, n)
        mx.eval(a)
    return a


def _down_factor(lin):
    """The global-scale factor ScaledQuantizedSwitchLinear would pass, or
    None when its GPU path is not the plain gather_qmm one."""
    if not hasattr(lin, "global_scale") or "pre_quant_scale" in lin or "bias" in lin:
        return None
    if lin.global_scale.size not in (1, lin.num_experts):
        return None
    cached = lin.__dict__.get("_fast_factor")
    if cached is None or cached[0] is not lin.global_scale:
        return None
    return cached[1]


def moe_routed(moe, x, fused, route_fn):
    """SparseMoE._routed for x [1, 1, D] (fusemoe + glue decode path) with
    constant lhs indices and the one-kernel router GEMV. Returns
    (out [1, 1, k, D], weights [1, 1, k]) or None if unsupported."""
    sm = moe.switch_mlp
    down = sm.down_proj
    factor = _down_factor(down)
    w32 = moe._fast_router_weight()
    if factor is None or fused is None or not router_supported(x, w32):
        return None
    logits = router_logits(x, w32).reshape(1, 1, -1)
    indices, weights = route_fn(logits, moe.e_score_correction_bias, moe.top_k, moe.route_scale)
    k = indices.shape[-1]
    xe = mx.expand_dims(x, (-2, -3))
    kw = {} if fused["factor"] is None else {"global_scale": fused["factor"]}
    gu = mx.gather_qmm(
        xe, fused["weight"], fused["scales"],
        lhs_indices=lhs_const(1), rhs_indices=indices, transpose=True,
        group_size=fused["group_size"], bits=fused["bits"], mode=fused["mode"],
        sorted_indices=False, **kw,
    )
    n = fused["n"]
    act = sm.activation(gu[..., n:], gu[..., :n])
    out = mx.gather_qmm(
        act, down.weight, down.scales,
        lhs_indices=lhs_const(k), rhs_indices=indices, transpose=True,
        group_size=down.group_size, bits=down.bits, mode=down.mode,
        global_scale=factor, sorted_indices=False,
    )
    return out.squeeze(-2), weights


# --------------------------------------------------------------------------
# Post-attention block (INKLING_FAST=mega_post), one threadgroup per row:
#   attn_sconv(attn out) [+ x]  -> residual (sconv_tail MODE 0 formula)
#   [block boundary: key = rms_norm(x.f32)]
#   mlp_input depth attention over keys + rms_norm(residual)  (attnres order)
#   [post_attention_layernorm(hidden).astype(bf16)]
# Replaces sconv_tail + (boundary RMSNorm) + attnres + (norm) launches.
# TG = D/4 threads: thread t owns channels 4t..4t+3, which is exactly MLX's
# rms_single_row layout, so every rms_norm reduces in the stock order.
# --------------------------------------------------------------------------

def _post_source(nki):
    from . import fast_kernels as fk
    src = _post_source_raw(nki)
    return fk._dpar(src) if fk.enabled("mega_dpar") else src


def _post_source_raw(nki):
    loads = "\n".join(
        f"        if (r == {i}) return k{i}[(size_t)d];" for i in range(nki)
    )
    return r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int N = NKI + (BKEY ? 1 : 0) + 1;
    threadgroup float pk[D];
    threadgroup float bk[BKEY ? D : 1];
    threadgroup float ov[D];
    threadgroup float red[32];
    threadgroup float sc[N];
    threadgroup float wts[N];
    threadgroup float part[8];
    const int n0 = (int)lid * 4;
    // rms_single_row reduction of acc (per-thread partial) -> inverse rms
    auto rms_inv = [&](float acc, float epsv) -> float {
        acc = simd_sum(acc);
        if (sg == 0) red[lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            acc = simd_sum(red[lane]);
            if (lane == 0) red[0] = metal::precise::rsqrt(acc / D + epsv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return inv;
    };
    // 1. attn sconv tail (sconv_tail MODE 0) -> residual (partial), next state
    float tx[4];
    {
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            const int c = n0 + i;
            const float hv = float(xin[c]);
            float value = 0;
            for (int j = 0; j < W; ++j) {
                float xv = j < W - 1 ? state[j * D + c] : hv;
                float w = weight[c * W + j];
                value = value + xv * w;
            }
            const bfloat16_t s = bfloat16_t((value + hv) / divisor);
            for (int j = 0; j < W - 1; ++j)
                nstate[j * D + c] = j < W - 2 ? state[(j + 1) * D + c] : hv;
            const float r = BKEY ? float(s) : float(s) + float(skip[c]);
            residual[c] = r;
            tx[i] = r;
            acc += tx[i] * tx[i];
        }
        // 2. rms_norm(partial) -> pk (attnres step 1; partial is f32 already)
        const float inv = rms_inv(acc, 1e-6f);
        for (int i = 0; i < 4; i++) pk[n0 + i] = 1.0f * static_cast<float>(tx[i] * inv);
    }
    // 3. boundary key = rms_norm(x.astype(f32))
    if (BKEY) {
        float bx[4];
        float acc = 0;
        for (int i = 0; i < 4; i++) { bx[i] = float(skip[n0 + i]); acc += bx[i] * bx[i]; }
        const float inv = rms_inv(acc, 1e-6f);
        for (int i = 0; i < 4; i++) {
            const float kv = 1.0f * static_cast<float>(bx[i] * inv);
            bk[n0 + i] = kv;
            bkey[n0 + i] = kv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto key = [&](int r, int d) -> float {
""" + loads + r"""
        if (BKEY && r == NKI) return bk[d];
        return pk[d];
    };
    // 4. scores: gemv (bn8, sn32, tn4) per key row (as fast_kernels.attn_res)
    for (int r = 0; r < N; ++r) {
        if (sg < 8) {
            float result = 0;
            int bn = ((int)sg * 32 + (int)lane) * 4;
            for (int i = 0; i < D / 1024; ++i) {
                float vc[4], it[4];
                for (int t = 0; t < 4; t++) vc[t] = float(query[bn + t]);
                for (int t = 0; t < 4; t++) it[t] = key(r, bn + t);
                for (int t = 0; t < 4; t++) result += it[t] * vc[t];
                bn += 1024;
            }
            if (D % 1024 > 0) {
                float vc[4], it[4];
                if (bn + 4 <= D) {
                    for (int t = 0; t < 4; t++) vc[t] = float(query[bn + t]);
                    for (int t = 0; t < 4; t++) it[t] = key(r, bn + t);
                } else {
                    for (int t = 0; t < 4; t++) {
                        vc[t] = bn + t < D ? float(query[bn + t]) : 0.0f;
                        it[t] = bn + t < D ? key(r, bn + t) : 0.0f;
                    }
                }
                for (int t = 0; t < 4; t++) result += it[t] * vc[t];
            }
            for (ushort s = 16; s >= 1; s >>= 1) result += simd_shuffle_down(result, s);
            if (lane == 0) part[sg] = result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float res = part[0];
            for (int s = 1; s < 8; s++) res += part[s];
            sc[r] = res * scale[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 5. softmax over the N keys (serial)
    if (lid == 0) {
        float m = -INFINITY;
        for (int r = 0; r < N; ++r) m = metal::max(m, sc[r]);
        float e[N];
        for (int r = 0; r < N; ++r) e[r] = metal::precise::exp(sc[r] - m);
        float z = 0;
        for (int r = 0; r < N; ++r) z = e[r] + z;
        for (int r = 0; r < N; ++r) wts[r] = e[r] / z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 6. weights @ keys (gemv_t sm8 sn4 tm4 grouping)
    for (int d = (int)lid; d < D; d += (int)TG) {
        float g[8];
        for (int m = 0; m < 8; ++m) {
            g[m] = 0;
            for (int t = 0; t < 4 && 4 * m + t < N; ++t) {
                const float vcf = wts[4 * m + t];
                g[m] += vcf * key(4 * m + t, d);
            }
        }
        const float a = (g[0] + g[4]) + (g[2] + g[6]);
        const float c = (g[1] + g[5]) + (g[3] + g[7]);
        hidden[d] = a + c;
        ov[d] = a + c;
    }
    // 7. post_attention_layernorm(hidden).astype(bf16)
    if (NORM) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float hx[4];
        float acc = 0;
        for (int i = 0; i < 4; i++) { hx[i] = ov[n0 + i]; acc += hx[i] * hx[i]; }
        const float inv = rms_inv(acc, eps[0]);
        for (int i = 0; i < 4; i++) {
            const float wv = static_cast<float>(nw[n0 + i]);
            hn[n0 + i] = static_cast<bfloat16_t>(wv * static_cast<float>(hx[i] * inv));
        }
    }
"""


def _dpar_tg_floats(n_keys):
    """Extra threadgroup floats mega_dpar declares (pv2[N*8] + ex[N])."""
    from . import fast_kernels as fk
    return 9 * (n_keys + 1) if fk.enabled("mega_dpar") else 0


def post_supported(module, state, attn, x, n_keys, query):
    """n_keys: keys the depth attention sees besides the partial (incl. a
    boundary key)."""
    D = x.shape[-1]
    return (
        x.size == D and attn.size == D
        and x.dtype == mx.bfloat16 and attn.dtype == mx.bfloat16
        and state.dtype == mx.float32 and state.shape[1] == 3
        and module.kernel_size == 4
        and query.ndim == 1
        and D % 128 == 0 and 256 <= D // 4 <= 1024
        and 1 <= n_keys + 1 <= 32
        # 31 Metal buffer slots: nki keys + 9 inputs + 5 outputs (megafix)
        and n_keys + 14 <= 31
        # threadgroup pk/bk/ov[D] + red/sc/wts/part floats <= 32 KB
        # + mega_dpar's pv2[N*8] and ex[N], N = n_keys + 1
        and (3 * D + 32 + 2 * 32 + 8 + _dpar_tg_floats(n_keys)) * 4 <= 32 * 1024
    )


def post_block(module, state, attn, x, boundary, keys, query, norm=None):
    """See the block comment above. keys: existing block keys ([1,1,1,D] f32,
    NOT including the boundary key). Returns dict with residual [1,1,D] f32,
    nstate, hidden [1,1,1,D] f32, and bkey [1,1,1,D] / hn [1,1,D] bf16 when
    requested."""
    D = x.shape[-1]
    nki = len(keys)
    name = f"post{nki}"
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name=f"inkling2_mega_{name}" + _dp_suffix(),
            input_names=[f"k{i}" for i in range(nki)]
            + ["state", "weight", "xin", "skip", "query", "scale", "divisor", "nw", "eps"],
            output_names=["residual", "nstate", "hidden", "bkey", "hn"],
            source=_post_source(nki),
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    from . import fast_kernels as fk
    W = module.kernel_size
    has_norm = norm is not None
    outs = k(
        inputs=[*keys, state, module.conv.weight, attn, x, query,
                fk._scalar(D**-0.5), fk.sconv_divisor(W),
                norm.weight if has_norm else query,
                fk._scalar(norm.eps if has_norm else 1e-6)],
        template=[("D", D), ("TG", D // 4), ("NKI", nki), ("W", W),
                  ("BKEY", bool(boundary)), ("NORM", has_norm)],
        grid=(D // 4, 1, 1),
        threadgroup=(D // 4, 1, 1),
        output_shapes=[(1, 1, D), state.shape, (1, 1, 1, D),
                       (1, 1, 1, D) if boundary else (1,),
                       (1, 1, D) if has_norm else (1,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.bfloat16],
    )
    res = {"residual": outs[0], "nstate": outs[1], "hidden": outs[2]}
    if boundary:
        res["bkey"] = outs[3]
    if has_norm:
        res["hn"] = outs[4]
    return res


# --------------------------------------------------------------------------
# qkvclip + sink_proj GEMV + exact decode mask in one launch
# (INKLING_FAST=mega_qsm; B*L == 1, needs qkvclip + xmask + exact FD).
# Extra threadgroups (one simdgroup each, one per query head h):
#   r_h  = the qkvclip r row (same code -> same values as ro)
#   sink = sink_proj(x)[h]: split-K gemv bm1 bn8 sm1 sn32 tm4 tn4 row,
#          its 8 simdgroups emulated in order by one simdgroup (same lane
#          data, same shuffles, same sequential threadgroup sum)
#   mask[h, j] for all columns: fast_kernels.exact_mask formula.
# --------------------------------------------------------------------------

_QSM_EXTRA = r"""
    if (row >= HQ + 2 * HK + HQ) {
        const int h = row - (HQ + 2 * HK + HQ);
        // r row (as the qkvclip r branch)
        float sum = 0.0f;
        for (int j = 0; j < DR; ++j) {
            const int c = h * DR + j;
            const float xv = float(r[b * r_strides[0] + l * r_strides[1] + c * r_strides[2]])
                             * float(rel_scale[h * rel_scale_strides[0] + j * rel_scale_strides[1]]);
            sum = _sq(xv) + sum;
        }
        float f = rthr[0] / (metal::precise::sqrt(sum) + 1e-6f);
        f = metal::min(f, 1.0f);
        float rv[DR];
        for (int j = 0; j < DR; ++j) {
            const int c = h * DR + j;
            const float xv = float(r[b * r_strides[0] + l * r_strides[1] + c * r_strides[2]])
                             * float(rel_scale[h * rel_scale_strides[0] + j * rel_scale_strides[1]]);
            T o = T(xv * f);
            if (HAS_TAU) o = T(float(o) * tau);
            rv[j] = float(o);
        }
        // sink logit: row h of sw @ xs (split-K gemv, 8 virtual simdgroups)
        float res[8];
        for (int vs = 0; vs < 8; ++vs) {
            float result = 0.0f;
            int bn = (vs * 32 + (int)lane) * 4;
            for (int it = 0; it < DX / 1024; ++it) {
                float vc[4];
                for (int t = 0; t < 4; t++) vc[t] = static_cast<float>(xs[bn + t]);
                T inter[4];
                for (int t = 0; t < 4; t++) inter[t] = sw[(size_t)h * DX + bn + t];
                for (int t = 0; t < 4; t++) result += inter[t] * vc[t];
                bn += 1024;
            }
            if (DX % 1024 > 0) {
                const bool full = bn + 4 <= DX;
                float vc[4];
                T inter[4];
                for (int t = 0; t < 4; t++) vc[t] = (full || bn + t < DX) ? static_cast<float>(xs[bn + t]) : 0.0f;
                for (int t = 0; t < 4; t++) inter[t] = (full || bn + t < DX) ? sw[(size_t)h * DX + bn + t] : T(0);
                for (int t = 0; t < 4; t++) result += inter[t] * vc[t];
            }
            for (ushort s = 16; s >= 1; s >>= 1) result += simd_shuffle_down(result, s);
            res[vs] = simd_broadcast(result, 0);
        }
        float sres = res[0];
        for (int vs = 1; vs < 8; vs++) sres += res[vs];
        const T sinkv = T(sres);
        // exact decode mask row
        const int p = qpos[0];  // B == 1
        for (int j = (int)lane; j < NCOL; j += 32) {
            int dist, sink_col;
            bool valid;
            if (GATHER) {
                const int S = metal::min(p + 1, WINDOW);
                const int base = metal::max(p - WINDOW + 1, 0);
                valid = j < S;
                dist = p - (base + j);
                sink_col = S;
            } else {
                valid = j <= p;
                dist = p - j;
                sink_col = p + 1;
            }
            T val;
            if (j == sink_col) {
                val = sinkv;
            } else if (!valid || dist < 0 || (WINDOW > 0 && dist >= WINDOW)) {
                val = (T)(-1e30f);
            } else if (dist < REL_EXTENT) {
                float acc = 0.0f;
                for (int d = 0; d < DR; ++d)
                    acc += rv[d] * (float)proj[(size_t)d * proj_strides[0] + (size_t)dist * proj_strides[1]];
                val = (T)acc;
            } else {
                val = (T)0;
            }
            mask[(size_t)h * NCOL + j] = val;
        }
        return;
    }
"""


# mega_qsm2 (megaperf): the qsm extra rows were one 32-lane threadgroup per
# query head doing the r row, the sink GEMV row AND every mask column serially
# (NCOL/32 iterations of a DR-long dot each). Now the extra rows are
# [split][head]: split 0 = the sink GEMV + the sink column only; splits 1..
# NSPL = QCPS-column slices of the mask (each recomputes the cheap r row).
# Every mask value is produced by the same code -> bitwise the same mask.
_QCPS = 128


def _qsm2(src):
    a = """    if (row >= HQ + 2 * HK + HQ) {
        const int h = row - (HQ + 2 * HK + HQ);"""
    assert a in src
    src = src.replace(a, """    if (row >= HQ + 2 * HK + HQ) {
        const int xr = row - (HQ + 2 * HK + HQ);
        const int h = xr % HQ;      // extra rows: [split][head]
        const int split = xr / HQ;  // 0: sink GEMV + sink column; 1..: column slices""", 1)
    i0 = src.index("        // sink logit")
    i1 = src.index("        const T sinkv = T(sres);")
    i2 = i1 + len("        const T sinkv = T(sres);")
    body = src[i0:i1]
    body = "\n".join("    " + l if l.strip() else l for l in body.split("\n"))
    src = (src[:i0] + "        T sinkv = T(0);\n        if (split == 0) {\n" + body
           + "            sinkv = T(sres);\n        }" + src[i2:])
    b = """        const int p = qpos[0];  // B == 1
        for (int j = (int)lane; j < NCOL; j += 32) {"""
    assert b in src
    src = src.replace(b, """        const int p = qpos[0];  // B == 1
        const int sc_ = GATHER ? metal::min(p + 1, WINDOW) : p + 1;
        int jb, je, js;
        if (split == 0) {
            jb = sc_; je = ((int)lane == 0 && sc_ < NCOL) ? sc_ + 1 : sc_; js = 1;
        } else {
            jb = (split - 1) * QCPS + (int)lane; je = metal::min(NCOL, split * QCPS); js = 32;
        }
        for (int j = jb; j < je; j += js) {
            if (split != 0 && j == sc_) continue;""", 1)
    return src


def qsm_supported(attn, cfg, x, proj):
    from . import fast_kernels as fk
    sw = attn.sink_proj.weight if hasattr(attn.sink_proj, "weight") else None
    return (
        fk.qkv_clip_supported(attn, cfg, x)
        and x.size == x.shape[-1]
        and x.dtype == mx.bfloat16
        and sw is not None and sw.dtype == mx.bfloat16
        and "bias" not in attn.sink_proj
        and type(attn.sink_proj).__name__ == "Linear"
        and sw.shape == (attn.n_heads, x.shape[-1])
        and x.shape[-1] >= 16 * attn.n_heads  # split-K gemv config
        and proj.dtype == mx.bfloat16
    )


_qsm_cst = {}


def _qsm_consts(cfg):
    """[qk clip thr, rel clip thr, eps] f32, cached (fk._scalar values)."""
    key = (cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps)
    a = _qsm_cst.get(key)
    if a is None:
        a = _qsm_cst[key] = mx.array(list(key), mx.float32)
        mx.eval(a)
    return a


def qkv_sink_mask(q, k, v, r, attn, cfg, tau, xin, proj, q_pos, ncol, window, gather, zrow,
                  kbuf=None, vbuf=None, dep=None):
    """fast_kernels.qkv_clip(...) + the exact decode mask (sinks from
    sink_proj(xin)). Returns qo, ko, vo, ro, mask [1, H, 1, ncol].

    kbuf/vbuf given (needs zrow): k, v and the zero row are ALSO written in
    place into the buffers at slots q_pos, q_pos + 1. MLX does not see that
    write, so the caller must (a) only read the buffers through ops that
    consume ``mask`` (SDPA: its mask input is this kernel's output, so MLX's
    encoder puts a full buffer barrier before it and orders it after this
    kernel), (b) not need the old buffer contents (the FD state is replaced
    every step and each visit owns its buffers).

    Cross-step: this step writes slot p (k/v) over what the
    previous step's SDPA read as its zero sink row.  MLX tracks only arrays a
    kernel *outputs*, so nothing orders that write-after-read unless this
    kernel has an input written after the previous SDPA.  ``dep`` (never read
    by the kernel) is that input: FastDecoder passes the visit's mlp conv
    state from the previous step, produced downstream of its SDPA.  Binding
    it makes MLX put a buffer barrier (same encoder) or a fence wait (later
    command buffer) in front of this kernel whenever the producer is still
    in flight, so the order no longer rests on the sampled-token data path.
    KVIP still mutates the incoming KV arrays in place: a saved copy of an
    earlier ``dec.state`` changes too (slots > its position only, which its
    masks never read)."""
    from . import fast_kernels as fk
    v2 = fk.enabled("mega_qsm2")
    name = "qsm2" if v2 else "qsm"
    kern = _kern.get(name)
    if kern is None:
        src = fk._QKV_SRC.replace(
            "    if (row < HQ + HK) {", _QSM_EXTRA + "    if (row < HQ + HK) {", 1
        )
        # KVIP (INKLING_FAST=mega_kvip): also write k/v and the zero sink row
        # straight into the KV buffers at slots p, p + 1 (in place; see
        # qkv_sink_mask docstring for the ordering argument).
        kw = """                ko[zi] = o;
                ko[zi + HD] = T(0);"""
        vw = """                vo[zi] = o;
                vo[zi + HD] = T(0);"""
        assert kw in src and vw in src
        src = src.replace(kw, kw + """
                if (KVIP) {
                    device T* kbw = (device T*)kbuf;
                    // [1, HK, KS, HD] by strides
                    const size_t bi = (size_t)h * kbuf_strides[1] + (size_t)qpos[0] * kbuf_strides[2]
                                      + (size_t)((int)lane * 4 + i) * kbuf_strides[3];
                    kbw[bi] = o;
                    kbw[bi + kbuf_strides[2]] = T(0);
                }""", 1)
        src = src.replace(vw, vw + """
                if (KVIP) {
                    device T* vbw = (device T*)vbuf;
                    // [1, HK, KS, HD] by strides
                    const size_t bi = (size_t)h * vbuf_strides[1] + (size_t)qpos[0] * vbuf_strides[2]
                                      + (size_t)((int)lane * 4 + i) * vbuf_strides[3];
                    vbw[bi] = o;
                    vbw[bi + vbuf_strides[2]] = T(0);
                }""", 1)
        assert src != fk._QKV_SRC
        # 31 Metal buffer slots: KVIP strides + the ordering-only ``dep``
        # input need 3 more than the 30 the kernel used, so the three scalar
        # constants share one buffer (same f32 values: bitwise unchanged).
        assert (src.count("rthr[0]"), src.count("thr[0]"), src.count("eps[0]"), "cst" in src) == (2, 3, 1, False)
        src = src.replace("rthr[0]", "cst[1]").replace("thr[0]", "cst[0]").replace("eps[0]", "cst[2]")
        # megaperf's qsm2 rewrite runs AFTER megafix's cst rewrite;
        # its anchors don't touch the constants.  Kernel
        # names stay distinct (inkling2_mega_qsm / _qsm2): MLX caches compiled
        # libraries by name.
        if v2:
            src = _qsm2(src)
        kern = _kern[name] = mx.fast.metal_kernel(
            name="inkling2_mega_" + name,
            input_names=["q", "k", "v", "r", "q_scale", "rel_scale", "cst",
                         "tau_in", "xs", "sw", "proj", "qpos", "kbuf", "vbuf", "dep"],
            output_names=["qo", "ko", "vo", "ro", "mask"],
            header=fk._QKV_HEADER,
            source=src,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    B, L = q.shape[0], q.shape[1]
    hq, hk, hd, dr = attn.n_heads, attn.n_kv, attn.head_dim, cfg.d_rel
    dtype = q.dtype
    has_tau = tau is not None
    tau_in = mx.broadcast_to(tau.reshape(-1), (B * L,)) if has_tau else fk._scalar(1.0)
    DX = xin.shape[-1]
    return kern(
        inputs=[q, k, v, r, attn.q_scale, attn.rel_scale,
                _qsm_consts(cfg), tau_in,
                xin.reshape(DX), attn.sink_proj.weight, proj, q_pos,
                # dummies (never written) must be device bf16 buffers too
                kbuf if kbuf is not None else attn.sink_proj.weight,
                vbuf if vbuf is not None else attn.sink_proj.weight,
                dep if dep is not None else q_pos],
        template=[("T", dtype), ("HQ", hq), ("HK", hk), ("HD", hd), ("DR", dr),
                  ("HAS_TAU", has_tau), ("VNORM", bool(cfg.use_value_norm_no_gamma)),
                  ("ZROW", bool(zrow)), ("DX", DX), ("NCOL", int(ncol)),
                  ("WINDOW", int(window)), ("GATHER", bool(gather)),
                  ("REL_EXTENT", proj.shape[1]), ("KVIP", kbuf is not None),
                  ("KS", kbuf.shape[2] if kbuf is not None else 1)]
        + ([("QCPS", _QCPS)] if v2 else []),
        grid=(32, hq + 2 * hk + hq + hq * ((1 + -(-int(ncol) // _QCPS)) if v2 else 1), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, L, hq, hd)]
        + ([(1, hk, 2, hd)] * 2 if zrow else [(B, L, hk, hd)] * 2)
        + [(B, L, hq, dr), (1, hq, 1, int(ncol))],
        output_dtypes=[dtype] * 5,
    )


# --------------------------------------------------------------------------
# Sliding k and v ShortConvolution (glue path, L=1) in one launch
# (INKLING_FAST=mega_sconv2). Same formula as fast_kernels._SCONV_SRC.
# --------------------------------------------------------------------------
_SCONV2_SRC = r"""
    const uint i = thread_position_in_grid.x;
    const int which = (int)thread_position_in_grid.y;
    if ((int)i >= C) return;
    const int c = (int)i;
    const device T* x = which ? xv : xk;
    const device float* st = which ? sv : sk;
    const device WT* w = which ? wv : wk;
    device T* o = which ? ov : ok;
    device float* ns = which ? nsv : nsk;
    const float xc = float(x[c]);
    float value = 0;
    for (int j = 0; j < W; ++j) {
        const float xvv = j < W - 1 ? st[j * C + c] : xc;
        const float wf = w[c * W + j];
        value = value + xvv * wf;
    }
    o[c] = T((value + xc) / divisor);
    for (int s = 0; s < W - 1; ++s)
        ns[s * C + c] = s < W - 2 ? st[(s + 1) * C + c] : xc;
"""


def sconv2_supported(kmod, vmod, xk, xv, sk, sv):
    C = xk.shape[-1]
    return (
        xk.size == C and xv.size == C and xv.shape[-1] == C
        and xk.dtype == mx.bfloat16 and xv.dtype == mx.bfloat16
        and kmod.kernel_size == 4 and vmod.kernel_size == 4
        and sk is not None and sv is not None
        and sk.dtype == mx.float32 and sv.dtype == mx.float32
        and sk.shape == (1, 3, C) and sv.shape == (1, 3, C)
        and kmod.conv.weight.dtype == vmod.conv.weight.dtype
        and kmod.conv.weight.shape == (C, 4, 1) and vmod.conv.weight.shape == (C, 4, 1)
        and C % 128 == 0
    )


def sconv_pair(kmod, vmod, xk, xv, sk, sv):
    from . import fast_kernels as fk
    C = xk.shape[-1]
    k = _kern.get("sconv2")
    if k is None:
        k = _kern["sconv2"] = mx.fast.metal_kernel(
            name="inkling2_mega_sconv2",
            input_names=["xk", "xv", "sk", "sv", "wk", "wv", "divisor"],
            output_names=["ok", "ov", "nsk", "nsv"],
            source=_SCONV2_SRC,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    return k(
        inputs=[xk, xv, sk, sv, kmod.conv.weight, vmod.conv.weight, fk.sconv_divisor(4)],
        template=[("T", xk.dtype), ("WT", kmod.conv.weight.dtype), ("W", 4), ("C", C)],
        grid=(C, 2, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[xk.shape, xv.shape, sk.shape, sv.shape],
        output_dtypes=[xk.dtype, xv.dtype, mx.float32, mx.float32],
    )


# --------------------------------------------------------------------------
# Output gate folded into the o_proj GEMV (INKLING_FAST=mega_oproj, B*L == 1)
#   g = (out.f32 * sigmoid(gate.f32)).astype(bf16)   (compiled output_gate)
#   y = g @ Wo.T   gemv bm4 bn1 sm1 sn32 tm4 tn4 (K < 16*N, N < 4096)
# The gate is applied as the GEMV loads its input vector (each threadgroup
# recomputes the gated values it reads; same values as the gate kernel).
# --------------------------------------------------------------------------
_OPROJ_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int out_row = (int)threadgroup_position_in_grid.x * 16 + (int)sg * 4;
    if (out_row >= N) return;
    const int64_t ws0 = w_strides[0], ws1 = w_strides[1];
    float result[4] = {0, 0, 0, 0};
    int bn = (int)lane * 4;
    for (int it = 0; it < K / 128; ++it) {
        float vc[4];
        for (int t = 0; t < 4; t++) {
            const int c = bn + t;
            const float o = float(att[c * att_strides[2]]);
            const float x = float(gate[c * gate_strides[2]]);
            const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            const float sgm = (x < 0) ? y : 1 - y;
            vc[t] = static_cast<float>(static_cast<bfloat16_t>(o * sgm));
        }
        for (int tm = 0; tm < 4; tm++) {
            bfloat16_t inter[4];
            for (int t = 0; t < 4; t++) inter[t] = w[(size_t)(out_row + tm) * ws0 + (size_t)(bn + t) * ws1];
            for (int t = 0; t < 4; t++) result[tm] += inter[t] * vc[t];
        }
        bn += 128;
    }
    for (int tm = 0; tm < 4; tm++)
        for (ushort s = 16; s >= 1; s >>= 1) result[tm] += simd_shuffle_down(result[tm], s);
    if (lane == 0)
        for (int tm = 0; tm < 4; tm++) y_out[out_row + tm] = static_cast<bfloat16_t>(result[tm]);
"""


# mega_oproj2 (megaperf): same per-row arithmetic as _OPROJ_SRC (lane loads
# columns lane*4 + 128*it + t, result += inter[t] * vc[t] in order, then the
# shuffle_down tree), but (a) the gated input vector is computed ONCE per
# threadgroup into threadgroup memory (was: every simdgroup recomputed all K
# sigmoids), and (b) one output row per simdgroup, RPT rows per threadgroup,
# so N simdgroups run instead of N/4 (the N=2304 GEMV under-filled the GPU at
# ~580 GB/s). Bitwise the same values.
_OPROJ2_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint lid = thread_position_in_threadgroup.x;
    threadgroup float gv[K];
    for (int c = (int)lid; c < K; c += RPT * 32) {
        const float o = float(att[c * att_strides[2]]);
        const float x = float(gate[c * gate_strides[2]]);
        const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        const float sgm = (x < 0) ? y : 1 - y;
        gv[c] = static_cast<float>(static_cast<bfloat16_t>(o * sgm));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int out_row = (int)threadgroup_position_in_grid.x * RPT + (int)sg;
    if (out_row >= N) return;
    const int64_t ws0 = w_strides[0], ws1 = w_strides[1];
    float result = 0;
    int bn = (int)lane * 4;
    for (int it = 0; it < K / 128; ++it) {
        float vc[4];
        for (int t = 0; t < 4; t++) vc[t] = gv[bn + t];
        bfloat16_t inter[4];
        for (int t = 0; t < 4; t++) inter[t] = w[(size_t)out_row * ws0 + (size_t)(bn + t) * ws1];
        for (int t = 0; t < 4; t++) result += inter[t] * vc[t];
        bn += 128;
    }
    for (ushort s = 16; s >= 1; s >>= 1) result += simd_shuffle_down(result, s);
    if (lane == 0) y_out[out_row] = static_cast<bfloat16_t>(result);
"""
_OPROJ2_RPT = 16


def oproj_supported(o_proj, att, gate):
    """Bitwise only for a row-major o_proj.weight: stock ``x @ w.T`` then runs
    the non-transposed gemv this kernel copies; a transposed/column view
    would send stock to gemv_t (other K order) while the kernel follows
    strides.  MLX's Python API exposes no strides/flags to
    check here; checkpoint weights are dense row-major, so this stays a
    documented precondition (assigning a transposed view to o_proj.weight
    breaks bitwise, not correctness)."""
    w = getattr(o_proj, "weight", None)
    if w is None or type(o_proj).__name__ != "Linear" or "bias" in o_proj:
        return False
    N, K = w.shape
    return (
        w.dtype == mx.bfloat16 and att.dtype == mx.bfloat16 and gate.dtype == mx.bfloat16
        and att.size == K and gate.size == K
        and att.ndim == 3 and gate.ndim == 3
        and K % 128 == 0 and N % 16 == 0
        and K > 64 and K < 16 * N and N < 4096
    )


def gate_oproj(o_proj, att, gate):
    """o_proj(output_gate(att, gate)) for one row; att, gate [1, 1, K] bf16."""
    from . import fast_kernels as fk
    N, K = o_proj.weight.shape
    v2 = fk.enabled("mega_oproj2") and K * 4 <= 16384 and N % _OPROJ2_RPT == 0
    name = "oproj2" if v2 else "oproj"
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name="inkling2_mega_" + name,
            input_names=["att", "gate", "w"],
            output_names=["y_out"],
            source=_OPROJ2_SRC if v2 else _OPROJ_SRC,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    if v2:
        return k(
            inputs=[att, gate, o_proj.weight],
            template=[("N", N), ("K", K), ("RPT", _OPROJ2_RPT)],
            grid=(32 * N, 1, 1),
            threadgroup=(32 * _OPROJ2_RPT, 1, 1),
            output_shapes=[(1, 1, N)],
            output_dtypes=[mx.bfloat16],
        )[0]
    return k(
        inputs=[att, gate, o_proj.weight],
        template=[("N", N), ("K", K)],
        grid=(128 * (N // 16), 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


# --------------------------------------------------------------------------
# Pre-attention block (INKLING_FAST=mega_pre), one threadgroup per row:
#   expert combine + mlp_sconv + residual (sconv_tail MODE 1) -> x_next (bf16)
#   next visit's attention_input depth attention (1 query) on x_next
#   [next visit's input_layernorm(mixed).astype(bf16)]
# Same kernel skeleton as post_block (stages 2, 4-7 shared).
# --------------------------------------------------------------------------
_PRE_STAGE1 = r"""    // 1. expert combine + mlp sconv + residual (sconv_tail MODE 1) -> x_next
    float tx[4];
    {
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            const int c = n0 + i;
            float total = 0.0f;
            for (int kk = 0; kk < TOPK; ++kk)
                total = metal::fma(float(xin[kk * D + c]), rw[kk], -0.0f) + total;
            const float hv = float(bfloat16_t(total));
            float value = 0;
            for (int j = 0; j < W; ++j) {
                float xv = j < W - 1 ? state[j * D + c] : hv;
                float w = weight[c * W + j];
                value = value + xv * w;
            }
            const bfloat16_t s = bfloat16_t((value + hv) / divisor);
            for (int j = 0; j < W - 1; ++j)
                nstate[j * D + c] = j < W - 2 ? state[(j + 1) * D + c] : hv;
            const bfloat16_t xo = bfloat16_t(skip[c] + float(s));
            residual[c] = xo;
            tx[i] = float(xo);
            acc += tx[i] * tx[i];
        }
        const float inv = rms_inv(acc, 1e-6f);
        for (int i = 0; i < 4; i++) pk[n0 + i] = 1.0f * static_cast<float>(tx[i] * inv);
    }
"""


def _pre_source(nki):
    src = _post_source(nki)
    a = src.index("    // 1. attn sconv tail")
    b = src.index("    // 3. boundary key")
    return src[:a] + _PRE_STAGE1 + src[b:]


def pre_supported(module, state, routes, residual, n_keys, query):
    D = residual.shape[-1]
    return (
        residual.size == D and residual.dtype == mx.float32
        and routes.dtype == mx.bfloat16 and routes.size == routes.shape[-2] * D
        and state.dtype == mx.float32 and state.shape[1] == 3
        and module.kernel_size == 4 and query.ndim == 1
        and D % 128 == 0 and 256 <= D // 4 <= 1024
        and 1 <= n_keys + 1 <= 32
        # 31 Metal buffer slots: nki keys + 10 inputs + 5 outputs (megafix)
        and n_keys + 15 <= 31
        # same threadgroup arrays as post_block
        and (3 * D + 32 + 2 * 32 + 8 + _dpar_tg_floats(n_keys)) * 4 <= 32 * 1024
    )


def pre_block(module, state, routes, residual, weights, keys, query, norm=None):
    """Returns dict: x [1,1,D] bf16 (layer output), nstate, mixed
    [1,1,1,D] f32 (next attention_input), hn [1,1,D] bf16 if norm."""
    from . import fast_kernels as fk
    D = residual.shape[-1]
    nki = len(keys)
    name = f"pre{nki}"
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name=f"inkling2_mega_{name}" + _dp_suffix(),
            input_names=[f"k{i}" for i in range(nki)]
            + ["state", "weight", "xin", "rw", "skip", "query", "scale", "divisor", "nw", "eps"],
            output_names=["residual", "nstate", "hidden", "bkey", "hn"],
            source=_pre_source(nki),
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    W = module.kernel_size
    K = routes.shape[-2]
    has_norm = norm is not None
    outs = k(
        inputs=[*keys, state, module.conv.weight, routes, weights, residual, query,
                fk._scalar(D**-0.5), fk.sconv_divisor(W),
                norm.weight if has_norm else query,
                fk._scalar(norm.eps if has_norm else 1e-6)],
        template=[("D", D), ("TG", D // 4), ("NKI", nki), ("W", W), ("TOPK", K),
                  ("BKEY", False), ("NORM", has_norm)],
        grid=(D // 4, 1, 1),
        threadgroup=(D // 4, 1, 1),
        output_shapes=[(1, 1, D), state.shape, (1, 1, 1, D), (1,),
                       (1, 1, D) if has_norm else (1,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32, mx.float32, mx.bfloat16],
    )
    res = {"x": outs[0], "nstate": outs[1], "mixed": outs[2]}
    if has_norm:
        res["hn"] = outs[4]
    return res


# --------------------------------------------------------------------------
# 15. mega_sdpa: exact decode SDPA (B=1, L=1, float mask, no sinks= arg)
#     reading the KV buffers in place (megaperf2).
# --------------------------------------------------------------------------
# Reproduces, bit for bit, the MLX 0.32.4 kernels (ml-explore/mlx, MIT,
# kernels/sdpa_vector.h; dispatch in scaled_dot_product_attention.cpp) that
# mx.fast.scaled_dot_product_attention runs for the FastDecoder exact operands
# (q [1, HQ, 1, D], keys [1, HK, N, D], float mask [1, HQ, 1, N]):
# * N < 1024 on 'd'/'s' GPUs: sdpa_vector (1-pass). One 1024-thread
#   threadgroup per q head; simdgroup s walks keys s, s + 32, ... serially
#   (online softmax: max, fast::exp factor, sum, o = o * f + e * v), then the
#   32 partials are combined (simd_max, simd_sum through threadgroup memory).
# * N >= 1024: sdpa_vector_2pass_1 (grid HK x 1 x blocks, threadgroup
#   32 x gqa; block b walks keys b, b + blocks, ...; partial o rounded to T)
#   then sdpa_vector_2pass_2 (blocks/32 rounds, same combine). blocks from the
#   same device-class rule (and MLX_SDPA_BLOCKS).
# Per key, per lane, every value is computed with the same expression as the
# stock kernel, in the same order. What differs:
# 1. Keys are read in place: the sliced regime reads buffer rows 0..N-1 (no
#    view copy); the gather regime (sliding ring, cap > slice cap) maps
#    temporal index i -> ring slot (base + i) % RING, i >= S -> the zero row
#    RING (exactly what ring_gather copies), so ring_gather is not launched.
# 2. Trailing keys after the sink row (index > n_real - 1, mask -1e30) are
#    skipped, but each simdgroup/block still processes its first key. Those
#    keys are exact no-ops for a chain that already saw a key: factor =
#    exp(0) = 1, e = exp(-1e30 - max) = 0 (the current exact path relies on
#    the same identity to feed cap + 1 keys where stock feeds S + 1).
# 3. Loads for the next TK keys are issued before the serial update chain
#    (the stock 1-pass kernel loads mask, then key, inside the chain).
_SDPA1_SRC = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int QPT = D / BD;
    typedef float U;
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    const int simd_gid = (int)simdgroup_index_in_threadgroup;
    const int simd_lid = (int)thread_index_in_simdgroup;
    const int h = (int)threadgroup_position_in_grid.x;
    const int kvh = h / (HQ / HK);
    const int p = qpos[0];
    int S = 0, base = 0, nreal;
    if (GATHER) {
        S = metal::min(p + 1, WINDOW);
        base = metal::max(p - WINDOW + 1, 0);
        nreal = S + 1;
    } else {
        nreal = p + 2;
    }
    const int nuse = metal::min(N, metal::max(nreal, BN));
    const device T* kh = kb + (size_t)kvh * KS * D + simd_lid * QPT;
    const device T* vh = vb + (size_t)kvh * KS * D + simd_lid * QPT;
    const device T* mh = mask + (size_t)h * N;
    const U sc = scale[0];
    U q[QPT];
    U o[QPT];
    for (int i = 0; i < QPT; i++) {
        q[i] = static_cast<U>(sc) * queries[(size_t)h * D + simd_lid * QPT + i];
        o[i] = 0;
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    for (int i0 = simd_gid; i0 < nuse; i0 += BN * TK) {
        U scv[TK];
        bool uk[TK];
        T vr[TK][QPT];
        // scores first (each is the stock expression; they do not depend on
        // the online-softmax state), unconditional clamped loads
        #pragma clang loop unroll(full)
        for (int t = 0; t < TK; t++) {
            const int i = i0 + t * BN;
            const bool in = i < nuse;
            const int ic = in ? i : 0;
            const int row = GATHER ? (ic < S ? (base + ic) % RING : RING) : ic;
            U k[QPT];
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; j++) k[j] = kh[(size_t)row * D + j];
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; j++) vr[t][j] = vh[(size_t)row * D + j];
            const T m = mh[ic];
            uk[t] = in && (m >= Limits<T>::finite_min);
            U score = 0;
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; j++) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);
            score += static_cast<U>(m);
            scv[t] = score;
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < TK; t++) {
            if (uk[t]) {
                const U score = scv[t];
                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPT; j++) {
                    o[j] = o[j] * factor + exp_score * vr[t][j];
                }
            }
        }
    }
    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
    for (int i = 0; i < QPT; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        for (int i = 0; i < QPT; i++) {
            out[(size_t)h * D + simd_gid * QPT + i] = static_cast<T>(o[i]);
        }
    }
"""

# sdpa_vector_2pass_1: threadgroup (32, gqa, 1) per (kv head, block).
_SDPA2A_SRC = r"""
    constexpr int BD = 32;
    constexpr int QPT = D / BD;
    constexpr int G = HQ / HK;
    typedef float U;
    const int simd_lid = (int)thread_index_in_simdgroup;
    const int kvh = (int)threadgroup_position_in_grid.x;
    const int block = (int)threadgroup_position_in_grid.z;
    const int h = G * kvh + (int)thread_position_in_threadgroup.y;
    const int p = qpos[0];
    int S = 0, base = 0, nreal;
    if (GATHER) {
        S = metal::min(p + 1, WINDOW);
        base = metal::max(p - WINDOW + 1, 0);
        nreal = S + 1;
    } else {
        nreal = p + 2;
    }
    const int nuse = metal::min(N, metal::max(nreal, BLOCKS));
    const device T* kh = kb + (size_t)kvh * KS * D + simd_lid * QPT;
    const device T* vh = vb + (size_t)kvh * KS * D + simd_lid * QPT;
    const device T* mh = mask + (size_t)h * N;
    const U sc = scale[0];
    U q[QPT];
    U o[QPT];
    for (int i = 0; i < QPT; i++) {
        q[i] = static_cast<U>(sc) * queries[(size_t)h * D + simd_lid * QPT + i];
        o[i] = 0;
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    for (int i0 = block; i0 < nuse; i0 += BLOCKS * TK) {
        U scv[TK];
        bool uk[TK];
        T vr[TK][QPT];
        // scores first (each is the stock expression; they do not depend on
        // the online-softmax state), unconditional clamped loads
        #pragma clang loop unroll(full)
        for (int t = 0; t < TK; t++) {
            const int i = i0 + t * BLOCKS;
            const bool in = i < nuse;
            const int ic = in ? i : 0;
            const int row = GATHER ? (ic < S ? (base + ic) % RING : RING) : ic;
            U k[QPT];
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; j++) k[j] = kh[(size_t)row * D + j];
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; j++) vr[t][j] = vh[(size_t)row * D + j];
            const T m = mh[ic];
            uk[t] = in && (m >= Limits<T>::finite_min);
            U score = 0;
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPT; j++) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);
            score += static_cast<U>(m);
            scv[t] = score;
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < TK; t++) {
            if (uk[t]) {
                const U score = scv[t];
                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPT; j++) {
                    o[j] = o[j] * factor + exp_score * static_cast<U>(vr[t][j]);
                }
            }
        }
    }
    if (simd_lid == 0) {
        sums[h * BLOCKS + block] = sum_exp_score;
        maxs[h * BLOCKS + block] = max_score;
    }
    for (int i = 0; i < QPT; i++) {
        part[((size_t)h * BLOCKS + block) * D + simd_lid * QPT + i] = static_cast<T>(o[i]);
    }
"""

# sdpa_vector_2pass_2, verbatim (one 1024-thread threadgroup per q head).
_SDPA2B_SRC = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int EPT = D / BD;
    typedef float U;
    threadgroup U outputs[BN * BD];
    const int simd_gid = (int)simdgroup_index_in_threadgroup;
    const int simd_lid = (int)thread_index_in_simdgroup;
    const int h = (int)threadgroup_position_in_grid.x;
    U o[EPT];
    for (int i = 0; i < EPT; i++) o[i] = 0;
    const device T* partials = part + (size_t)h * BLOCKS * D + simd_gid * D + simd_lid * EPT;
    const device float* sm = sums + h * BLOCKS;
    const device float* mxs = maxs + h * BLOCKS;
    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;
    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mxs[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mxs[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sm[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mxs[simd_gid] - max_score);
        for (int i = 0; i < EPT; i++) {
            o[i] += factor * static_cast<U>(partials[i]);
        }
        mxs += BN;
        sm += BN;
        partials += BN * D;
    }
    for (int i = 0; i < EPT; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        for (int i = 0; i < EPT; i++) {
            out[(size_t)h * D + simd_gid * EPT + i] = static_cast<T>(o[i]);
        }
    }
"""


def _sdpa_kernel(name, src, inputs, outputs):
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name=f"inkling2_mega_{name}",
            input_names=inputs,
            output_names=outputs,
            source=src,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    return k


def sdpa_config(n_q, n_kv, N):
    """(two_pass, blocks) exactly as MLX 0.32.4 picks for L=1 decode."""
    import os
    devc = mx.device_info().get("architecture", "")[-1:]
    two = (devc in ("d", "s") and N >= 1024) or (n_kv < n_q and N >= 4096)
    if not two:
        return False, 0
    n_simds = n_q // n_kv
    if devc == "s":
        blocks = 64
        if N > 1024 and n_simds > 4:
            blocks = 128 if N <= 8192 else 256 if N <= 32768 else 512 if N <= 65536 else 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and N > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= N < 65536:
                blocks = 512
            elif N >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    env = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0)
    if env > 0:
        blocks = ((env + 31) // 32) * 32
    return True, blocks


def sdpa_supported(q, kb, vb, mask, N, gather=False, ring=None):
    if q.shape[0] != 1 or q.shape[1] != 1 or kb.shape[0] != 1:
        return False
    hq, D = q.shape[2], q.shape[3]
    hk = kb.shape[1]
    return (
        q.dtype in (mx.bfloat16, mx.float16) and kb.dtype == q.dtype
        and vb.dtype == q.dtype and mask.dtype == q.dtype
        and D % 32 == 0 and kb.shape[3] == D and vb.shape == kb.shape
        and hq % hk == 0 and tuple(mask.shape) == (1, hq, 1, N)
        # the kernel reads rows by index with no
        # bounds check: sliced reads rows 0..N-1; gather reads ring slots and
        # the zero row at index RING.
        and (kb.shape[2] == ring + 1 if gather else N <= kb.shape[2])
    )


_sdpa_selftest = {}


def sdpa_selftest(hq, hk, D, window, ring, dtype=mx.bfloat16):
    """one 1-pass and one 2-pass replica call vs
    ``mx.fast.scaled_dot_product_attention`` on random operands (sliced
    regime, as tests_fast/test_sdpa.py).  False on any mismatch: catches MLX
    routing / Metal-compiler FMA drift that the version pin cannot.  Cached
    per (hq, hk, D, dtype): runs once per process at FastDecoder init, never
    in the decode step."""
    key = (hq, hk, D, window, ring, dtype)
    if key in _sdpa_selftest:
        return _sdpa_selftest[key]
    ok = True
    try:
        key_ = mx.random.key(20261121)  # explicit keys: global RNG untouched
        for N, p in ((600, 500), (4097, 3000)):
            k1, k2, k3, k4, key_ = mx.random.split(key_, 5)
            q = mx.random.normal((1, 1, hq, D), key=k1).astype(dtype)
            row = mx.arange(N)[None, None, :, None]
            kb = mx.where(row <= p, mx.random.normal((1, hk, N, D), key=k2), 0).astype(dtype)
            vb = mx.where(row <= p, mx.random.normal((1, hk, N, D), key=k3), 0).astype(dtype)
            col = mx.arange(N)[None, None, None, :]
            m = (mx.random.normal((1, hq, 1, N), key=k4) * 3).astype(dtype)
            m = mx.where(col > p + 1, mx.array(-1e30, dtype), m)
            sc = D ** -0.5
            ref = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3), kb, vb, scale=sc, mask=m).transpose(0, 2, 1, 3)
            got = sdpa_exact(q, kb, vb, m, mx.array([p], mx.int32), N, False, window, ring, sc)
            ok = ok and bool(mx.array_equal(ref, got).item())
    except Exception as e:  # a build/launch failure also disables the replica
        import sys
        print(f"[inkling2] mega_sdpa self-test error: {e!r}", file=sys.stderr, flush=True)
        ok = False
    _sdpa_selftest[key] = ok
    return ok


SDPA_TK = int(__import__("os").environ.get("INKLING_SDPA_TK", "4"))


def sdpa_exact(q, kb, vb, mask, qpos, N, gather, window, ring, scale):
    """Exact decode attention: same result as
    ``mx.fast.scaled_dot_product_attention(q.T, k_ext, v_ext, scale, mask)``
    on the FastDecoder exact operands, where k_ext = kb[:, :, :N] (sliced,
    gather=False) or ring_gather(kb, ...) (gather=True, N = window + 1).
    q: [1, 1, HQ, D] (pre-transpose); kb/vb: whole buffers [1, HK, KS, D];
    mask: [1, HQ, 1, N]; qpos: [1] int32 position. Returns [1, 1, HQ, D]."""
    hq, D = q.shape[2], q.shape[3]
    hk, KS = kb.shape[1], kb.shape[2]
    two, blocks = sdpa_config(hq, hk, int(N))
    sc = __import__("mlx_vlm.models.inkling2.fast_kernels", fromlist=["_scalar"])._scalar(scale)
    tmpl = [("T", q.dtype), ("D", D), ("HQ", hq), ("HK", hk), ("N", int(N)),
            ("KS", KS), ("GATHER", bool(gather)), ("WINDOW", int(window)),
            ("RING", int(ring)), ("TK", SDPA_TK)]
    qpos = qpos.reshape(-1)[:1].astype(mx.int32) if qpos.dtype != mx.int32 else qpos.reshape(-1)[:1]
    if not two:
        k1 = _sdpa_kernel("sdpa1v2", _SDPA1_SRC,
                          ["queries", "kb", "vb", "mask", "qpos", "scale"], ["out"])
        return k1(
            inputs=[q, kb, vb, mask, qpos, sc], template=tmpl,
            grid=(1024 * hq, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(1, 1, hq, D)], output_dtypes=[q.dtype],
        )[0]
    ka = _sdpa_kernel("sdpa2av2", _SDPA2A_SRC,
                      ["queries", "kb", "vb", "mask", "qpos", "scale"],
                      ["part", "sums", "maxs"])
    part, sums, maxs = ka(
        inputs=[q, kb, vb, mask, qpos, sc], template=tmpl + [("BLOCKS", blocks)],
        grid=(32 * hk, hq // hk, blocks), threadgroup=(32, hq // hk, 1),
        output_shapes=[(hq, blocks, D), (hq, blocks), (hq, blocks)],
        output_dtypes=[q.dtype, mx.float32, mx.float32],
    )
    kb2 = _sdpa_kernel("sdpa2b", _SDPA2B_SRC, ["part", "sums", "maxs"], ["out"])
    return kb2(
        inputs=[part, sums, maxs], template=[("T", q.dtype), ("D", D), ("BLOCKS", blocks)],
        grid=(1024 * hq, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(1, 1, hq, D)], output_dtypes=[q.dtype],
    )[0]
