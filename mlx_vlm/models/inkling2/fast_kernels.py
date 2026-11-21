# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""Opt-in decode fast paths for Inkling 2 (``INKLING_FAST`` env flag).

``INKLING_FAST`` is a comma list read once at import. Unset means stock
behaviour everywhere. Tokens:

* ``head``  Bitwise LM head for rows (B*L) == 1: reads the BF16 unembedding
            (no cached FP32 copy needed at B=1) in exactly the per-row FMA and
            simd_shuffle_down order of MLX 0.32.4's FP32 gemv; fused softcap
            and unpadded-vocab slice. Other row counts use stock.
* ``head_approx`` The older head GEMV for rows <= 15 (FP32 accumulate,
            different order: ordering-only, NOT bitwise).
* ``mask``  Decode (L=1) rel-bias mask computed in physical ring order (no
            ``_temporal_order`` K/V copies) and, for B=1, the query sink passed
            through ``scaled_dot_product_attention(sinks=...)`` (no zero K/V
            concat, no mask concat). For B>1 the sink is written as the mask's
            last column by the same kernel.
            ``mask_ring`` / ``mask_sink`` enable only one half (for A/B).
            Not bitwise: SDPA sums keys in a different order (ring) and over
            S instead of S+1 keys (sinks); <= ~1 bf16 ulp per attention output.
* ``glue``  Cached FP32 router weight, cached per-expert global-scale factor,
            cached sconv divisor, log-n tau once per step, compiled output gate
            and MessageController, conv-state update folded into the sconv
            kernel.
* ``fuse``  One GEMV for [q; r; gate; sink] (and [k; v] when k/v share x) at
            M=1. Weights are concatenated once; the per-projection weights
            become row views of the fused buffer (memory +0).
* ``fusemoe`` One gather_qmm for the MoE gate+up at decode (L=1), when both
            share the same global scale. gate/up become views of the fused
            buffer (non-contiguous along experts). gather_qmm reads such
            views without copying, but the custom masked/paired verify kernels
            (MTP, 1<L<=8) use ensure_row_contiguous and WOULD copy them on
            every call: do not combine fusemoe with MTP until that is fixed.
"""

import math
import os

import mlx.core as mx

FLAGS = frozenset(
    t.strip() for t in os.environ.get("INKLING_FAST", "").split(",") if t.strip()
)
if "all" in FLAGS:
    FLAGS = frozenset({"head", "mask", "glue", "fuse", "fusemoe"}) | (FLAGS - {"all"})


def enabled(name):
    if name.startswith("mega_") and name in FLAGS:
        return replica_ok()
    return name in FLAGS


_REPLICA_OK = None


def replica_ok():
    """every ``mega_*`` kernel (mega_sdpa included)
    replicates MLX 0.32.4 reduction orders and kernel selection, so they run
    only on the exact verified MLX version and GPU arch; otherwise
    ``enabled("mega_*")`` is False and the stock/base paths run.  Unlike the
    import-time check, INKLING_FAST_UNVERIFIED_OK does not lift this.
    Evaluated once per process (no per-step cost)."""
    global _REPLICA_OK
    if _REPLICA_OK is None:
        try:
            arch = mx.device_info().get("architecture", "") if mx.metal.is_available() else ""
        except Exception:
            arch = ""
        _REPLICA_OK = mx.__version__ in EXACT_MLX_VERSIONS and bool(arch) and arch in EXACT_GPU_ARCHS
        if not _REPLICA_OK and any(f.startswith("mega_") for f in FLAGS):
            import sys
            print(f"[inkling2] mega_* replica kernels DISABLED: verified only on mlx "
                  f"{EXACT_MLX_VERSIONS} / {EXACT_GPU_ARCHS}, found mlx {mx.__version__} on "
                  f"{arch or '<no arch>'}; running the stock paths", file=sys.stderr, flush=True)
    return _REPLICA_OK


# Bitwise claims copy MLX kernel-selection rules (gemv split-K in
# gemv_reduction_matches_fused, SDPA 1-pass/2-pass thresholds in
# fast_decode._sdpa_cap_limits) verified only on this MLX and GPU family.
EXACT_MLX_VERSIONS = ("0.32.4",)
EXACT_GPU_ARCHS = ("applegpu_g17d",)  # M5 Ultra


def check_exact_runtime(what="INKLING_FAST"):
    """Refuse to run the bitwise fast paths on an unverified MLX/GPU.

    Override (after re-verifying with check.py's BITWISE bar) with
    INKLING_FAST_UNVERIFIED_OK=<mlx version>.
    """
    ver = mx.__version__
    arch = mx.device_info().get("architecture", "") if _on_gpu() else "cpu"
    if ver in EXACT_MLX_VERSIONS and arch in EXACT_GPU_ARCHS:
        return
    if os.environ.get("INKLING_FAST_UNVERIFIED_OK", "") == ver:
        return
    raise RuntimeError(
        f"{what}: bitwise fast paths are verified only on mlx {EXACT_MLX_VERSIONS} / "
        f"{EXACT_GPU_ARCHS}; found mlx {ver} on {arch}. Re-verify (check.py BITWISE "
        f"bar), then set INKLING_FAST_UNVERIFIED_OK={ver}, or unset INKLING_FAST."
    )



def _on_gpu():
    return mx.metal.is_available() and mx.default_device() == mx.gpu


# --------------------------------------------------------------------------
# 1. LM head: BF16 weight x FP32 activation GEMV, FP32 accumulate.
# --------------------------------------------------------------------------

_HEAD_SRC = r"""
    // h: [M, D] float32 (row contiguous), w: [V, D] bfloat16 (row contiguous)
    // out: [M, VOUT] float32. Each simdgroup owns ROWS consecutive vocab rows.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row0 = (int)((threadgroup_position_in_grid.x * NSG + sg) * ROWS);
    if (row0 >= VOUT) return;
    constexpr int D8 = D / 8;
    const device uint4* wv = (const device uint4*)w;
    const device float4* hv = (const device float4*)h;
    float acc[ROWS][M];
    for (int r = 0; r < ROWS; ++r)
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    int rows[ROWS];
    for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, VOUT - 1);
    for (int i = (int)lane; i < D8; i += 32) {
        float4 a[ROWS], b[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            uint4 u = wv[(size_t)rows[r] * D8 + i];
            // bf16 -> f32 is an exact 16-bit shift (little endian pairs).
            a[r] = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                          as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
            b[r] = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                          as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
        }
        for (int m = 0; m < M; ++m) {
            float4 h0 = hv[(size_t)m * (D / 4) + 2 * i];
            float4 h1 = hv[(size_t)m * (D / 4) + 2 * i + 1];
            for (int r = 0; r < ROWS; ++r)
                acc[r][m] += dot(a[r], h0) + dot(b[r], h1);
        }
    }
    const float c = cap[0];
    for (int r = 0; r < ROWS; ++r) {
        for (int m = 0; m < M; ++m) {
            float s = simd_sum(acc[r][m]);
            if (lane == 0 && row0 + r < VOUT) {
                if (c > 0.0f) s = metal::precise::tanh(s / c) * c;
                out[(size_t)m * VOUT + row0 + r] = s;
            }
        }
    }
"""

_head_kernel = None
_HEAD_NSG = 8


def _get_head_kernel():
    global _head_kernel
    if _head_kernel is None:
        _head_kernel = mx.fast.metal_kernel(
            name="inkling2_head_gemv_bf16w_f32a",
            input_names=["h", "w", "cap"],
            output_names=["out"],
            source=_HEAD_SRC,
            compile_options={"math_mode": "safe"},
        )
    return _head_kernel


def head_rows_per_simdgroup(m):
    # Swept on M4 Pro (V=201024, D=2304): 4 rows best up to M=12, 2 above.
    return 4 if m <= 12 else 2


def lm_head_gemv(h, weight, vout=None, cap=0.0, rows=None):
    """h: [..., D] (any float dtype, promoted to FP32), weight: [V, D] BF16.

    Returns FP32 logits [..., vout] = softcap(h @ weight[:vout].T).
    """
    lead = h.shape[:-1]
    D = h.shape[-1]
    V = weight.shape[0]
    vout = V if vout is None else int(vout)
    h2 = h.reshape(-1, D).astype(mx.float32)
    M = h2.shape[0]
    assert D % 8 == 0 and weight.dtype == mx.bfloat16 and weight.shape[1] == D
    rows = rows or head_rows_per_simdgroup(M)
    per_tg = _HEAD_NSG * rows
    n_tg = (vout + per_tg - 1) // per_tg
    if not isinstance(cap, mx.array):
        cap = mx.array([float(cap or 0.0)], mx.float32)
    out = _get_head_kernel()(
        inputs=[h2, weight, cap],
        template=[("M", M), ("D", D), ("ROWS", rows), ("NSG", _HEAD_NSG), ("VOUT", vout)],
        grid=(n_tg * _HEAD_NSG * 32, 1, 1),
        threadgroup=(_HEAD_NSG * 32, 1, 1),
        output_shapes=[(M, vout)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(*lead, vout)


# --------------------------------------------------------------------------
# 1b. Bitwise LM head for M = 1 (default for ``head``).
#
# Stock M=1 decode computes  h_f32 @ W_f32.T  (W_f32 = cached BF16->FP32 copy)
# with MLX 0.32.4's ``gemv_float32_bm8_bn1_sm1_sn32_tm4_tn4`` (matmul.cpp
# gemv_axbpy, non-transposed branch: N >= 4096 -> bm=8, K < 16N -> bn=1).
# Per output row that kernel does, for lane l in 0..31:
#     acc_l = 0;  for blk in 0..K/128-1: for t in 0..3:
#         acc_l += W[row, blk*128 + 4l + t] * h[blk*128 + 4l + t]   (FMA)
# then  acc += simd_shuffle_down(acc, s)  for s = 16, 8, 4, 2, 1  (lane 0).
# This kernel reproduces exactly that per-row order while reading the BF16
# weight (BF16->FP32 widening is exact), so it is bitwise to stock and
# streams 0.93 GB instead of the 1.85 GB FP32 copy. Softcap and vocab slice
# use the same ops stock uses (x / c, precise::tanh, * c).
# M > 1 stays on stock (steel GEMM: simdgroup-MMA order is not reproducible).
# --------------------------------------------------------------------------

_HEAD_EXACT_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row0 = (int)((threadgroup_position_in_grid.x * NSG + sg) * ROWS);
    if (row0 >= VOUT) return;
    constexpr int NBLK = D / 128;
    const device bfloat* wb = (const device bfloat*)w;
    float acc[ROWS];
    int rows[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
        rows[r] = min(row0 + r, VOUT - 1);
    }
    for (int blk = 0; blk < NBLK; ++blk) {
        const int k = blk * 128 + (int)lane * 4;
        const float4 v = *(const device float4*)(h + k);
        for (int r = 0; r < ROWS; ++r) {
            const uint2 u = *(const device uint2*)(wb + (size_t)rows[r] * D + k);
            const float w0 = as_type<float>(u.x << 16);
            const float w1 = as_type<float>(u.x & 0xffff0000u);
            const float w2 = as_type<float>(u.y << 16);
            const float w3 = as_type<float>(u.y & 0xffff0000u);
#if HEAD_FMA
            acc[r] = metal::fma(w0, v.x, acc[r]);
            acc[r] = metal::fma(w1, v.y, acc[r]);
            acc[r] = metal::fma(w2, v.z, acc[r]);
            acc[r] = metal::fma(w3, v.w, acc[r]);
#else
            acc[r] = acc[r] + w0 * v.x;
            acc[r] = acc[r] + w1 * v.y;
            acc[r] = acc[r] + w2 * v.z;
            acc[r] = acc[r] + w3 * v.w;
#endif
        }
    }
    const float c = cap[0];
    for (int r = 0; r < ROWS; ++r) {
        float s = acc[r];
        s += simd_shuffle_down(s, (ushort)16);
        s += simd_shuffle_down(s, (ushort)8);
        s += simd_shuffle_down(s, (ushort)4);
        s += simd_shuffle_down(s, (ushort)2);
        s += simd_shuffle_down(s, (ushort)1);
        if (lane == 0 && row0 + r < VOUT) {
            if (c > 0.0f) s = metal::precise::tanh(s / c) * c;
            out[row0 + r] = s;
        }
    }
"""

_head_exact_kernels = {}


def _get_head_exact_kernel(use_fma):
    k = _head_exact_kernels.get(use_fma)
    if k is None:
        k = _head_exact_kernels[use_fma] = mx.fast.metal_kernel(
            name=f"inkling2_head_gemv_exact_fma{int(use_fma)}",
            input_names=["h", "w", "cap"],
            output_names=["out"],
            header=f"#define HEAD_FMA {int(use_fma)}\n",
            source=_HEAD_EXACT_SRC,
            compile_options={"math_mode": "safe"},
        )
    return k


HEAD_EXACT_FMA = os.environ.get("INKLING_HEAD_FMA", "1") != "0"
HEAD_EXACT_ROWS = int(os.environ.get("INKLING_HEAD_ROWS", "4"))


def head_exact_supported(m, d):
    return m == 1 and d % 128 == 0


def lm_head_gemv_exact(h, weight, vout=None, cap=0.0, rows=None, use_fma=None):
    """Bitwise twin of stock's M=1 FP32 head: h [..., D] with prod(lead) == 1.

    ``h`` must already be BF16-rounded (stock casts through embed_norm dtype).
    Returns FP32 [..., vout] = softcap(h @ weight[:vout].T).
    """
    lead = h.shape[:-1]
    D = h.shape[-1]
    V = weight.shape[0]
    vout = V if vout is None else int(vout)
    h2 = h.reshape(-1, D).astype(mx.float32)
    assert h2.shape[0] == 1 and D % 128 == 0
    assert weight.dtype == mx.bfloat16 and weight.shape[1] == D
    rows = rows or HEAD_EXACT_ROWS
    use_fma = HEAD_EXACT_FMA if use_fma is None else use_fma
    per_tg = _HEAD_NSG * rows
    n_tg = (vout + per_tg - 1) // per_tg
    if not isinstance(cap, mx.array):
        cap = mx.array([float(cap or 0.0)], mx.float32)
    out = _get_head_exact_kernel(bool(use_fma))(
        inputs=[h2, weight, cap],
        template=[("D", D), ("ROWS", rows), ("NSG", _HEAD_NSG), ("VOUT", vout)],
        grid=(n_tg * _HEAD_NSG * 32, 1, 1),
        threadgroup=(_HEAD_NSG * 32, 1, 1),
        output_shapes=[(1, vout)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(*lead, vout)


# --------------------------------------------------------------------------
# 2. Rel-bias mask in physical slot order (+ optional sink column).
# --------------------------------------------------------------------------

_MASK_SRC = r"""
    // rel: [B, 1, H, D_REL] (strided), proj: [D_REL, R] (strided)
    // RING=1: qpos[b] = ring write index after the update (cache._idx);
    //         slot j holds backward distance (idx - 1 - j) mod S.
    // RING=0: slot_pos[b, j] = logical position (-1 empty), qpos[b] = query pos.
    // out: [B, H, 1, S + HAS_SINK]; column S = sinks[b, h] when HAS_SINK.
    const int B = rel_shape[0];
    const int H = rel_shape[2];
    const int S = kref_shape[2];  // unread shape carrier
    const int j = (int)thread_position_in_grid.x;
    const int bh = (int)thread_position_in_grid.z;
    if (j > S - 1 + HAS_SINK || bh >= B * H) return;
    const int b = bh / H, h = bh % H;
    const size_t o = (size_t)bh * (S + HAS_SINK) + j;
    if (HAS_SINK && j == S) {
        out[o] = (T)sinks[(size_t)b * sinks_strides[0] + (size_t)h * sinks_strides[1]];
        return;
    }
    const int q = qpos[b * qpos_strides[0]];
    int dist;
    bool empty = false;
    if (RING) {
        dist = ((q - 1 - j) % S + S) % S;
    } else {
        const int p = slot_pos[(size_t)b * slot_pos_strides[0] + (size_t)j * slot_pos_strides[1]];
        empty = p < 0;
        dist = q - p;
    }
    T val;
    if (empty || dist < 0 || (WINDOW > 0 && dist >= WINDOW)) {
        val = (T)(-1e30f);
    } else if (dist < REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0] + (size_t)h * rel_strides[2];
        for (int d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0] + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[o] = val;
"""

_mask_kernel = None


def _get_mask_kernel():
    global _mask_kernel
    if _mask_kernel is None:
        _mask_kernel = mx.fast.metal_kernel(
            name="inkling2_slot_mask",
            input_names=["rel", "proj", "slot_pos", "qpos", "sinks", "kref"],
            output_names=["out"],
            source=_MASK_SRC,
            ensure_row_contiguous=False,
        )
    return _mask_kernel


def _run_mask(rel, proj, slot_pos, qpos, kref, window, sinks, ring):
    S = kref.shape[2]
    B, L, H, d_rel = rel.shape
    assert L == 1, "slot mask is decode-only (L=1)"
    has_sink = sinks is not None
    if has_sink:
        sinks = sinks.reshape(B, H)
    else:
        sinks = mx.zeros((1, 1), rel.dtype)
    width = S + int(has_sink)
    return _get_mask_kernel()(
        inputs=[rel, proj, slot_pos, qpos, sinks, kref],
        template=[
            ("T", rel.dtype),
            ("D_REL", d_rel),
            ("REL_EXTENT", proj.shape[1]),
            ("WINDOW", int(window)),
            ("RING", bool(ring)),
            ("HAS_SINK", bool(has_sink)),
        ],
        grid=(width, 1, B * H),
        threadgroup=(min(256, width), 1, 1),
        output_shapes=[(B, H, 1, width)],
        output_dtypes=[rel.dtype],
    )[0]


def slot_mask(rel, proj, slot_pos, q_pos, window, sinks=None):
    """General physical-order mask (compile-friendly: positions are arrays).

    rel [B,1,H,d_rel], proj [d_rel,R], slot_pos [B,S] int32 (-1 = empty),
    q_pos [B] int32, window: int (0 = no cap), sinks: [B,H] / [B,1,H] or None.
    Semantics equal stock ``banded_additive_mask`` on the temporally ordered
    keys: dist = q_pos - slot_pos; -1e30 if empty, dist<0 or dist>=window;
    fp32 dot over d_rel rounded once to rel.dtype if dist < R; else 0.
    """
    B, S = slot_pos.shape
    return _run_mask(
        rel, proj, slot_pos, q_pos, slot_pos.reshape(B, 1, S), window, sinks, ring=False
    )


def ring_mask(rel, proj, idx, kref, window, sinks=None):
    """Mask for a ring buffer (keep=0) read in physical order, L=1.

    ``idx`` is an int32 array [1] or [B] holding the ring write index after the
    update (``RotatingKVCache._idx``); passing idx == S gives temporal order
    (plain ``KVCache``). ``kref`` is any array with shape[2] == S (e.g. keys);
    it is not read, so S stays out of the template (one pipeline per layer kind).
    """
    idx = mx.broadcast_to(idx.reshape(-1), (rel.shape[0],))
    return _run_mask(rel, proj, idx, idx, kref, window, sinks, ring=True)


# --------------------------------------------------------------------------
# 3. Glue: short convolution with the next conv state as a second output.
# --------------------------------------------------------------------------

_SCONV_SRC = r"""
    uint i = thread_position_in_grid.x;
    int B = x_shape[0], L = x_shape[1], C = x_shape[2];
    if (i >= B * L * C) return;
    int c = i % C, t = (i / C) % L, b = i / (C * L);
    float value = 0;
    for (int j = 0; j < W; ++j) {
        int p = t + j - W + 1;
        float xv = p < 0
            ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                          + c * state_strides[2]])
            : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]);
        float w = weight[c * weight_strides[0] + j * weight_strides[1]];
        value = value + xv * w;
    }
    out[i] = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                               + c * x_strides[2]])) / divisor);
    // concat([state, x], 1)[:, -(W-1):] written once per (b, c) by t == L-1.
    if (t == L - 1) {
        for (int s = 0; s < W - 1; ++s) {
            int r = L + s - (W - 1);
            nstate[(b * (W - 1) + s) * C + c] = r < 0
                ? float(state[b * state_strides[0] + (r + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : float(x[b * x_strides[0] + r * x_strides[1] + c * x_strides[2]]);
        }
    }
"""

_sconv_kernel = None
_divisors = {}


def sconv_divisor(width):
    d = _divisors.get(width)
    if d is None:
        d = _divisors[width] = mx.array(math.sqrt(width + 1), mx.float32)
    return d


def short_convolution_with_state(state, weight, x, width):
    """Unmasked stock ``short_convolution`` plus ``_next_conv_state``.

    Returns (out [B,L,C] x.dtype, next_state [B,width-1,C] float32).
    """
    global _sconv_kernel
    if _sconv_kernel is None:
        _sconv_kernel = mx.fast.metal_kernel(
            name="inkling2_sconv_with_state",
            input_names=["state", "weight", "x", "divisor"],
            output_names=["out", "nstate"],
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
            source=_SCONV_SRC,
        )
    B, L, C = x.shape
    out, nstate = _sconv_kernel(
        inputs=[state, weight, x, sconv_divisor(width)],
        template=[("T", x.dtype), ("W", width)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (B, width - 1, C)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out, nstate


# Compiled glue bodies. Same op sequence as stock, fused by mx.compile.


@mx.compile
def output_gate(out, gate):
    return (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(
        out.dtype
    )


def _message_controller(x, message, down_weight, forget_weight, gain_weight, eps):
    pair = mx.concatenate(
        [mx.fast.rms_norm(v, None, eps) for v in (x, message)], axis=-1
    )
    hidden = (pair.astype(down_weight.dtype) @ down_weight.T).astype(mx.float32)
    forget = mx.tanh(hidden @ forget_weight.T)
    gain = mx.tanh(hidden @ gain_weight.T)
    return (1 - forget) * x + gain * message


message_controller = mx.compile(_message_controller)


# log-n tau: identical for every global visit of one step (same offset).
_tau_memo = {}


def log_n_tau(offset, length, alpha, floor):
    key = (int(offset), int(length), float(alpha), float(floor))
    tau = _tau_memo.get(key)
    if tau is None:
        _tau_memo.clear()
        positions = mx.arange(length) + mx.array(offset)[..., None] + 1
        tau = 1 + alpha * mx.log(mx.maximum(positions / floor, 1))
        _tau_memo[key] = tau
    return tau


def global_scale_factor(linear):
    """Cached ``2688*scale`` (or ``2688/scale``) broadcast to num_experts."""
    scale = linear.global_scale
    cached = linear.__dict__.get("_fast_factor")
    if cached is not None and cached[0] is scale:
        return cached[1]
    s = scale.astype(mx.float32)
    factor = 2688.0 / s if linear._scale_name == "weight_global_scale" else 2688.0 * s
    factor = mx.broadcast_to(factor.reshape(-1), (linear.num_experts,))
    linear.__dict__["_fast_factor"] = (scale, factor)
    return factor


# --------------------------------------------------------------------------
# 4. Projection fusion helpers.
# --------------------------------------------------------------------------


def gemv_reduction_matches_fused(weight):
    """True if an M=1 ``x @ weight.T`` reduces over K in the same order as the
    same rows inside a wider fused weight (MLX 0.32 gemv dispatch).

    For a row-major [N, K] weight MLX picks gemv with 32 lanes over K; when
    K >= 16 * N (and K > 64) it switches to a split-K variant (bm=1, bn=8)
    whose partial sums combine in a different order. A wide fused GEMV never
    takes that branch, so such a member would not be bitwise vs stock.
    """
    n, k = weight.shape
    return k <= 64 or k < 16 * n


def fuse_linears(linears):
    """Concatenate nn.Linear weights (rows) once; each linear keeps a row view.

    Returns (fused_weight, splits). Requires plain bias-free nn.Linear.
    """
    weights = [l.weight for l in linears]
    fused = mx.concatenate(weights, axis=0)
    mx.eval(fused)
    splits, start = [], 0
    for l, w in zip(linears, weights):
        end = start + w.shape[0]
        l.weight = fused[start:end]
        splits.append(end)
        start = end
    mx.eval([l.weight for l in linears])
    return fused, splits[:-1]


def fuse_switch_gate_up(switch_mlp):
    """Fuse SwitchGLU gate/up quantized weights along the output axis.

    Returns a dict describing the fused projection, or None when the two
    projections are not fusable (different modes/scales or biases).
    """
    g, u = switch_mlp.gate_proj, switch_mlp.up_proj
    if not hasattr(g, "mode") or type(g) is not type(u):
        return None
    if (g.mode, g.bits, g.group_size) != (u.mode, u.bits, u.group_size):
        return None
    if g.get("biases") is not None or u.get("biases") is not None:
        return None
    if "bias" in g or "bias" in u or "pre_quant_scale" in g or "pre_quant_scale" in u:
        return None
    factor = None
    if hasattr(g, "global_scale"):
        if g._scale_name != u._scale_name:
            return None
        if g.global_scale.shape != u.global_scale.shape or not mx.array_equal(
            g.global_scale, u.global_scale
        ).item():
            return None
        if g.global_scale.size not in (1, g.num_experts):
            return None
        factor = global_scale_factor(g)
    n = g.weight.shape[-2]
    w = mx.concatenate([g.weight, u.weight], axis=-2)
    s = mx.concatenate([g.scales, u.scales], axis=-2)
    mx.eval(w, s)
    g.weight, u.weight = w[..., :n, :], w[..., n:, :]
    g.scales, u.scales = s[..., :n, :], s[..., n:, :]
    mx.eval(g.weight, u.weight, g.scales, u.scales)
    return {
        "weight": w,
        "scales": s,
        "n": n,
        "mode": g.mode,
        "bits": g.bits,
        "group_size": g.group_size,
        "factor": factor,
    }


def fused_gate_up(fused, x, idx, sorted_indices=False):
    """One gather_qmm for gate and up; returns (up, gate) like SwitchGLU."""
    kwargs = {}
    if fused["factor"] is not None:
        kwargs["global_scale"] = fused["factor"]
    out = mx.gather_qmm(
        x,
        fused["weight"],
        fused["scales"],
        rhs_indices=idx,
        transpose=True,
        group_size=fused["group_size"],
        bits=fused["bits"],
        mode=fused["mode"],
        sorted_indices=sorted_indices,
        **kwargs,
    )
    n = fused["n"]
    return out[..., n:], out[..., :n]


# --------------------------------------------------------------------------
# 5. route: sigmoid + bias + top-k + gather + normalise in one kernel.
# --------------------------------------------------------------------------
# Stock _route: scores = sigmoid(x.f32 @ W.f32.T); idx = argpartition(
# -(scores + bias), k-1)[..., :k]; w = take(scores, idx); w / (sum + 1e-20) * s.
# On Metal argpartition runs the stable merge sort, so idx is the first k of a
# stable ascending argsort of -(scores + bias): descending score+bias, ties ->
# lower expert first, NaN last. The k-sum is sequential (checked vs mx.sum).
# The logits GEMV stays the stock matmul (same primitive, bitwise).

_ROUTE_SRC = r"""
    // logits [R, E] f32, bias [E] f32 -> idx [R, K] uint32, w [R, K] f32.
    // One threadgroup per row, one thread per expert. Each expert's rank is
    // the number of experts that sort before it (stable ascending order of
    // key = -(score + bias), NaN last); ranks < K are the selection, in order.
    threadgroup float keys[E];
    threadgroup float sel_s[K];
    threadgroup uint sel_i[K];
    threadgroup float denom_tg[1];
    const uint t = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    float s = 0.0f, kt = 0.0f;
    if (t < (uint)E) {
        float x = logits[(size_t)row * E + t];
        float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        s = (x < 0) ? y : 1 - y;
        kt = -(s + bias[t]);
        keys[t] = kt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < (uint)E) {
        bool tn = isnan(kt);
        uint rank = 0;
        for (uint j = 0; j < (uint)E; ++j) {
            float kj = keys[j];
            bool jn = isnan(kj);
            bool before = tn ? (jn ? j < t : true)
                             : (!jn && (kj < kt || (kj == kt && j < t)));
            rank += before ? 1u : 0u;
        }
        if (rank < (uint)K) { sel_s[rank] = s; sel_i[rank] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float total = 0.0f;
        for (int k = 0; k < K; ++k) total = total + sel_s[k];
        denom_tg[0] = total + 1e-20f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < (uint)K) {
        idx[(size_t)row * K + t] = sel_i[t];
        w[(size_t)row * K + t] = sel_s[t] / denom_tg[0] * scale[0];
    }
"""

_route_kernel = None


def route_topk(logits, bias, top_k, route_scale):
    """Fused tail of stock ``_route`` given logits = x.f32 @ W.f32.T.

    logits [..., E] float32 -> (indices [..., K] uint32, weights [..., K] f32).
    """
    global _route_kernel
    if _route_kernel is None:
        _route_kernel = mx.fast.metal_kernel(
            name="inkling2_route_topk",
            input_names=["logits", "bias", "scale"],
            output_names=["idx", "w"],
            source=_ROUTE_SRC,
            compile_options={"math_mode": "safe"},
        )
    lead, E = logits.shape[:-1], logits.shape[-1]
    assert top_k <= E <= 1024
    R = 1
    for d in lead:
        R *= d
    tg = ((E + 31) // 32) * 32
    idx, w = _route_kernel(
        inputs=[logits.reshape(R, E), bias.astype(mx.float32), _scalar(route_scale)],
        template=[("E", E), ("K", int(top_k))],
        grid=(R * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(R, top_k), (R, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return idx.reshape(*lead, top_k), w.reshape(*lead, top_k)


_scalars = {}


def _scalar(v):
    a = _scalars.get(float(v))
    if a is None:
        a = _scalars[float(v)] = mx.array([float(v)], mx.float32)
    return a


def route(x, weight_f32, bias, top_k, route_scale):
    """Drop-in for stock ``_route`` (weight already FP32)."""
    logits = x.astype(mx.float32) @ weight_f32.T
    return route_topk(logits, bias, top_k, route_scale)


# --------------------------------------------------------------------------
# 6. moesort: token-sorted grouped expert matvecs for batched decode (L=1).
# --------------------------------------------------------------------------
# Routes are sorted by expert; the existing native-reduction grouped/paired
# kernels (models/masked_quantized.py) then read each expert's packed weights
# once per pair of routes instead of once per route. Projections the kernels
# do not support (e.g. K % 256 != 0, Inkling 2 Mini's down_proj K=1152) run
# through the stock gather_qmm on the sorted routes.

def _grouped_kind(linear, input_alignment):
    from mlx_vlm.models.masked_quantized import supports_masked_qmv, supports_paired_mxfp8_qmv

    if supports_masked_qmv(linear, input_alignment=input_alignment):
        return "nvfp4"
    if supports_paired_mxfp8_qmv(linear):
        return "mxfp8"
    return None


class _FusedGateUpView:
    """gate_proj stand-in exposing the contiguous fused [gate; up] buffer.

    INKLING_FAST=fusemoe leaves gate/up weights as strided views of one
    [E, 2n, K] buffer; mx.fast.metal_kernel copies non-row-contiguous inputs,
    so grouped kernels fed those views copied every expert's gate and up
    weights on every call (batch decode ~40% slower). Rows are computed
    independently, so running the kernel once on the fused buffer with
    N = 2n gives bitwise the same gate and up rows.
    """

    def __init__(self, gate, fused):
        self._gate = gate
        self.weight = fused["weight"]
        self.scales = fused["scales"]
        self.output_dims = 2 * fused["n"]

    def __getattr__(self, name):
        return getattr(self._gate, name)


def sorted_switch_glu(switch_mlp, x, indices, fused=None):
    """Same result as ``switch_mlp(x, indices)`` for x [B, 1, D], indices [B, 1, K].

    Returns [B, 1, K, D]. Uses expert-sorted routes and grouped kernels.
    ``fused``: the fusemoe state (gate/up are then views of its buffer).
    """
    from mlx_vlm.models.masked_quantized import grouped_masked_qmv, paired_masked_mxfp8_qmv

    B, L, D = x.shape
    K = indices.shape[-1]
    flat = indices.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat)
    inverse = mx.argsort(order)
    sel = flat[order]
    vals = x.reshape(-1, D)[order // K]

    def project(linear, v):
        kind = _grouped_kind(linear, 256)
        if kind == "nvfp4":
            return grouped_masked_qmv(v, linear, sel)
        if kind == "mxfp8" and sel.size % 2 == 0:
            return paired_masked_mxfp8_qmv(v, linear, sel)
        # stock gather_qmm, one route per row (same kernel stock uses)
        return linear(v[:, None, None], sel[:, None], sorted_indices=False).reshape(
            sel.size, -1
        )

    kind = _grouped_kind(switch_mlp.gate_proj, 256)
    if fused is not None and (kind == "nvfp4" or (kind == "mxfp8" and sel.size % 2 == 0)):
        view = _FusedGateUpView(switch_mlp.gate_proj, fused)
        if kind == "nvfp4":
            gu = grouped_masked_qmv(vals, view, sel)
        else:
            gu = paired_masked_mxfp8_qmv(vals, view, sel)
        n = fused["n"]
        gate, up = gu[..., :n], gu[..., n:]
    else:
        up = project(switch_mlp.up_proj, vals)
        gate = project(switch_mlp.gate_proj, vals)
    out = project(switch_mlp.down_proj, switch_mlp.activation(up, gate))
    return out[inverse].reshape(B, L, K, -1)


def moesort_supported(switch_mlp):
    return all(
        _grouped_kind(getattr(switch_mlp, n), 256) is not None
        for n in ("gate_proj", "up_proj")
    )


# --------------------------------------------------------------------------
# 7. ctrlbatch: attn/k/v MessageControllers as one batched computation.
# --------------------------------------------------------------------------
# Same op sequence as MessageController.__call__ on stacked [3, ...] weights
# (identical to fast_decode._controllers3). Bitwise vs three separate
# controllers verified up to M = B*L = 8; callers gate on M <= 8.

CTRL_NAMES = ("attn", "k", "v")
CTRL_WEIGHTS = ("down_weight", "forget_weight", "gain_weight")


def stacked_controllers(layer):
    """layer.__dict__['_fd_ctrl3'] (shared with FastDecoder): stacked weights.

    Built once; the three controllers' weights become views of the stacks
    (memory +0). Rebuilt if any controller weight was replaced since.
    """
    cs = [layer.controllers[n] for n in CTRL_NAMES]
    stk = layer.__dict__.get("_fd_ctrl3")
    if stk is not None:
        views = layer.__dict__.get("_fd_ctrl3_views")
        current = tuple(getattr(c, w) for w in CTRL_WEIGHTS for c in cs)
        if views is None or all(a is b for a, b in zip(views, current)):
            return stk
    stk = {}
    for wname in CTRL_WEIGHTS:
        w = mx.stack([getattr(c, wname) for c in cs])
        mx.eval(w)
        for i, c in enumerate(cs):
            setattr(c, wname, w[i])
        stk[wname] = w
    layer.__dict__["_fd_ctrl3"] = stk
    layer.__dict__["_fd_ctrl3_views"] = tuple(
        getattr(c, w) for w in CTRL_WEIGHTS for c in cs
    )
    return stk


def _controllers3(x3, m3, dw, fw, gw, eps):
    pair = mx.concatenate([mx.fast.rms_norm(v, None, eps) for v in (x3, m3)], axis=-1)
    shape = pair.shape
    pair = pair.reshape(shape[0], -1, shape[-1])
    hidden = (pair.astype(dw.dtype) @ dw.swapaxes(-1, -2)).astype(mx.float32)
    forget = mx.tanh(hidden @ fw.swapaxes(-1, -2)).reshape(x3.shape)
    gain = mx.tanh(hidden @ gw.swapaxes(-1, -2)).reshape(x3.shape)
    return (1 - forget) * x3 + gain * m3


controllers3 = _controllers3
controllers3_compiled = mx.compile(_controllers3)


# --------------------------------------------------------------------------
# 8. xpass: exact cross-pass reuse in AttentionResidual (stock path).
# --------------------------------------------------------------------------
# (a) rms_norm(block) is computed once, when the block is appended (row-wise
#     op: bitwise vs normalising the stacked states every call).
# (b) Depth logits keys @ query.T for blocks that are identical across loop
#     passes (prelude blocks + the prelude partial appended at core_start)
#     are computed once per step per AttentionResidual and reused by passes
#     2..N. Bitwise caveat: MLX picks a different matmul kernel when a piece
#     has a single row (M=1), so such pieces are padded to two rows; with that
#     every split reproduces the full matmul's rows exactly for B*L <= 24
#     (tested); larger M (prefill) tiles differently, so reuse is gated to
#     B*L <= 16.


def _block_key(x):
    return mx.fast.rms_norm(x.astype(mx.float32)[..., None, :], None, 1e-6)


class KeyedBlocks(list):
    """List of block states that also keeps their rms_norm keys.

    ``memo`` (shared by copies, one dict per forward call) holds prefix depth
    logits per AttentionResidual; ``stable`` is how many leading blocks are
    identical across loop passes (set by TextModel; 0 = no logit reuse).
    """

    def __init__(self, items=(), keys=(), memo=None, stable=0):
        super().__init__(items)
        self.keys = list(keys)
        self.memo = {} if memo is None else memo
        self.stable = stable

    def append(self, x):
        super().append(x)
        self.keys.append(_block_key(x))

    def copy(self):
        return KeyedBlocks(self, self.keys, self.memo, self.stable)


def _rows_scores(keys, query):
    """(keys @ query.T) * D**-0.5 with the multi-row matmul kernel (M >= 2)."""
    D = keys.shape[-1]
    qT = query.reshape(-1, D).T
    if keys.shape[-2] == 1:
        two = mx.concatenate([keys, keys], axis=-2)
        return ((two @ qT) * D**-0.5)[..., :1, :]
    return (keys @ qT) * D**-0.5


def _attn_res_from_keys(keys_list, partial, query):
    keys = mx.concatenate([*keys_list, _block_key(partial)], axis=-2)
    D = keys.shape[-1]
    scores = (keys @ query.reshape(-1, D).T) * D**-0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ keys


def _attn_res_prefix(prefix_scores, keys_list, rest_keys, partial, query):
    rest = mx.concatenate([*rest_keys, _block_key(partial)], axis=-2)
    scores = mx.concatenate([prefix_scores, _rows_scores(rest, query)], axis=-2)
    keys = mx.concatenate([*keys_list, rest], axis=-2)
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ keys


attn_res_from_keys = mx.compile(_attn_res_from_keys)
attn_res_prefix = mx.compile(_attn_res_prefix)


def depth_scores(keys_list, query):
    """Prefix depth logits for blocks reused across passes (FastDecoder API)."""
    return _rows_scores(mx.concatenate(keys_list, axis=-2), query)


def attention_residual_keyed(module, blocks, partial, reuse_logits=True):
    """AttentionResidual.__call__ for KeyedBlocks (normalize_values=True)."""
    rows = partial.size // partial.shape[-1]
    # Split-matmul rows match the full matmul bitwise for B*L <= 24 on M4 Pro
    # (not at >= 32: other GEMM tiling), so reuse only for decode/verify sizes.
    P = min(blocks.stable, len(blocks)) if reuse_logits and rows <= 16 else 0
    if P == 0:
        return attn_res_from_keys(blocks.keys, partial, module.query)
    key = id(module)
    entry = blocks.memo.get(key)
    prefix = tuple(blocks[:P])
    if entry is not None and len(entry[0]) == P and all(
        a is b for a, b in zip(entry[0], prefix)
    ):
        scores = entry[1]
    else:
        scores = depth_scores(blocks.keys[:P], module.query)
        blocks.memo[key] = (prefix, scores)
    return attn_res_prefix(scores, blocks.keys[:P], blocks.keys[P:], partial, module.query)


# --------------------------------------------------------------------------
# 9. msgpack: one pass's cross-loop messages for all core layers at once.
# --------------------------------------------------------------------------
# Idea from mw00/project-maya PR #1095 (MIT): "a pass's messages for all
# twelve core layers are one launch (the bank does not change within a
# pass)". Re-implemented for MLX here (no code copied).


def packed_messages(queries, bank):
    """Stock per-layer ``softmax(q_i @ bank^T * D**-0.5) @ bank`` for every
    core layer i in one batched computation.

    queries: [n_core, 4, D]; bank: [..., nb, D]. Returns [..., n_core, 4, D].
    The batched matmuls use the same kernels and per-row reduction as the
    per-layer [4, D] ones (verified bitwise at B=1/4, nb=2/3, compiled or
    not, on the M5 Ultra; tests_fast/test_port.py).
    """
    b = bank[..., None, :, :]
    scores = (queries @ b.swapaxes(-1, -2)) * bank.shape[-1] ** -0.5
    return mx.softmax(scores, axis=-1) @ b


# --------------------------------------------------------------------------
# 10. qkvclip: l2_clip(q), l2_clip(k), rms_norm(v), l2_clip(r) and the
#     log-n tau scaling of q and r in ONE kernel per attention call.
# --------------------------------------------------------------------------
# Idea from mw00/project-maya PR #1095 (MIT): "its three norms one", "the
# sinks the clips' launch" (one launch for the per-head norms/clips after the
# projections). Re-implemented for MLX/Metal here; the arithmetic reproduces
# the MLX 0.32.4 ops it replaces (ml-explore/mlx, MIT):
# * l2_clip's Sum over the last axis: head_dim 128 rows go through
#   row_reduce_looped / row_reduce_simple (32 lanes x 4 consecutive elements,
#   summed serially from 0, then simd_sum); d_rel <= 64 rows through
#   row_reduce_small (one thread sums the row serially from 0). The squares
#   are a separate Square kernel in stock, so they are rounded to FP32 before
#   the sum (no fma: see _sq below).
# * rms_norm(v) (axis 128): rms_single_row, 32 lanes x 4, acc += x * x,
#   simd_sum, precise::rsqrt(acc / 128 + eps).
# * the clip tail: precise::sqrt, + 1e-6, threshold / ., min(., 1), x * .
# Each simdgroup owns one head row of one token.

_QKV_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const int row = (int)threadgroup_position_in_grid.y;
    const int tok = (int)threadgroup_position_in_grid.z;
    const int L = q_shape[1];
    const int b = tok / L, l = tok % L;
    const float tau = HAS_TAU ? tau_in[tok * tau_in_strides[0]] : 1.0f;
    if (row < HQ + HK) {
        // l2_clip of a head_dim row (q with q_scale, or k)
        const bool isq = row < HQ;
        const int h = isq ? row : row - HQ;
        float xs[4];
        float part = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int c = h * HD + (int)lane * 4 + i;
            float xv;
            if (isq) {
                xv = float(q[b * q_strides[0] + l * q_strides[1] + c * q_strides[2]])
                     * float(q_scale[h * q_scale_strides[0] + ((int)lane * 4 + i) * q_scale_strides[1]]);
            } else {
                xv = float(k[b * k_strides[0] + l * k_strides[1] + c * k_strides[2]]);
            }
            xs[i] = xv;
            part = _sq(xv) + part;
        }
        const float sum = simd_sum(part);
        float f = thr[0] / (metal::precise::sqrt(sum) + 1e-6f);
        f = metal::min(f, 1.0f);
        for (int i = 0; i < 4; ++i) {
            T o = T(xs[i] * f);
            if (isq && HAS_TAU) o = T(float(o) * tau);
            if (ZROW && !isq) {
                // [1, HK, 2, HD]: row 0 = k, row 1 = the zero sink row.
                const int zi = (h * 2) * HD + (int)lane * 4 + i;
                ko[zi] = o;
                ko[zi + HD] = T(0);
            } else {
                const int idx = (tok * (isq ? HQ : HK) + h) * HD + (int)lane * 4 + i;
                if (isq) qo[idx] = o; else ko[idx] = o;
            }
        }
    } else if (row < HQ + 2 * HK) {
        // v: rms_norm(v.astype(f32), None, eps).astype(T)  (or a plain copy)
        const int h = row - HQ - HK;
        float xv[4];
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int c = h * HD + (int)lane * 4 + i;
            xv[i] = float(v[b * v_strides[0] + l * v_strides[1] + c * v_strides[2]]);
            acc += xv[i] * xv[i];
        }
        float inv = 1.0f;
        if (VNORM) {
            acc = simd_sum(acc);
            inv = metal::precise::rsqrt(acc / float(HD) + eps[0]);
        }
        for (int i = 0; i < 4; ++i) {
            const T o = VNORM ? T(1.0f * float(xv[i] * inv)) : T(xv[i]);
            if (ZROW) {
                const int zi = (h * 2) * HD + (int)lane * 4 + i;
                vo[zi] = o;
                vo[zi + HD] = T(0);
            } else {
                vo[(tok * HK + h) * HD + (int)lane * 4 + i] = o;
            }
        }
    } else {
        // l2_clip of a d_rel row (r with rel_scale): serial sum like
        // row_reduce_small; lanes < DR write.
        const int h = row - HQ - 2 * HK;
        float sum = 0.0f;
        for (int j = 0; j < DR; ++j) {
            const int c = h * DR + j;
            const float xv = float(r[b * r_strides[0] + l * r_strides[1] + c * r_strides[2]])
                             * float(rel_scale[h * rel_scale_strides[0] + j * rel_scale_strides[1]]);
            sum = _sq(xv) + sum;
        }
        float f = rthr[0] / (metal::precise::sqrt(sum) + 1e-6f);
        f = metal::min(f, 1.0f);
        if ((int)lane < DR) {
            const int c = h * DR + (int)lane;
            const float xv = float(r[b * r_strides[0] + l * r_strides[1] + c * r_strides[2]])
                             * float(rel_scale[h * rel_scale_strides[0] + (int)lane * rel_scale_strides[1]]);
            T o = T(xv * f);
            if (HAS_TAU) o = T(float(o) * tau);
            ro[(tok * HQ + h) * DR + (int)lane] = o;
        }
    }
"""

_QKV_HEADER = r"""
// x * x rounded to FP32 on its own (stock squares in a separate kernel, so
// the square must not be contracted into the following add):
// fma(x, x, -0) == round(x * x) exactly.
inline float _sq(float x) {
    return metal::fma(x, x, -0.0f);
}
"""

_qkv_kernel = None


def qkv_clip_supported(attn, cfg, x):
    return (
        _on_gpu()
        and attn.head_dim == 128
        and cfg.d_rel <= 32
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
    )


def qkv_clip(q, k, v, r, attn, cfg, tau=None, zrow=False):
    """One kernel for the post-projection per-head ops of Attention.

    q, r: [B, L, Hq*HD], [B, L, Hq*DR]; k, v: [B, L, Hk*HD] (views allowed,
    last stride 1). tau: None or FP32 broadcastable to [B*L].
    Returns q [B, L, Hq, HD], k, v [B, L, Hk, HD], r [B, L, Hq, DR] in q's
    dtype, bitwise equal to the stock op sequence.
    """
    global _qkv_kernel
    if _qkv_kernel is None:
        _qkv_kernel = mx.fast.metal_kernel(
            name="inkling2_qkv_clip",
            input_names=["q", "k", "v", "r", "q_scale", "rel_scale", "thr", "rthr",
                         "eps", "tau_in"],
            output_names=["qo", "ko", "vo", "ro"],
            header=_QKV_HEADER,
            source=_QKV_SRC,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    B, L = q.shape[0], q.shape[1]
    hq, hk, hd, dr = attn.n_heads, attn.n_kv, attn.head_dim, cfg.d_rel
    dtype = q.dtype
    has_tau = tau is not None
    # 1-D, any stride (a broadcast view is read with stride 0).
    tau_in = mx.broadcast_to(tau.reshape(-1), (B * L,)) if has_tau else _scalar(1.0)
    qo, ko, vo, ro = _qkv_kernel(
        inputs=[q, k, v, r, attn.q_scale, attn.rel_scale,
                _scalar(cfg.qk_l2_clip_threshold), _scalar(cfg.rel_l2_clip_threshold),
                _scalar(cfg.rms_norm_eps), tau_in],
        template=[("T", dtype), ("HQ", hq), ("HK", hk), ("HD", hd), ("DR", dr),
                  ("HAS_TAU", has_tau), ("VNORM", bool(cfg.use_value_norm_no_gamma)),
                  ("ZROW", bool(zrow))],
        grid=(32, hq + 2 * hk + hq, B * L),
        threadgroup=(32, 1, 1),
        # zrow (INKLING_FAST=mega_kvz, B*L == 1): k and v come out head-major
        # as [1, HK, 2, HD] with a zero second row (the exact path's sink slot),
        # ready for one slice_update each at slot p.
        output_shapes=[(B, L, hq, hd)]
        + ([(1, hk, 2, hd)] * 2 if zrow else [(B, L, hk, hd)] * 2)
        + [(B, L, hq, dr)],
        output_dtypes=[dtype] * 4,
    )
    return qo, ko, vo, ro


# --------------------------------------------------------------------------
# 11. xmask: FastDecoder exact-operand mask in one kernel.
# --------------------------------------------------------------------------
# Idea from exo-explore/exo PR #2530 (Apache-2.0): one kernel writes each
# attention call's additive mask and works the causal / window / padding
# masks out from the key distances (no arange / where / select chains).
# Re-implemented for our exact decode operands (FastDecoder._exact_operands);
# values are bit-identical to slot_mask + the sink select it replaces.
# GATHER=0: column j holds position j (j <= p valid), the sink at j == p + 1,
#           columns > p + 1 masked.
# GATHER=1: W + 1 temporal columns, oldest first: S = min(p + 1, W) valid
#           columns from position max(p - W + 1, 0), the sink at column S.

_XMASK_SRC = r"""
    const int B = rel_shape[0];
    const int H = rel_shape[2];
    const int N = kref_shape[2];  // unread shape carrier (columns)
    const int j = (int)thread_position_in_grid.x;
    const int bh = (int)thread_position_in_grid.z;
    if (j >= N || bh >= B * H) return;
    const int b = bh / H, h = bh % H;
    const size_t o = (size_t)bh * N + j;
    const int p = qpos[b * qpos_strides[0]];
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
        val = (T)sinks[(size_t)b * sinks_strides[0] + (size_t)h * sinks_strides[1]];
    } else if (!valid || dist < 0 || (WINDOW > 0 && dist >= WINDOW)) {
        val = (T)(-1e30f);
    } else if (dist < REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0] + (size_t)h * rel_strides[2];
        for (int d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0] + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[o] = val;
"""

_xmask_kernel = None


def exact_mask(rel, proj, q_pos, sinks, kref, window, gather):
    """[B, H, 1, N] mask for FastDecoder's exact decode operands (L=1).

    rel [B,1,H,d_rel]; proj [d_rel,R]; q_pos [B] int32; sinks [B,1,H];
    kref: any array with shape[2] == N (not read); window: sliding window
    (0 = global); gather: False = sliced buffer (column j = position j),
    True = temporal gather of W + 1 rows.
    """
    global _xmask_kernel
    if _xmask_kernel is None:
        _xmask_kernel = mx.fast.metal_kernel(
            name="inkling2_exact_mask",
            input_names=["rel", "proj", "qpos", "sinks", "kref"],
            output_names=["out"],
            source=_XMASK_SRC,
            ensure_row_contiguous=False,
        )
    N = kref.shape[2]
    B, L, H, d_rel = rel.shape
    assert L == 1
    return _xmask_kernel(
        inputs=[rel, proj, q_pos, sinks.reshape(B, H), kref],
        template=[
            ("T", rel.dtype),
            ("D_REL", d_rel),
            ("REL_EXTENT", proj.shape[1]),
            ("WINDOW", int(window)),
            ("GATHER", bool(gather)),
        ],
        grid=(N, 1, B * H),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(B, H, 1, N)],
        output_dtypes=[rel.dtype],
    )[0]


# --------------------------------------------------------------------------
# 12. sconvtail: the decoder layer's short convolutions with the residual
#     adds folded in, and (MLP side) the routed-expert combine folded in.
# --------------------------------------------------------------------------
# Ideas from exo-explore/exo PR #2530 (Apache-2.0): "one kernel computes the
# float32 depthwise convolution, its residual, the decoder layer's
# residual-stream add and the next convolution state" and "expert combine:
# one kernel weights and sums the routed experts' outputs in float32 in the
# order MLX's column reduction adds a few rows"; jundot/omlx PR #2463
# (Apache-2.0): fold the top-k weighted sum into the routed tail instead of a
# separate combine. Re-implemented for Inkling 2's ops (L=1 decode):
# MODE 0 (attn side): res = f32(sconv(a)) [+ f32(x)]          -> f32
# MODE 1 (mlp side):  h = bf16(sum_k round(f32(e_k) * w_k))   (stock
#                     _combine_routes: products in their own kernel, then
#                     col_reduce_small adds the 8 rows serially from 0)
#                     out = T(res + f32(sconv(h)))             -> T
# The convolution arithmetic is the glue kernel's (_SCONV_SRC), which is
# bitwise vs stock.

_STAIL_SRC = r"""
    uint i = thread_position_in_grid.x;
    const int B = state_shape[0], C = state_shape[2];
    if ((int)i >= B * C) return;
    const int c = i % C, b = i / C;
    float hv;
    if (MODE == 1) {
        float total = 0.0f;
        for (int k = 0; k < TOPK; ++k) {
            const float e = float(xin[b * xin_strides[0] + k * xin_strides[2] + c * xin_strides[3]]);
            const float wk = rw[b * rw_strides[0] + k * rw_strides[2]];
            total = metal::fma(e, wk, -0.0f) + total;
        }
        hv = float(T(total));
    } else {
        hv = float(xin[b * xin_strides[0] + c * xin_strides[2]]);
    }
    float value = 0;
    for (int j = 0; j < W; ++j) {
        float xv = j < W - 1
            ? float(state[b * state_strides[0] + j * state_strides[1] + c * state_strides[2]])
            : hv;
        float w = weight[c * weight_strides[0] + j * weight_strides[1]];
        value = value + xv * w;
    }
    const T s = T((value + hv) / divisor);
    for (int j = 0; j < W - 1; ++j) {
        nstate[(b * (W - 1) + j) * C + c] = j < W - 2
            ? float(state[b * state_strides[0] + (j + 1) * state_strides[1] + c * state_strides[2]])
            : hv;
    }
    const float r = float(skip[b * skip_strides[0] + c * skip_strides[2]]);
    if (MODE == 1) {
        out[b * C + c] = OT(T(r + float(s)));
    } else {
        out[b * C + c] = OT(SKIP ? float(s) + r : float(s));
    }
"""

_stail_kernel = None


def sconv_tail(module, state, xin, skip=None, route_weights=None):
    """MODE 0: xin = attention output [B,1,C] (T); skip = layer input x [B,1,C]
    (T) or None (block boundary); returns (residual f32 [B,1,C], next state).
    MODE 1 (route_weights given): xin = expert outputs [B,1,K,C] (T),
    route_weights [B,1,K] f32, skip = residual [B,1,C] f32; returns
    (layer output T [B,1,C], next state). state: [B, W-1, C] f32."""
    global _stail_kernel
    if _stail_kernel is None:
        _stail_kernel = mx.fast.metal_kernel(
            name="inkling2_sconv_tail",
            input_names=["state", "weight", "xin", "rw", "skip", "divisor"],
            output_names=["out", "nstate"],
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
            source=_STAIL_SRC,
        )
    width = module.kernel_size
    B, C = state.shape[0], state.shape[2]
    mode = 1 if route_weights is not None else 0
    T = xin.dtype
    out, nstate = _stail_kernel(
        inputs=[state, module.conv.weight, xin,
                route_weights if mode else _scalar(0.0),
                skip if skip is not None else xin,
                sconv_divisor(width)],
        template=[("T", T), ("OT", T if mode else mx.float32), ("W", width),
                  ("MODE", mode), ("SKIP", skip is not None),
                  ("TOPK", xin.shape[2] if mode else 1)],
        grid=(B * C, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, 1, C), (B, width - 1, C)],
        output_dtypes=[T if mode else mx.float32, mx.float32],
    )
    return out, nstate


# --------------------------------------------------------------------------
# 13. xmask (gather regime): temporal K/V gather of the sliding ring in one
#     kernel (index math inline; replaces arange/min/max/where/% + 2 takes).
# --------------------------------------------------------------------------
_RGATHER_SRC = r"""
    // grid: (D, W + 1, B * NKV * 2); out row t = ring slot of position
    // base + t for t < S, else the permanent zero slot R.
    const int d = (int)thread_position_in_grid.x;
    const int t = (int)thread_position_in_grid.y;
    const int z = (int)thread_position_in_grid.z;
    const int D = kb_shape[3], NKV = kb_shape[1];
    if (d >= D || t > WINDOW) return;
    const int which = z % 2, bh = z / 2;
    const int b = bh / NKV, h = bh % NKV;
    const int p = qpos[b * qpos_strides[0]];
    const int S = metal::min(p + 1, WINDOW);
    const int base = metal::max(p - WINDOW + 1, 0);
    const int slot = t < S ? (base + t) % RING : RING;
    const size_t o = (((size_t)b * NKV + h) * (WINDOW + 1) + t) * D + d;
    if (which == 0)
        ko[o] = kb[b * kb_strides[0] + h * kb_strides[1] + slot * kb_strides[2] + d * kb_strides[3]];
    else
        vo[o] = vb[b * vb_strides[0] + h * vb_strides[1] + slot * vb_strides[2] + d * vb_strides[3]];
"""

_rgather_kernel = None

# mega_rg2 (megaperf): same gather, 4 elements (8 bytes, one uint2) per thread
# when the buffers are innermost-contiguous; the 1-element version ran at
# ~200 GB/s (21 us per call x 40 sliding visits at 4k ctx). Pure copy: bitwise.
_RGATHER2_SRC = r"""
    const int d4 = (int)thread_position_in_grid.x;
    const int t = (int)thread_position_in_grid.y;
    const int z = (int)thread_position_in_grid.z;
    const int D = kb_shape[3], NKV = kb_shape[1];
    if (d4 * 4 >= D || t > WINDOW) return;
    const int which = z % 2, bh = z / 2;
    const int b = bh / NKV, h = bh % NKV;
    const int p = qpos[b * qpos_strides[0]];
    const int S = metal::min(p + 1, WINDOW);
    const int base = metal::max(p - WINDOW + 1, 0);
    const int slot = t < S ? (base + t) % RING : RING;
    const size_t o = (((size_t)b * NKV + h) * (WINDOW + 1) + t) * D;
    if (which == 0) {
        const device T* src = kb + b * kb_strides[0] + h * kb_strides[1] + slot * kb_strides[2];
        // the uint2 path needs 8-byte aligned rows (base
        // pointer incl. the array offset, and every outer stride % 4 == 0);
        // anything else takes the scalar path (same values).
        if (kb_strides[3] == 1 && (kb_strides[0] % 4) == 0 && (kb_strides[1] % 4) == 0
            && (kb_strides[2] % 4) == 0 && (((ulong)kb) & 7) == 0) {
            ((device uint2*)(ko + o))[d4] = ((const device uint2*)src)[d4];
        } else {
            for (int i = 0; i < 4; ++i) ko[o + d4 * 4 + i] = src[(d4 * 4 + i) * kb_strides[3]];
        }
    } else {
        const device T* src = vb + b * vb_strides[0] + h * vb_strides[1] + slot * vb_strides[2];
        // the uint2 path needs 8-byte aligned rows (base
        // pointer incl. the array offset, and every outer stride % 4 == 0);
        // anything else takes the scalar path (same values).
        if (vb_strides[3] == 1 && (vb_strides[0] % 4) == 0 && (vb_strides[1] % 4) == 0
            && (vb_strides[2] % 4) == 0 && (((ulong)vb) & 7) == 0) {
            ((device uint2*)(vo + o))[d4] = ((const device uint2*)src)[d4];
        } else {
            for (int i = 0; i < 4; ++i) vo[o + d4 * 4 + i] = src[(d4 * 4 + i) * vb_strides[3]];
        }
    }
"""
_rgather2_kernel = None


def ring_gather(kb, vb, q_pos, window, ring):
    """[B, n_kv, W + 1, D] temporal K and V (oldest first, zero rows after
    the valid ones) from the ring buffers [B, n_kv, ring + 1, D]."""
    global _rgather_kernel, _rgather2_kernel
    B, n, _, D = kb.shape
    if enabled("mega_rg2") and D % 4 == 0 and kb.dtype.size == 2 and vb.dtype == kb.dtype:
        if _rgather2_kernel is None:
            _rgather2_kernel = mx.fast.metal_kernel(
                name="inkling2_ring_gather2",
                input_names=["kb", "vb", "qpos"],
                output_names=["ko", "vo"],
                source=_RGATHER2_SRC,
                ensure_row_contiguous=False,
            )
        shape = (B, n, window + 1, D)
        return _rgather2_kernel(
            inputs=[kb, vb, q_pos],
            template=[("T", kb.dtype), ("WINDOW", int(window)), ("RING", int(ring))],
            grid=(D // 4, window + 1, B * n * 2),
            threadgroup=(D // 4, max(1, 256 // (D // 4)), 1),
            output_shapes=[shape, shape],
            output_dtypes=[kb.dtype, vb.dtype],
        )
    if _rgather_kernel is None:
        _rgather_kernel = mx.fast.metal_kernel(
            name="inkling2_ring_gather",
            input_names=["kb", "vb", "qpos"],
            output_names=["ko", "vo"],
            source=_RGATHER_SRC,
            ensure_row_contiguous=False,
        )
    B, n, _, D = kb.shape
    shape = (B, n, window + 1, D)
    return _rgather_kernel(
        inputs=[kb, vb, q_pos],
        template=[("T", kb.dtype), ("WINDOW", int(window)), ("RING", int(ring))],
        grid=(D, window + 1, B * n * 2),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[kb.dtype, vb.dtype],
    )


# --------------------------------------------------------------------------
# 14. attnres: one depth-attention (AttentionResidual, one query,
#     normalize_values) per kernel launch, keys read in place (no concat).
# --------------------------------------------------------------------------
# Idea from mw00/project-maya PR #1095 (MIT): "the attention residuals ask
# for four states at once" -- the depth attention as one launch instead of a
# chain of small ops. Re-implemented for MLX; it reproduces, bit for bit, the
# MLX 0.32.4 kernels (ml-explore/mlx, MIT) that stock dispatches for
# _attn_res_keys / attn_res_prefix at B*L <= 16:
# * rms_norm(partial) over D: rms_single_row (D/4 threads x 4, acc += x*x,
#   simd_sum, zero-padded threadgroup simd_sum, precise::rsqrt(acc/D + eps));
# * scores = keys @ q.T (q cast to f32): gemv bm1 bn8 sm1 sn32 tm1 tn4 per
#   key row (8 simdgroups split K in 4-wide chunks, shuffle_down 16..1, then
#   simdgroup 0 adds the others in order), times D**-0.5;
# * softmax over the key axis: serial max, precise::exp(s - max), serial sum
#   from 0, divide;
# * weights @ keys: gemv_t sm8 sn4 tm4 tn4: rows summed serially in groups of
#   4 (g0 rows 0-3, g1 rows 4-7, ...), then ((g0+g4)+(g2+g6))+((g1+g5)+(g3+g7)).

def _attnres_source(nk):
    loads = "\n".join(
        f"        if (r == {i}) return k{i}[(size_t)tok * k{i}_strides[2] * 0 + "
        f"(size_t)b * k{i}_strides[0] + (size_t)l * k{i}_strides[1] + (size_t)d * k{i}_strides[3]];"
        for i in range(nk)
    )
    return r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int tok = (int)threadgroup_position_in_grid.y;
    const int L = partial_shape[1];
    const int b = tok / L, l = tok % L;
    constexpr int N = NK + 1;
    threadgroup float pk[D];
    threadgroup float red[32];
    threadgroup float sc[N];
    threadgroup float wts[N];
    auto key = [&](int r, int d) -> float {
""" + loads + r"""
        return pk[d];
    };
    // 1. rms_norm(partial): rms_single_row layout, D/4 threads.
    {
        float acc = 0;
        float tx[4];
        const int n0 = (int)lid * 4;
        for (int i = 0; i < 4; i++) {
            tx[i] = partial[(size_t)b * partial_strides[0] + (size_t)l * partial_strides[1]
                            + (size_t)(n0 + i) * partial_strides[2]];
            acc += tx[i] * tx[i];
        }
        acc = simd_sum(acc);
        if (sg == 0) red[lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = 0;
        if (sg == 0) {
            acc = simd_sum(red[lane]);
            if (lane == 0) red[0] = metal::precise::rsqrt(acc / D + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv = red[0];
        for (int i = 0; i < 4; i++) pk[n0 + i] = 1.0f * static_cast<float>(tx[i] * inv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 2. scores: gemv (bn8, sn32, tn4) per key row, simdgroups 0..7.
    threadgroup float part[8];
    for (int r = 0; r < N; ++r) {
        if (sg < 8) {
            float result = 0;
            int bn = ((int)sg * 32 + (int)lane) * 4;
            for (int i = 0; i < D / 1024; ++i) {
                float vc[4], it[4];
                for (int t = 0; t < 4; t++) vc[t] = float(query[(bn + t) * query_strides[0]]);
                for (int t = 0; t < 4; t++) it[t] = key(r, bn + t);
                for (int t = 0; t < 4; t++) result += it[t] * vc[t];
                bn += 1024;
            }
            if (D % 1024 > 0) {
                float vc[4], it[4];
                if (bn + 4 <= D) {
                    for (int t = 0; t < 4; t++) vc[t] = float(query[(bn + t) * query_strides[0]]);
                    for (int t = 0; t < 4; t++) it[t] = key(r, bn + t);
                } else {
                    for (int t = 0; t < 4; t++) {
                        vc[t] = bn + t < D ? float(query[(bn + t) * query_strides[0]]) : 0.0f;
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
    // 3. softmax over the N keys (serial, as the decomposed reduce ops).
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
    // 4. weights @ keys: gemv_t (sm8 sn4 tm4) row grouping.
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
        out[(size_t)tok * D + d] = a + c;
    }
"""


# --------------------------------------------------------------------------
# mega_dpar (megaperf): depth-attention scores parallel over keys.
# The attnres / mega_post / mega_pre kernels computed each key's score with
# 8 simdgroups, then two threadgroup barriers, key after key (2N barriers and
# N serialised load round-trips on one threadgroup). Here simdgroup s scores
# (key r, virtual simdgroup v) pairs round-robin: each pair is lane-for-lane
# virtual simdgroup v of key r's gemv (bn8, sn32, tn4) with the same per-lane
# code and shuffle_down tree; lane r then sums its 8 partials in order, so
# every score is bitwise the same. (v1, one simdgroup per key doing all 8
# virtual simdgroups, was SLOWER at the real 2-6 keys: KERNEL_PROFILE.md.) Softmax: max is order-free (simd_max), the exps
# run one per lane, z is still summed serially from 0, then e/z per lane.
# --------------------------------------------------------------------------
def _dpar(src):
    """Rewrite the serial scores + softmax stages of a depth-attention source."""
    lines = src.split("\n")
    def find(pred, start=0):
        for i in range(start, len(lines)):
            if pred(lines[i]):
                return i
        raise ValueError("mega_dpar: marker not found")
    s0 = find(lambda l: l.strip().startswith("//") and "scores: gemv" in l)
    s1 = find(lambda l: l.strip().startswith("//") and "softmax over the N keys" in l, s0)
    s2 = find(lambda l: l.strip().startswith("//") and "weights @ keys" in l, s1)
    blk = lines[s0:s1]
    b0 = next(i for i, l in enumerate(blk) if l.strip() == "if (sg < 8) {")
    b1 = next(i for i, l in enumerate(blk) if "simd_shuffle_down(result, s)" in l)
    body = blk[b0 + 1:b1]
    assert any("((int)sg * 32 + (int)lane) * 4" in l for l in body)
    body = [l.replace("((int)sg * 32 + (int)lane) * 4", "(v * 32 + (int)lane) * 4") for l in body]
    new_scores = [
        "    // scores, parallel over (key, virtual simdgroup) pairs (mega_dpar; bitwise,",
        "    // see fast_kernels._dpar): pair (r, v) is lane-for-lane virtual simdgroup v",
        "    // of key r's gemv; then lane r sums its 8 partials in order.",
        "    threadgroup float pv2[N * 8];",
        "    for (int pr = (int)sg; pr < N * 8; pr += TG / 32) {",
        "        const int r = pr / 8, v = pr % 8;",
        "        {",
        *body,
        "            for (ushort s = 16; s >= 1; s >>= 1) result += simd_shuffle_down(result, s);",
        "            if (lane == 0) pv2[pr] = result;",
        "        }",
        "    }",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    if (sg == 0 && (int)lane < N) {",
        "        float res = pv2[(int)lane * 8];",
        "        for (int s = 1; s < 8; s++) res += pv2[(int)lane * 8 + s];",
        "        sc[lane] = res * scale[0];",
        "    }",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
    ]
    new_soft = [
        "    // softmax over the N keys (mega_dpar): simd_max (order-free), one exp per lane,",
        "    // z summed serially from 0 by every lane (same order as the serial loop), e/z per lane.",
        "    threadgroup float ex[N];",
        "    if (sg == 0) {",
        "        float mv = (int)lane < N ? sc[lane] : -INFINITY;",
        "        mv = simd_max(mv);",
        "        if ((int)lane < N) ex[lane] = metal::precise::exp(sc[lane] - mv);",
        "        simdgroup_barrier(mem_flags::mem_threadgroup);",
        "        float z = 0;",
        "        for (int r = 0; r < N; ++r) z = ex[r] + z;",
        "        if ((int)lane < N) wts[lane] = ex[lane] / z;",
        "    }",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
    ]
    return "\n".join(lines[:s0] + new_scores + new_soft + lines[s2:])


_attnres_kernels = {}


def attn_res_supported(partial, query, n_keys):
    """The kernel's score reduction is MLX's split-K gemv (bm1 bn8 sn32 tn4).
    Stock ``keys @ q.T`` collapses its batch into M = rows * (n_keys + 1)
    gemv rows (matmul.cpp "Collapse batches into M") and keeps that variant
    only while K >= 16 * M (gemv_axbpy); above it MLX switches to bm4/sn32,
    a different K order, so the kernel would no longer be bitwise.  Uncollapsed batches use
    M = n_keys + 1, so the collapsed bound is the conservative one."""
    D = partial.shape[-1]
    rows = partial.size // D
    return (
        _on_gpu()
        and query.ndim == 1
        and query.shape[0] == D
        and D % 128 == 0
        and 256 <= D // 4 <= 1024
        and 1 <= n_keys + 1 <= 32
        # Metal has 31 buffer slots: 2 per key + partial(3) + query(2) +
        # scale + out = 2 * n_keys + 7 (n_keys <= 12, N <= 13); past it the
        # library fails to build (megafix, found by the N sweep)
        and 2 * n_keys + 7 <= 31
        and rows <= 16
        and D >= 16 * rows * (n_keys + 1)
    )


def attn_res(keys_list, partial, query, raw_partial=False):
    """AttentionResidual (1 query, normalize_values) from cached block keys.

    keys_list: n arrays [B, L, 1, D] f32 (rms-normed block states);
    partial: [B, L, D] (any float); query [D]. Returns [B, L, 1, D] f32,
    bitwise equal to _attn_res_keys / attn_res_prefix."""
    nk = len(keys_list)
    kern = _attnres_kernels.get(nk)
    if kern is None:
        kern = _attnres_kernels[nk] = mx.fast.metal_kernel(
            name=f"inkling2_attn_res_{nk}" + ("_dp" if enabled("mega_dpar") else ""),
            input_names=[f"k{i}" for i in range(nk)] + ["partial", "query", "scale"],
            output_names=["out"],
            source=_dpar(_attnres_source(nk)) if enabled("mega_dpar") else _attnres_source(nk),
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    B, L, D = partial.shape
    tg = D // 4
    return kern(
        # raw_partial (INKLING_FAST=mega_attnres): the kernel reads a bf16/f16
        # partial and converts on load, exactly like the astype it skips.
        inputs=[*keys_list, partial if raw_partial else partial.astype(mx.float32),
                query, _scalar(D**-0.5)],
        template=[("NK", nk), ("D", D), ("TG", tg)],
        grid=(tg, B * L, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, L, 1, D)],
        output_dtypes=[mx.float32],
    )[0]
# Refuse unverified MLX/GPU at import (needs _on_gpu above).
if FLAGS:
    check_exact_runtime()
