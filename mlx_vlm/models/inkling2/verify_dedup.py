# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""Expert-grouped (dedup) quantized matvecs for the compiled block verify.

All routes of one expert (sorted ids) are handled by one threadgroup, V input vectors per weight read.
Each vector's reduction mirrors MLX's own fp quantized matvec (mlx 0.32 fp_quantized.h):
  FAST=1 : fp_qmv_fast_impl  (block = 32 * P, no remainder)          -- used when K % block == 0
  FAST=0 : fp_qmv_impl       (packs_per_thread = 1, last block via load_vector_safe / qdot_safe)
so outputs are meant to be bitwise equal to stock gather_qmm(sorted_indices=False) on one route per row.
Which MLX kernel gather_qmm picks for a given (mode, K) is checked empirically (harness/dspark_kern_check.py).

Flags (read by fast_decode.FastDecoder):
  INKLING_DS_DEDUP_DOWN=1   NVFP4 down_proj (K=1152) through the grouped kernel (was: per-route gather_qmm)
  INKLING_DS_DEDUP_MXFP8=1  MXFP8 layers (20-23): per-token routing, grouped gate/up/down (was: tokenwise MoE)
"""

from functools import lru_cache

import mlx.core as mx

_EXTRA = r"""
template <typename T, typename U, int values_per_thread>
inline void ds_load_vector_safe(const device T* x, thread U* x_thread, int n) {
  for (int i = 0; i < n; i++) x_thread[i] = x[i];
  for (int i = n; i < values_per_thread; i++) x_thread[i] = 0;
}
template <typename U, int values_per_thread, int bits>
inline U ds_qdot_safe(const device uint8_t* w, const thread U* x_thread, U scale, int n) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (n / 4); i++) {
      accum += (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
                x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
                x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
                x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < n; i++) accum += x_thread[i] * Dequantize<8>{}(w[i]);
  }
  return scale * accum;
}
"""

_SRC = r"""
uint first = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
int row0 = threadgroup_position_in_grid.y * 8 + sg * 4;
int expert = ids[first];
if (expert < 0) {
  if (lane == 0) for (int r = 0; r < 4; ++r) out[first * N + row0 + r] = T(0);
  return;
}
if (first > 0 && ids[first - 1] == expert) return;
int end = first + 1;
while (end < ids_shape[0] && ids[end] == expert) ++end;
constexpr int BLOCK = P * 32;
constexpr int RB = K * BITS / 8;           // bytes per weight row
constexpr int KG = K / GS;                 // scales per row
constexpr int KLIM = FAST ? K : K - BLOCK;
for (int begin = first; begin < end; begin += V) {
  thread float xt[V][P];
  thread float result[V][4] = {0};
  const device uint8_t* ws = (const device uint8_t*)w + size_t(expert) * N * RB + size_t(row0) * RB + lane * (P * BITS / 8);
  const device uint8_t* sc = s + size_t(expert) * N * KG + size_t(row0) * KG + lane / (GS / P);
  int k = 0;
  for (; k < KLIM; k += BLOCK) {
    #pragma unroll
    for (int v = 0; v < V; ++v) load_vector<T, float, P>(x + min(begin + v, end - 1) * K + lane * P + k, xt[v]);
    for (int r = 0; r < 4; ++r) {
      auto wl = ws + r * RB;
      float scale = dequantize_scale<float, GS>(sc[r * KG]);
      #pragma unroll
      for (int v = 0; v < V; ++v) result[v][r] += qdot<float, P, BITS>(wl, xt[v], scale);
    }
    ws += BLOCK * BITS / 8;
    sc += BLOCK / GS;
  }
  if (!FAST) {
    int rem = clamp(int(K - k - int(lane) * P), 0, P);
    if (rem > 0) {
      for (int v = 0; v < V; ++v) ds_load_vector_safe<T, float, P>(x + min(begin + v, end - 1) * K + lane * P + k, xt[v], rem);
      for (int r = 0; r < 4; ++r) {
        auto wl = ws + r * RB;
        float scale = dequantize_scale<float, GS>(sc[r * KG]);
        for (int v = 0; v < V; ++v) result[v][r] += ds_qdot_safe<float, P, BITS>(wl, xt[v], scale, rem);
      }
    }
  }
  float inv = 1.0f;
  if (HASG) inv = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
  for (int r = 0; r < 4; ++r) {
    #pragma unroll
    for (int v = 0; v < V; ++v) {
      result[v][r] = simd_sum(result[v][r]);
      if (lane == 0 && begin + v < end) {
        if (HASG) out[(begin + v) * N + row0 + r] = T(result[v][r] * inv);
        else out[(begin + v) * N + row0 + r] = T(result[v][r]);
      }
    }
  }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    from ..masked_quantized import _HEADER

    return mx.fast.metal_kernel(
        name="ds_grouped_fp_qmv",
        input_names=["x", "w", "s", "g", "ids"],
        output_names=["out"],
        header=_HEADER + _EXTRA,
        source=_SRC,
        compile_options={"math_mode": "safe"},
    )


def kind_of(linear):
    """(bits, group_size, has_global_scale) for supported fp-quantized switch linears, else None."""
    from ...quantization.nvfp4 import ScaledQuantizedSwitchLinear
    from ..switch_layers import QuantizedSwitchLinear

    if not isinstance(linear, (QuantizedSwitchLinear, ScaledQuantizedSwitchLinear)):
        return None
    if (
        "bias" in linear
        or linear.get("biases") is not None
        or "pre_quant_scale" in linear
        or getattr(linear, "_quantize_activations", False)
        or linear.weight.dtype != mx.uint32
        or linear.scales.dtype != mx.uint8
        or linear.output_dims % 8
    ):
        return None
    if linear.mode == "nvfp4" and linear.bits == 4 and linear.group_size == 16:
        if isinstance(linear, ScaledQuantizedSwitchLinear) and linear.global_scale.size not in (
            1, linear.num_experts):
            return None
        return (4, 16, True)
    if linear.mode == "mxfp8" and linear.bits == 8 and linear.group_size == 32 and type(
            linear) is QuantizedSwitchLinear:
        return (8, 32, False)
    return None


def grouped_qmv(x, linear, ids, V=2, fast=None):
    """x [R, K] (one vector per route, routes sorted by expert), ids [R] int32 sorted -> [R, N]."""
    from ..masked_quantized import _global_scale

    bits, gs, hasg = kind_of(linear._gate if isinstance(linear, _FusedView) else linear)
    K, N = linear.input_dims, linear.output_dims
    P_fast = 2 * (32 // bits)  # fp_qmv_fast: packs_per_thread = 2
    if fast is None:
        fast = K % (32 * P_fast) == 0
    P = P_fast if fast else 32 // bits  # fp_qmv: packs_per_thread = 1
    g = _global_scale(linear) if hasg else mx.zeros((64,), mx.float32)
    return _kernel()(
        inputs=[x, linear.weight, linear.scales, g, ids],
        template=[("T", x.dtype), ("K", K), ("N", N), ("V", V), ("P", P), ("BITS", bits),
                  ("GS", gs), ("HASG", int(hasg)), ("FAST", int(fast))],
        grid=(32, N // 8 * 2, ids.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(ids.size, N)],
        output_dtypes=[x.dtype],
    )[0]


def stock_route(moe, x):
    """Exactly SparseMoE.__call__'s routing for x (any L)."""
    from . import fast_kernels as fk
    from .language import _route

    if fk.enabled("route") and not moe.training and fk._on_gpu():
        return fk.route(x, moe._fast_router_weight(), moe.e_score_correction_bias, moe.top_k,
                        moe.route_scale)
    return _route(x, moe._fast_router_weight() if fk.enabled("glue") else moe.gate_weight,
                  moe.e_score_correction_bias, moe.top_k, moe.route_scale)


def route_tokenwise(moe, x):
    """Route each position separately (the decode L=1 shapes), concat -> [B, L, k]."""
    parts = [stock_route(moe, x[:, i : i + 1]) for i in range(x.shape[1])]
    return (mx.concatenate([p[0] for p in parts], axis=1),
            mx.concatenate([p[1] for p in parts], axis=1))


def supported(switch_mlp):
    return all(kind_of(getattr(switch_mlp, n)) is not None for n in ("gate_proj", "up_proj", "down_proj"))


class _FusedView:
    """gate_proj stand-in over the contiguous fusemoe [gate; up] buffer (N = 2n), so the
    kernel never sees the strided gate/up views (metal_kernel would copy them per call)."""

    def __init__(self, gate, fused):
        self._gate, self.weight, self.scales = gate, fused["weight"], fused["scales"]
        self.output_dims = 2 * fused["n"]

    def __getattr__(self, name):
        return getattr(self._gate, name)

    def get(self, name, default=None):
        return self._gate.get(name, default)

    def __contains__(self, name):
        return name in self._gate


def dedup_routes(switch_mlp, x, indices, V=2, fused=None):
    """Grouped gate/up/down over all B*L*k routes -> per-route outputs [B, L, k, D] (pre-combine)."""
    sm = switch_mlp
    B, L, D = x.shape
    k = indices.shape[-1]
    flat = indices.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    sel = flat[order]
    vals = x.reshape(-1, D)[order // k]
    if fused is not None:
        n = fused["n"]
        gu = grouped_qmv(vals, _FusedView(sm.gate_proj, fused), sel, V)
        gate, up = gu[..., :n], gu[..., n:]
    else:
        up = grouped_qmv(vals, sm.up_proj, sel, V)
        gate = grouped_qmv(vals, sm.gate_proj, sel, V)
    out = grouped_qmv(sm.activation(up, gate), sm.down_proj, sel, V)
    return out[inv].reshape(B, L, k, D)
