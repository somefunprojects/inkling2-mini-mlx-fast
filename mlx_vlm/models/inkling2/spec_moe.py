# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""INKLING_SPECFAST=moe: speculative-verify MoE (B*L rows, L in 2..8) for NVFP4 experts.

Decode-row arithmetic, expert-grid execution (specfast-moe):
  1. router_rows: mega_kernels' router GEMV (MLX gemv order at M=1) per row, row on grid y
     (no TF32 router GEMM at M>1).
  2. fast_kernels.route_topk on the R x E logits (row-independent; == decode per row).
  3. gua: grid z = P = R*top_k route slots; threadgroup z takes the z-th distinct routed
     expert (exits when there are fewer), lists the routes that chose it (ascending), and
     runs MLX 0.32.4 fp_qmv_impl / fp_qmv_fast_impl per-row code on each routed token's x
     row -- every weight word is read once and applied to up to TV tokens, and each
     (token, row) accumulator sees exactly stock's sequence of qdots -- then the stock
     SoftcappedSwiGLU on the T-rounded gate/up (megaperf3 mega_gua epilogue).
  4. down: same expert grid over act rows -> y [P, D] (stock fp_gather_qmv per-row code).
  5. language._combine_routes (stock combine, as the existing verify path).
No argsort / gather / scatter. Global-scale factors are computed in-kernel from the raw
global_scale with the stock f32 expression (nothing cached inside a compile trace).
Per-row code (_QMV_HEADER) copied from megaperf3 mega_kernels section 16.
"""
import math
import os

import mlx.core as mx

from . import fast_kernels as fk
from . import mega_kernels as mk

_QMV_HEADER = r"""
constant constexpr float QF8E4M3_MAX = 448.0f;
constant constexpr float QF4E2M1_MAX = 6.0f;
struct qfp4_e2m1 {
  operator float16_t() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }
  operator float() thread { return static_cast<float>(this->operator float16_t()); }
  uint8_t bits;
};
struct qfp8_e4m3 {
  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }
  operator float() thread { return static_cast<float>(this->operator float16_t()); }
  uint8_t bits;
};
struct qfp8_e8m0 {
  operator float() thread {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }
  uint8_t bits;
};
template <typename T, int group_size>
static inline T q_dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return T(*(thread qfp8_e4m3*)(&s));
  } else {
    return T(*(thread qfp8_e8m0*)(&s));
  }
}
template <int bits, typename U = float>
struct QDequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread qfp8_e4m3*)(&x));
    } else {
      return U(*(thread qfp4_e2m1*)(&x));
    }
  }
};
// qdot over values already in registers (wv: the lane's packed words).
template <typename U, int values_per_thread, int bits>
inline U q_qdot(const thread uint8_t* w, const thread U* x_thread, U scale) {
  U accum = 0;
  if constexpr (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * QDequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * QDequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * QDequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * QDequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * QDequantize<8>{}(w[i]);
    }
  }
  return scale * accum;
}
template <typename U, int values_per_thread, int bits>
inline U q_qdot_safe(const thread uint8_t* w, const thread U* x_thread, U scale, int N) {
  U accum = 0;
  if constexpr (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * QDequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * QDequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * QDequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * QDequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * QDequantize<8>{}(w[i]);
    }
  }
  return scale * accum;
}
"""


# Expert-grid helpers + multi-token per-row qmv (stock fp_qmv[_fast]_impl per-row code).
_SPEC_FN = r"""
#ifndef SPEC_PERF
#define SPEC_PERF 0
#endif
// The z-th distinct expert id (ascending) among idx[0..P), or -1.
template <int E, int P, typename IP>
inline int spec_expert(IP idx, int z, int lane) {
  constexpr int NW = (E + 31) / 32;
  uint m[NW];
  for (int w = 0; w < NW; w++) m[w] = 0u;
  for (int p0 = 0; p0 < P; p0 += 32) {
    const int p = p0 + lane;
    const uint e = p < P ? idx[p] : 0xFFFFFFFFu;
    for (int w = 0; w < NW; w++)
      m[w] |= simd_or((p < P && (int)(e >> 5) == w) ? (1u << (e & 31u)) : 0u);
  }
  int cnt = 0;
  for (int w = 0; w < NW; w++) {
    const int c = (int)popcount(m[w]);
    if (z < cnt + c) {
      uint mm = m[w];
      for (int j = 0; j < z - cnt; j++) mm &= mm - 1u;
      return w * 32 + (int)ctz(mm);
    }
    cnt += c;
  }
  return -1;
}
// Route positions p (ascending) with idx[p] == e -> plist[0..n); returns n.
template <int P, typename IP>
inline int spec_routes(IP idx, int e, int lane, threadgroup int* plist) {
  int n = 0;
  for (int p0 = 0; p0 < P; p0 += 32) {
    const int p = p0 + lane;
    const bool f = p < P && (int)idx[p] == e;
    const int pos = n + (int)simd_prefix_exclusive_sum((uint)f);
    if (f) plist[pos] = p;
    n += (int)simd_sum((uint)f);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  return n;
}
// Global-scale factor exactly as ScaledQuantizedSwitchLinear / global_scale_factor
// build it (f32 2688/s or 2688*s), then fp_qmv's inv_scale_enc = factor / (448*6).
template <int MODE, typename GP>
inline float spec_inv_scale(GP gs, int gn, int e) {
  if (MODE == 0) return 1.0f;
  const float s = static_cast<float>(gs[gn == 1 ? 0 : e]);
  const float f = MODE == 1 ? 2688.0f / s : 2688.0f * s;
  return f / (QF8E4M3_MAX * QF4E2M1_MAX);
}
// Byte -> (fp4 low nibble, fp4 high nibble) as floats, built with MLX's own
// Dequantize<4> conversion (exact values incl. -0.0). Built by every TG once.
inline void spec_build_lut(threadgroup float2* lut, uint tid, uint nthreads) {
  for (uint b = tid; b < 256; b += nthreads)
    lut[b] = float2(QDequantize<4>{}((uint8_t)(b & 15u)), QDequantize<4>{}((uint8_t)(b >> 4)));
}
// MLX qdot (bits 4) with the lane's values already dequantized: same expression and
// the same function boundary (scale * accum is returned, then added by the caller), so
// the compiler makes the same FMA-contraction choices as for stock qdot.
inline float spec_qdot_d(const thread float* x, const thread float* d, float scale, int nv) {
  float accum = 0;
  for (int i = 0; i < (nv / 4); i++)
    accum += (x[4 * i] * d[4 * i] + x[4 * i + 1] * d[4 * i + 1] + x[4 * i + 2] * d[4 * i + 2] +
              x[4 * i + 3] * d[4 * i + 3]);
  return scale * accum;
}
// MLX fp_qmv_impl (FAST=false: 1 pack/lane, remainder via qdot_safe) /
// fp_qmv_fast_impl (FAST=true: 2 packs/lane, K % 512 == 0) per-row code for R rows of
// one expert and TV input vectors. Each lane's packed word is dequantized once (LUT) and
// applied to every vector with MLX qdot's expression
//   accum += (x0*w0 + x1*w1 + x2*w2 + x3*w3);  result += scale * accum
// so each (vector, row) accumulator sees exactly stock's sequence.
// result[v * R + r] += per-lane partials (no simd_sum).
// 8 consecutive input values -> float (exact widening). 16-byte vector loads for
// bf16 / f32 device inputs; plain loads otherwise (e.g. small inputs MLX binds as constant).
template <typename XP>
inline void spec_load8(XP p, thread float* o) {
  for (int i = 0; i < 8; i++) o[i] = static_cast<float>(p[i]);
}
inline void spec_load8(const device bfloat16_t* p, thread float* o) {
  const uint4 u = *(const device uint4*)p;
  o[0] = as_type<float>(u.x << 16); o[1] = as_type<float>(u.x & 0xFFFF0000u);
  o[2] = as_type<float>(u.y << 16); o[3] = as_type<float>(u.y & 0xFFFF0000u);
  o[4] = as_type<float>(u.z << 16); o[5] = as_type<float>(u.z & 0xFFFF0000u);
  o[6] = as_type<float>(u.w << 16); o[7] = as_type<float>(u.w & 0xFFFF0000u);
}
inline void spec_load8(const device float* p, thread float* o) {
  const float4 a = *(const device float4*)p, b = *(const device float4*)(p + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
#ifndef SPEC_LUT
#define SPEC_LUT 1
#endif
#ifndef SPEC_XV
#define SPEC_XV 1
#endif
template <int PPT>
inline void spec_deq(const thread uint32_t* wv, thread float* d, const threadgroup float2* lut) {
#pragma unroll
  for (int j = 0; j < PPT; j++) {
    if (SPEC_LUT) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float2 dq = lut[(wv[j] >> (8 * q)) & 255u];
        d[8 * j + 2 * q] = dq.x;
        d[8 * j + 2 * q + 1] = dq.y;
      }
    } else {
#pragma unroll
      for (int q = 0; q < 8; q++) d[8 * j + q] = QDequantize<4>{}((uint8_t)(wv[j] >> (4 * q)));
    }
  }
}
template <typename XP>
inline void spec_loadx(XP p, thread float* o, int nv) {
  for (int i = 0; i < 8; i++) o[i] = i < nv ? static_cast<float>(p[i]) : 0.0f;
}
// Stock loop structure (pointer increments, 32-bit offsets in the loop); per-row
// arithmetic as MLX fp_qmv_impl / fp_qmv_fast_impl.
template <typename T, int K, int GS, int R, int TV, bool FAST, typename WP, typename SP, typename XP>
inline void qmv_tok(WP wbase, SP sbase, const thread int* roff, const thread int* xoff, XP x,
                    int lane, thread float* result, const threadgroup float2* lut) {
  constexpr int PF = 8;
  constexpr int PPT = FAST ? 2 : 1;
  constexpr int VPT = PF * PPT;
  constexpr int BLOCK = VPT * 32;
  constexpr int KW = K / PF;
  constexpr int KG = K / GS;
  if (SPEC_PERF == 3) return;
  WP wl[R];
  SP sl[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    wl[r] = wbase + roff[r] * KW + lane * PPT;
    sl[r] = sbase + roff[r] * KG + (lane * VPT) / GS;
  }
  XP xp[TV];
#pragma unroll
  for (int v = 0; v < TV; v++) xp[v] = x + xoff[v] + lane * VPT;
  float xt[TV][VPT];
  int k = 0;
  for (; FAST ? (k < K) : (k < K - BLOCK); k += BLOCK) {
#pragma unroll
    for (int v = 0; v < TV; v++)
#pragma unroll
      for (int h = 0; h < PPT; h++) {
        if (SPEC_XV) spec_load8(xp[v] + 8 * h, xt[v] + 8 * h);
        else for (int i = 0; i < 8; i++) xt[v][8 * h + i] = static_cast<float>(xp[v][8 * h + i]);
      }
#pragma unroll
    for (int r = 0; r < R; r++) {
      uint32_t wv[PPT];
#pragma unroll
      for (int j = 0; j < PPT; j++) wv[j] = wl[r][j];
      float d[VPT];
      spec_deq<PPT>(wv, d, lut);
      const float s = q_dequantize_scale<float, GS>(sl[r][0]);
#pragma unroll
      for (int v = 0; v < TV; v++) result[v * R + r] += spec_qdot_d(xt[v], d, s, VPT);
      wl[r] += BLOCK / PF;
      sl[r] += BLOCK / GS;
    }
#pragma unroll
    for (int v = 0; v < TV; v++) xp[v] += BLOCK;
  }
  if (!FAST) {
    const int rem = metal::clamp(K - k - lane * VPT, 0, VPT);
    if (rem > 0) {
      for (int v = 0; v < TV; v++) spec_loadx(xp[v], xt[v], rem);
      for (int r = 0; r < R; r++) {
        uint32_t wv[1] = {wl[r][0]};
        float d[VPT];
        spec_deq<1>(wv, d, lut);
        const float s = q_dequantize_scale<float, GS>(sl[r][0]);
        for (int v = 0; v < TV; v++) result[v * R + r] += spec_qdot_d(xt[v], d, s, rem);
      }
    }
  }
}
"""

# Shared prologue: LUT, expert of this z slot, its route list (per simdgroup).
_PROLOGUE = r"""
    // meta[z] = {expert, n routes, route positions...} from the meta kernel.
    threadgroup float2 lut[256];
    const int lane = (int)thread_index_in_simdgroup;
    const int sg = (int)simdgroup_index_in_threadgroup;
    if (SPEC_PERF == 4) return;
    // Grid z = ZS <= P slots; threadgroup z serves distinct experts z, z+ZS, ...
    if (meta[(int)threadgroup_position_in_grid.z * (2 + RMAX)] < 0) return;  // uniform
    if (SPEC_LUT) {
        spec_build_lut(lut, (uint)(sg * 32 + lane), (uint)(SGS * 32));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int zs = (int)threadgroup_position_in_grid.z; zs < P; zs += ZS) {
    auto mz = meta + zs * (2 + RMAX);
    const int e = mz[0];
    if (e < 0) break;
    const int n = mz[1];
    auto plist = mz + 2;
    const int ybeg = (int)threadgroup_position_in_grid.y * RPC;
    if (SPEC_PERF == 2) return;
    constexpr int KW = K / 8;
    constexpr int KG = K / GS;
"""

# Gate+up(+SoftcappedSwiGLU) over unique experts. Grid z = P route slots; TG z
# takes the z-th distinct routed expert (exits if fewer), finds its routes and
# runs the per-row code on its tokens' x rows in chunks of <= 4 (exact chunk sizes).
# act[p, :] = activation(up, gate) for route p = row * KT + slot.
_GUA_SRC = _PROLOGUE + r"""
    auto wgb = wg + (size_t)e * WR * KW;
    auto sgb = sgs + (size_t)e * WR * KG;
    auto wub = wu + (size_t)e * WR * KW;
    auto sub = sus + (size_t)e * WR * KG;
    const float ig = spec_inv_scale<SMG>(gsg, GNG, e);
    const float iu = spec_inv_scale<SMU>(gsu, GNU, e);
    const float gc = caps[0], uc = caps[1];
    for (int row0 = ybeg + sg * ROWS; row0 < ybeg + RPC; row0 += SGS * ROWS) {
    int ro2[2 * ROWS];
    for (int r = 0; r < ROWS; r++) { ro2[r] = row0 + r; ro2[ROWS + r] = UOFF + row0 + r; }
    auto emit = [&](const thread float* rg, const thread float* ru, int rstride, int t0, int tv) {
        for (int v = 0; v < tv; v++) {
            const int p = plist[t0 + v];
            for (int r = 0; r < ROWS; r++) {
                const float sgv = simd_sum(rg[v * rstride + r]);
                const float suv = simd_sum(ru[v * rstride + r]);
                if (lane == 0) {
                    const float g = static_cast<float>(static_cast<T>(SMG ? sgv * ig : sgv));
                    const float u = static_cast<float>(static_cast<T>(SMU ? suv * iu : suv));
                    const float yg = 1 / (1 + metal::precise::exp(metal::abs(g)));
                    const float sig = (g < 0) ? yg : 1 - yg;
                    float a = metal::precise::tanh(g / gc) * gc;
                    a = a * sig;
                    a = a * metal::precise::tanh(u / uc);
                    a = a * uc;
                    act[(size_t)p * NOUT + row0 + r] = static_cast<T>(a);
                }
            }
        }
    };
#define SPEC_GUA_CHUNK(TVC)                                                              \
    {                                                                                     \
        int xoff[TVC];                                                                    \
        for (int v = 0; v < TVC; v++) xoff[v] = (plist[t0 + v] / KT) * K;                 \
        if (SAMEW) {                                                                      \
            float res[TVC * 2 * ROWS];                                                    \
            for (int i = 0; i < TVC * 2 * ROWS; i++) res[i] = 0;                          \
            qmv_tok<T, K, GS, 2 * ROWS, TVC, FAST>(wgb, sgb, ro2, xoff, x, lane, res, lut); \
            emit(res, res + ROWS, 2 * ROWS, t0, TVC);                                     \
        } else {                                                                          \
            float rg[TVC * ROWS], ru[TVC * ROWS];                                         \
            for (int i = 0; i < TVC * ROWS; i++) { rg[i] = 0; ru[i] = 0; }                \
            qmv_tok<T, K, GS, ROWS, TVC, FAST>(wgb, sgb, ro2, xoff, x, lane, rg, lut);      \
            qmv_tok<T, K, GS, ROWS, TVC, FAST>(wub, sub, ro2 + ROWS, xoff, x, lane, ru, lut); \
            emit(rg, ru, ROWS, t0, TVC);                                                  \
        }                                                                                 \
    }
    for (int t0 = 0; t0 < n;) {
        const int tv = min(n - t0, TV);
        if (tv >= 4 && TV >= 4) { SPEC_GUA_CHUNK(4) t0 += 4; }
        else if (tv == 3 && TV >= 3) { SPEC_GUA_CHUNK(3) t0 += 3; }
        else if (tv == 2 && TV >= 2) { SPEC_GUA_CHUNK(2) t0 += 2; }
        else { SPEC_GUA_CHUNK(1) t0 += 1; }
    }
    }
    }
#undef SPEC_GUA_CHUNK
"""

# Down projection over unique experts: y[p, :] = act[p, :] @ W[idx[p]].T (stock per-row code).
_DOWN_SRC = _PROLOGUE + r"""
    auto wb = w + (size_t)e * NOUT * KW;
    auto sb = scales + (size_t)e * NOUT * KG;
    const float inv = spec_inv_scale<SM>(gs, GN, e);
    for (int row0 = ybeg + sg * ROWS; row0 < ybeg + RPC; row0 += SGS * ROWS) {
    int roff[ROWS];
    for (int r = 0; r < ROWS; r++) roff[r] = row0 + r;
#define SPEC_DOWN_CHUNK(TVC)                                                         \
    {                                                                                \
        int xoff[TVC];                                                               \
        for (int v = 0; v < TVC; v++) xoff[v] = plist[t0 + v] * K;                   \
        float res[TVC * ROWS];                                                       \
        for (int i = 0; i < TVC * ROWS; i++) res[i] = 0;                             \
        qmv_tok<T, K, GS, ROWS, TVC, FAST>(wb, sb, roff, xoff, act, lane, res, lut);  \
        for (int v = 0; v < TVC; v++) {                                              \
            const int p = plist[t0 + v];                                             \
            for (int r = 0; r < ROWS; r++) {                                         \
                const float rr = simd_sum(res[v * ROWS + r]);                        \
                if (lane == 0) y[(size_t)p * NOUT + row0 + r] = static_cast<T>(SM ? rr * inv : rr); \
            }                                                                        \
        }                                                                            \
    }
    for (int t0 = 0; t0 < n;) {
        const int tv = min(n - t0, TV);
        if (tv >= 4 && TV >= 4) { SPEC_DOWN_CHUNK(4) t0 += 4; }
        else if (tv == 3 && TV >= 3) { SPEC_DOWN_CHUNK(3) t0 += 3; }
        else if (tv == 2 && TV >= 2) { SPEC_DOWN_CHUNK(2) t0 += 2; }
        else { SPEC_DOWN_CHUNK(1) t0 += 1; }
    }
    }
    }
#undef SPEC_DOWN_CHUNK
"""

# One simdgroup: distinct routed experts (ascending) and, for each, its route positions.
_META_SRC = r"""
    // One threadgroup of MT >= P threads. Thread z: z-th distinct routed expert (ascending),
    // its route count and positions (ascending) -> meta[z] = {e, n, p0..p_{RMAX-1}}.
    constexpr int NW = (E + 31) / 32;
    threadgroup uint sidx[P];
    threadgroup uint smask[NW];
    const int t = (int)thread_position_in_threadgroup.x;
    if (t < P) sidx[t] = idx[t];
    if (t < NW) smask[t] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < P) atomic_fetch_or_explicit((threadgroup atomic_uint*)&smask[sidx[t] >> 5], 1u << (sidx[t] & 31u),
                                        memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= P) return;
    int e = -1, cnt = 0;
    for (int w = 0; w < NW && e < 0; w++) {
        const uint m = smask[w];
        const int c = (int)popcount(m);
        if (t < cnt + c) {
            uint mm = m;
            for (int j = 0; j < t - cnt; j++) mm &= mm - 1u;
            e = w * 32 + (int)ctz(mm);
        }
        cnt += c;
    }
    device int* out = meta + t * (2 + RMAX);
    out[0] = e;
    if (e < 0) return;
    int n = 0;
    for (int p = 0; p < P; p++)
        if ((int)sidx[p] == e) out[2 + n++] = p;
    out[1] = n;
"""


_PROBE_SRC = r"""
    threadgroup int plist[RMAX];
    const int lane = (int)thread_index_in_simdgroup;
    const int z = (int)threadgroup_position_in_grid.z;
    const int e = spec_expert<E, P>(idx, z, lane);
    int n = -1;
    if (e >= 0) n = spec_routes<P>(idx, e, lane, plist);
    if (lane == 0) {
        out[z * (2 + RMAX)] = e;
        out[z * (2 + RMAX) + 1] = n;
        for (int i = 0; i < RMAX; i++) out[z * (2 + RMAX) + 2 + i] = i < n ? plist[i] : -9;
    }
"""

# Router logits for R rows: mega_kernels._ROUTER_SRC (MLX gemv order, bitwise vs
# x.astype(f32) @ W.T at M=1) with the row on grid y.
_ROUTER_ROWS_PRE = r"""
    x += (size_t)threadgroup_position_in_grid.y * K;
    logits += (size_t)threadgroup_position_in_grid.y * E;
"""

GUA_ROWS = int(os.environ.get("INKLING_SMOE_GUA_ROWS", "2"))
GUA_SGS = int(os.environ.get("INKLING_SMOE_GUA_SGS", "4"))
GUA_TV = int(os.environ.get("INKLING_SMOE_GUA_TV", "1"))
DOWN_ROWS = int(os.environ.get("INKLING_SMOE_DOWN_ROWS", "4"))
DOWN_SGS = int(os.environ.get("INKLING_SMOE_DOWN_SGS", "2"))
DOWN_TV = int(os.environ.get("INKLING_SMOE_DOWN_TV", "1"))
GUA_CH = int(os.environ.get("INKLING_SMOE_GUA_CH", "0"))  # 0: one row tile per simdgroup    # threadgroups (row chunks) per expert
DOWN_CH = int(os.environ.get("INKLING_SMOE_DOWN_CH", "0"))
ZFRAC = float(os.environ.get("INKLING_SMOE_ZFRAC", "0.6"))  # z slots / routes (distinct experts expected <~ this)
MAX_ROWS = 16  # B*L rows handled (verify uses L <= 8)

_kern = {}
_perf_hdr = ""  # tuning knobs (#define SPEC_PERF/SPEC_LUT/SPEC_XV) for benches


def _ch(n, rows, sgs, ch):
    return ch if ch else n // (rows * sgs)


def _kernel(name, inputs, outputs, src, header, contiguous=True):
    k = _kern.get(name)
    if k is None:
        k = _kern[name] = mx.fast.metal_kernel(
            name=f"inkling2_spec_moe_{name}" + ("_" + str(abs(hash(_perf_hdr))) if _perf_hdr else ""),
            input_names=inputs,
            output_names=outputs,
            header=header,
            source=src,
            ensure_row_contiguous=contiguous,
            compile_options={"math_mode": "safe"},
        )
    return k


def router_rows(x2, w32):
    """x2 [R, K] (bf16/f16/f32), w32 [E, K] f32 -> logits [R, E] f32; row i ==
    mega_kernels.router_logits(x2[i]) == (x2[i].astype(f32) @ w32.T) at M=1."""
    R, K = x2.shape
    E = w32.shape[0]
    k = _kernel("router", ["x", "w"], ["logits"], _ROUTER_ROWS_PRE + mk._ROUTER_SRC, "")
    return k(
        inputs=[x2, w32],
        template=[("K", K), ("E", E)],
        grid=(256 * (E // 4), R, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, E)],
        output_dtypes=[mx.float32],
    )[0]


def _scale_mode(lin):
    """(mode, global_scale array, size) for the per-expert factor the stock
    ScaledQuantizedSwitchLinear passes to gather_qmm (mode 0: no global scale)."""
    if not hasattr(lin, "global_scale"):
        return 0, _dummy(), 1
    gs = lin.global_scale
    mode = 1 if lin._scale_name == "weight_global_scale" else 2
    return mode, gs.reshape(-1), int(gs.size)


_dummies = {}
_capsd = {}


def _caps(gc, uc):
    key = (float(gc), float(uc))
    a = _capsd.get(key)
    if a is None:
        a = _capsd[key] = mx.array([float(gc), float(uc)], mx.float32)
    return a


def _dummy():
    a = _dummies.get(0)
    if a is None:
        a = _dummies[0] = mx.array([0.0], mx.float32)  # data-backed: safe inside a trace
    return a


def _lin_ok(lin, K, N):
    from ..masked_quantized import supports_masked_qmv

    if not supports_masked_qmv(lin, input_alignment=16):
        return False
    if lin.input_dims != K or lin.output_dims != N:
        return False
    if hasattr(lin, "global_scale"):
        if lin.global_scale.size not in (1, lin.num_experts):
            return False
        if lin._scale_name not in ("weight_global_scale", "weight_scale_2"):
            return False
    return True


def _fast_variant(K):
    # MLX 0.32 gather_qmv dispatch: fp_qmv_fast when K % 512 == 0 (and N % 8 == 0).
    return K % 512 == 0


def supported(moe, x, fused):
    if moe.sharding_group is not None or not fk._on_gpu():
        return False
    if x.ndim != 3 or x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    B, L, D = x.shape
    if B * L > MAX_ROWS or B * L < 1 or B * L * moe.top_k > 1024:
        return False
    sm = moe.switch_mlp
    act = sm.activation
    if not (hasattr(act, "gate_cap") and hasattr(act, "up_cap")):
        return False
    w32 = moe._fast_router_weight()
    E = w32.shape[0]
    if not (w32.dtype == mx.float32 and w32.shape[1] == D and E % 4 == 0 and D >= 16 * E
            and D % 4 == 0 and E <= 1024 and moe.top_k <= E):
        return False
    g, u, dn = sm.gate_proj, sm.up_proj, sm.down_proj
    NH = g.output_dims
    if not (_lin_ok(g, D, NH) and _lin_ok(u, D, NH) and _lin_ok(dn, NH, D)):
        return False
    if g.num_experts != E or u.num_experts != E or dn.num_experts != E:
        return False
    gch, dch = _ch(NH, GUA_ROWS, GUA_SGS, GUA_CH), _ch(D, DOWN_ROWS, DOWN_SGS, DOWN_CH)
    if (not gch or not dch or NH % gch or (NH // gch) % GUA_ROWS or D % dch
            or (D // dch) % DOWN_ROWS):
        return False
    if fused is not None and not (fused.get("mode") == "nvfp4" and fused.get("n") == NH):
        return False
    return True


def moe_verify(moe, x, fused=None, stages=False, stages_act=None, routed=False):
    """SparseMoE.__call__ for x [B, L, D] (L > 1) with every row computed by the
    L=1 decode path's arithmetic (router GEMV, route_topk, fp_gather_qmv per-row
    gate/up/down, SoftcappedSwiGLU, _combine_routes), reading each routed
    expert's weights once per verify step. None if unsupported."""
    global _perf_hdr
    if not supported(moe, x, fused):
        return None
    from .language import _combine_routes

    B, L, D = x.shape
    R = B * L
    sm = moe.switch_mlp
    g, u, dn = sm.gate_proj, sm.up_proj, sm.down_proj
    NH = g.output_dims
    x2 = x.reshape(R, D)
    w32 = moe._fast_router_weight()
    E = w32.shape[0]
    KT = int(moe.top_k)
    P = R * KT
    gch, dch = _ch(NH, GUA_ROWS, GUA_SGS, GUA_CH), _ch(D, DOWN_ROWS, DOWN_SGS, DOWN_CH)
    logits = router_rows(x2, w32)
    idx, wts = fk.route_topk(logits, moe.e_score_correction_bias, KT, moe.route_scale)
    idx = idx.reshape(P)
    if stages == "upto_route":
        return idx
    mt = max(32, (P + 31) // 32 * 32)
    zs = max(1, min(P, E, int(math.ceil(P * ZFRAC))))
    meta = _kernel("meta", ["idx"], ["meta"], _META_SRC, _QMV_HEADER + _SPEC_FN)(
        inputs=[idx], template=[("E", E), ("P", P), ("RMAX", R)],
        grid=(mt, 1, 1), threadgroup=(mt, 1, 1),
        output_shapes=[(P * (2 + R),)], output_dtypes=[mx.int32])[0]
    if stages == "upto_meta":
        return meta
    smg, gsg, gng = _scale_mode(g)
    smu, gsu, gnu = _scale_mode(u)
    if fused is not None:
        wg = wu = fused["weight"]
        sg = su = fused["scales"]
        wr, uoff, samew = 2 * NH, NH, True
    else:
        wg, sg, wu, su = g.weight, g.scales, u.weight, u.scales
        wr, uoff, samew = NH, 0, False
    caps = _caps(sm.activation.gate_cap, sm.activation.up_cap)
    gua = _kernel(
        "gua", ["meta", "x", "wg", "sgs", "gsg", "wu", "sus", "gsu", "caps"], ["act"],
        _GUA_SRC, _perf_hdr + _QMV_HEADER + _SPEC_FN,
    )
    act = gua(
        inputs=[meta, x2, wg, sg, gsg, wu, su, gsu, caps],
        template=[("T", x.dtype), ("K", D), ("NOUT", NH), ("E", E), ("KT", KT), ("P", P),
                  ("RMAX", R), ("GS", 16), ("ROWS", GUA_ROWS), ("SGS", GUA_SGS),
                  ("TV", GUA_TV), ("FAST", _fast_variant(D)), ("WR", wr), ("UOFF", uoff),
                  ("SAMEW", samew), ("SMG", smg), ("SMU", smu), ("GNG", gng), ("GNU", gnu),
                  ("RPC", NH // gch), ("ZS", zs)],
        grid=(32, GUA_SGS * gch, zs),
        threadgroup=(32, GUA_SGS, 1),
        output_shapes=[(P, NH)],
        output_dtypes=[x.dtype],
    )[0]
    if stages == "upto_gua":
        return act
    if stages == "act_in":
        act = stages_act
    smd, gsd, gnd = _scale_mode(dn)
    down = _kernel(
        "down", ["meta", "act", "w", "scales", "gs"], ["y"],
        _DOWN_SRC, _perf_hdr + _QMV_HEADER + _SPEC_FN,
    )
    y = down(
        inputs=[meta, act, dn.weight, dn.scales, gsd],
        template=[("T", x.dtype), ("K", NH), ("NOUT", D), ("E", E), ("P", P), ("RMAX", R),
                  ("GS", 16), ("ROWS", DOWN_ROWS), ("SGS", DOWN_SGS),
                  ("TV", DOWN_TV), ("FAST", _fast_variant(NH)), ("SM", smd), ("GN", gnd),
                  ("RPC", D // dch), ("ZS", zs)],
        grid=(32, DOWN_SGS * dch, zs),
        threadgroup=(32, DOWN_SGS, 1),
        output_shapes=[(P, D)],
        output_dtypes=[x.dtype],
    )[0]
    if stages == "act_in":
        return y
    if stages:
        probe = _kernel("probe", ["idx"], ["out"], _PROBE_SRC, _QMV_HEADER + _SPEC_FN)(
            inputs=[idx], template=[("E", E), ("P", P), ("RMAX", R)],
            grid=(32, 1, P), threadgroup=(32, 1, 1),
            output_shapes=[(P, 2 + R)], output_dtypes=[mx.int32])[0]
        return logits, idx, wts, act, y, probe
    if routed:  # per-route outputs + weights, as SparseMoE._routed (sconv_tail combines)
        return y.reshape(B, L, KT, D), wts.reshape(B, L, KT)
    return _combine_routes(y.reshape(B, L, KT, D), wts.reshape(B, L, KT))
