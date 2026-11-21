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
#pragma unroll
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
__attribute__((always_inline)) inline void spec_load8(XP p, thread float* o) {
  for (int i = 0; i < 8; i++) o[i] = static_cast<float>(p[i]);
}
__attribute__((always_inline)) inline void spec_load8(const device bfloat16_t* p, thread float* o) {
  const uint4 u = *(const device uint4*)p;
  o[0] = as_type<float>(u.x << 16); o[1] = as_type<float>(u.x & 0xFFFF0000u);
  o[2] = as_type<float>(u.y << 16); o[3] = as_type<float>(u.y & 0xFFFF0000u);
  o[4] = as_type<float>(u.z << 16); o[5] = as_type<float>(u.z & 0xFFFF0000u);
  o[6] = as_type<float>(u.w << 16); o[7] = as_type<float>(u.w & 0xFFFF0000u);
}
__attribute__((always_inline)) inline void spec_load8(const device float* p, thread float* o) {
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
__attribute__((always_inline)) inline void spec_deq(const thread uint32_t* wv, thread float* d, const threadgroup float2* lut) {
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


# meta2: like _META_SRC, but each distinct expert's routes are split into chunks of TV (ascending route
# order), one chunk per slot: meta2[z] = {e, nv, p0..p_{TV-1}}; slots beyond the chunk count get e = -1.
# Experts in ascending id, chunks of one expert adjacent (they run concurrently and share weight reads).
# Every token still sees one pass over its expert (the per-row arithmetic does not depend on chunking).
_META2_SRC = r"""
    constexpr int NW = (E + 31) / 32;
    threadgroup uint sidx[P];
    threadgroup uint smask[NW];
    threadgroup int scnt[P];
    const int t = (int)thread_position_in_threadgroup.x;
    if (t < P) sidx[t] = idx[t];
    if (t < NW) smask[t] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < P) atomic_fetch_or_explicit((threadgroup atomic_uint*)&smask[sidx[t] >> 5], 1u << (sidx[t] & 31u),
                                        memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int e = -1, cnt = 0;
    if (t < P) {
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
    }
    int n = 0;
    if (e >= 0)
        for (int p = 0; p < P; p++) n += ((int)sidx[p] == e) ? 1 : 0;
    const int nch = (n + TV - 1) / TV;
    if (t < P) scnt[t] = nch;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= P) return;
    int base = 0;
    for (int j = 0; j < t; j++) base += scnt[j];
    int total = base;
    for (int j = t; j < P; j++) total += scnt[j];
    // slots [total, P): empty
    for (int z = total + t; z < P; z += P) meta[z * (2 + TV)] = -1;
    if (e < 0) return;
    int k = 0, ch = 0;
    for (int p = 0; p < P; p++) {
        if ((int)sidx[p] == e) {
            device int* out = meta + (base + ch) * (2 + TV);
            out[2 + k] = p;
            k++;
            if (k == TV) { out[0] = e; out[1] = TV; k = 0; ch++; }
        }
    }
    if (k > 0) {
        device int* out = meta + (base + ch) * (2 + TV);
        out[0] = e; out[1] = k;
    }
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

# ---------------------------------------------------------------------------------------------
# v2: one pass over each expert's weights for ALL its routed tokens.
# At M = 4..16 rows the TV=1 kernels above re-dequantize every weight word once per routed token
# (n passes); with ~128 routes/layer that is instruction-issue bound (~35 SG-instructions per
# (row, token, 8-value block)). Here each lane dequantizes its words once per block (fp4 byte LUT
# + an fp8 scale LUT built with q_dequantize_scale's own conversion) and applies them to up to TVM
# tokens whose accumulators stay in registers; the per-(token, row) arithmetic is unchanged
# (spec_qdot_d at stock's function boundary, same lane partition, same block order, simd_sum),
# so every row stays bitwise equal to the L=1 decode path. Epilogues are spread over lanes.
_V2_FN = r"""
#ifndef SPEC_PF
#define SPEC_PF 1
#endif
#ifndef SPEC_HD
#define SPEC_HD 1
#endif
#if SPEC_HD
typedef half2 spec_lut_t;   // fp4 values are exact in half; x * float(d) == x * d_f32
typedef half spec_d_t;
#else
typedef float2 spec_lut_t;
typedef float spec_d_t;
#endif
inline void spec_build_luts(threadgroup spec_lut_t* lut, threadgroup float* slut, uint tid, uint nthreads) {
  for (uint b = tid; b < 256; b += nthreads) {
    lut[b] = spec_lut_t(QDequantize<4>{}((uint8_t)(b & 15u)), QDequantize<4>{}((uint8_t)(b >> 4)));
    slut[b] = q_dequantize_scale<float, 16>((uint8_t)b);
  }
}
#ifndef SPEC_ALU
#define SPEC_ALU 0
#endif
template <int PPT>
__attribute__((always_inline)) inline void spec_deq2(const thread uint32_t* wv, thread spec_d_t* d,
                                                     const threadgroup spec_lut_t* lut) {
#pragma unroll
  for (int j = 0; j < PPT; j++)
#pragma unroll
    for (int q = 0; q < 4; q++) {
#if SPEC_ALU && SPEC_HD
      // qfp4_e2m1 for both nibbles of byte q at once: as_type<half>((b & 7) << 9) * 16384, sign from bit 3
      // (sign applied to the bit pattern: -0.0 for b == 8, as MLX's `bits & 8 ? -converted : converted`).
      const uint b = (wv[j] >> (8 * q)) & 255u;
      const uint u = ((b & 7u) << 9) | (((b >> 4) & 7u) << 25) | ((b & 8u) << 12) | ((b & 128u) << 24);
      const half2 dq = as_type<half2>(u) * half2(16384.0h, 16384.0h);
#else
      const spec_lut_t dq = lut[(wv[j] >> (8 * q)) & 255u];
#endif
      d[8 * j + 2 * q] = dq.x;
      d[8 * j + 2 * q + 1] = dq.y;
    }
}
// spec_qdot_d with the dequantized values held as spec_d_t (widened exactly at use).
inline float spec_qdot_h(const thread float* x, const thread spec_d_t* d, float scale, int nv) {
  float accum = 0;
#pragma unroll
  for (int i = 0; i < (nv / 4); i++)
    accum += (x[4 * i] * static_cast<float>(d[4 * i]) + x[4 * i + 1] * static_cast<float>(d[4 * i + 1]) +
              x[4 * i + 2] * static_cast<float>(d[4 * i + 2]) + x[4 * i + 3] * static_cast<float>(d[4 * i + 3]));
  return scale * accum;
}
// NR weight rows (offsets roff) x up to TVM tokens (x rows at xoff[v], v < nv) in one pass.
// result[v * NR + r] += lane partials, in stock fp_qmv[_fast]_impl order.
template <typename T, int K, int GS, int NR, int TVM, bool FAST, typename WP, typename SP, typename XP>
__attribute__((always_inline)) inline void qmv_multi(
    WP wbase, SP sbase, const thread int* roff, const thread int* xoff, int nv, XP x, int lane,
    thread float* result, const threadgroup spec_lut_t* lut, const threadgroup float* slut) {
  constexpr int PF = 8;
  constexpr int PPT = FAST ? 2 : 1;
  constexpr int VPT = PF * PPT;
  constexpr int BLOCK = VPT * 32;
  constexpr int KW = K / PF;
  constexpr int KG = K / GS;
  static_assert(GS == 16, "nvfp4");
  static_assert(K % 8 == 0, "K % 8");
  // 32-bit offsets (no per-row / per-token 64-bit pointers held in registers).
  int wo[NR], so[NR], xo[TVM];
#pragma unroll
  for (int r = 0; r < NR; r++) {
    wo[r] = roff[r] * KW + lane * PPT;
    so[r] = roff[r] * KG + (lane * VPT) / GS;
  }
#pragma unroll
  for (int v = 0; v < TVM; v++) xo[v] = xoff[v] + lane * VPT;
  int k = 0;
#if SPEC_PF
  // Software pipelining: the next block's weight words and scale bytes are loaded before
  // this block's math (loads only; the arithmetic is unchanged).
  // Guarded like the in-loop prefetch: a lane past K (K < BLOCK, e.g. tiny NH=128) would read the next
  // row, or past the buffer on the last row. Its values are never used.
  uint32_t wn[NR][PPT];
  uint sn[NR];
  const bool first = lane * VPT < K;
#pragma unroll
  for (int r = 0; r < NR; r++) {
#pragma unroll
    for (int j = 0; j < PPT; j++) wn[r][j] = first ? wbase[wo[r] + j] : 0u;
    sn[r] = first ? sbase[so[r]] : 0u;
  }
#endif
  for (; FAST ? (k < K) : (k < K - BLOCK); k += BLOCK) {
    spec_d_t d[NR][VPT];
    float s[NR];
#if SPEC_PF
    const bool more = k + BLOCK + lane * VPT < K;
#endif
#pragma unroll
    for (int r = 0; r < NR; r++) {
      uint32_t wv[PPT];
#if SPEC_PF
#pragma unroll
      for (int j = 0; j < PPT; j++) wv[j] = wn[r][j];
      const uint sc = sn[r];
      wo[r] += BLOCK / PF;
      so[r] += BLOCK / GS;
      if (more) {
#pragma unroll
        for (int j = 0; j < PPT; j++) wn[r][j] = wbase[wo[r] + j];
        sn[r] = sbase[so[r]];
      }
      spec_deq2<PPT>(wv, d[r], lut);
      s[r] = slut[sc];
#else
#pragma unroll
      for (int j = 0; j < PPT; j++) wv[j] = wbase[wo[r] + j];
      spec_deq2<PPT>(wv, d[r], lut);
      s[r] = slut[sbase[so[r]]];
      wo[r] += BLOCK / PF;
      so[r] += BLOCK / GS;
#endif
    }
#pragma unroll
    for (int v = 0; v < TVM; v++) {
      if (v < nv) {
        float xt[VPT];
#pragma unroll
        for (int h = 0; h < PPT; h++) spec_load8(x + xo[v] + 8 * h, xt + 8 * h);
#pragma unroll
        for (int r = 0; r < NR; r++) result[v * NR + r] += spec_qdot_h(xt, d[r], s[r], VPT);
      }
      xo[v] += BLOCK;
    }
  }
  if (!FAST) {
    // rem = clamp(K - k - lane*8, 0, 8) is 0 or 8: qdot_safe(N = 8) == the full-block expression.
    if (K - k - lane * VPT > 0) {
      spec_d_t d[NR][VPT];
      float s[NR];
#pragma unroll
      for (int r = 0; r < NR; r++) {
#if SPEC_PF
        uint32_t wv[1] = {wn[r][0]};
        spec_deq2<1>(wv, d[r], lut);
        s[r] = slut[sn[r]];
#else
        uint32_t wv[1] = {wbase[wo[r]]};
        spec_deq2<1>(wv, d[r], lut);
        s[r] = slut[sbase[so[r]]];
#endif
      }
#pragma unroll
      for (int v = 0; v < TVM; v++) {
        if (v < nv) {
          float xt[VPT];
          spec_load8(x + xo[v], xt);
#pragma unroll
          for (int r = 0; r < NR; r++) result[v * NR + r] += spec_qdot_h(xt, d[r], s[r], VPT);
        }
      }
    }
  }
}
"""

_PROLOGUE2 = r"""
    threadgroup spec_lut_t lut[256];
    threadgroup float slut[256];
    const int lane = (int)thread_index_in_simdgroup;
    const int sg = (int)simdgroup_index_in_threadgroup;
    if (meta[(int)threadgroup_position_in_grid.z * (2 + RMAX)] < 0) return;  // uniform
    spec_build_luts(lut, slut, (uint)(sg * 32 + lane), (uint)(SGS * 32));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int zs = (int)threadgroup_position_in_grid.z; zs < P; zs += ZS) {
    auto mz = meta + zs * (2 + RMAX);
    const int e = mz[0];
    if (e < 0) break;
    const int n = mz[1];
    auto plist = mz + 2;
    const int ybeg = (int)threadgroup_position_in_grid.y * RPC;
    constexpr int KW = K / 8;
    constexpr int KG = K / GS;
"""

# act[p, :] = SoftcappedSwiGLU(gate, up) for route p = row * KT + slot; RG gate rows (+ RG up rows)
# per simdgroup, tokens in chunks of TVM (one weight pass per chunk).
_GUA2_SRC = _PROLOGUE2 + r"""
    auto wgb = wg + (size_t)e * WR * KW;
    auto sgb = sgs + (size_t)e * WR * KG;
    auto wub = wu + (size_t)e * WR * KW;
    auto sub = sus + (size_t)e * WR * KG;
    const float ig = spec_inv_scale<SMG>(gsg, GNG, e);
    const float iu = spec_inv_scale<SMU>(gsu, GNU, e);
    const float gc = caps[0], uc = caps[1];
    for (int row0 = ybeg + sg * RG; row0 < ybeg + RPC; row0 += SGS * RG) {
    int ro2[2 * RG];
#pragma unroll
    for (int r = 0; r < RG; r++) { ro2[r] = row0 + r; ro2[RG + r] = UOFF + row0 + r; }
    for (int t0 = 0; t0 < n; t0 += TVM) {
        const int nv = min(n - t0, TVM);
        int xoff[TVM];
#pragma unroll
        for (int v = 0; v < TVM; v++) xoff[v] = v < nv ? (plist[t0 + v] / KT) * K : 0;
        float res[TVM * 2 * RG];
        float ru[TVM * RG];
#pragma unroll
        for (int i = 0; i < TVM * 2 * RG; i++) res[i] = 0;
#pragma unroll
        for (int i = 0; i < TVM * RG; i++) ru[i] = 0;
        if (SAMEW) {
            qmv_multi<T, K, GS, 2 * RG, TVM, FAST>(wgb, sgb, ro2, xoff, nv, x, lane, res, lut, slut);
        } else {
            qmv_multi<T, K, GS, RG, TVM, FAST>(wgb, sgb, ro2, xoff, nv, x, lane, res, lut, slut);
            qmv_multi<T, K, GS, RG, TVM, FAST>(wub, sub, ro2 + RG, xoff, nv, x, lane, ru, lut, slut);
        }
        // Lane j (< nv*RG) takes output (v, r) = (j / RG, j % RG).
        float mg = 0, mu = 0;
#pragma unroll
        for (int v = 0; v < TVM; v++) {
            if (v < nv) {
#pragma unroll
                for (int r = 0; r < RG; r++) {
                    const float sgv = simd_sum(SAMEW ? res[v * 2 * RG + r] : res[v * RG + r]);
                    const float suv = simd_sum(SAMEW ? res[v * 2 * RG + RG + r] : ru[v * RG + r]);
                    if (lane == v * RG + r) { mg = sgv; mu = suv; }
                }
            }
        }
        if (lane < nv * RG) {
            const int v = lane / RG, r = lane % RG;
            const int p = plist[t0 + v];
            const float g = static_cast<float>(static_cast<T>(SMG ? mg * ig : mg));
            const float u = static_cast<float>(static_cast<T>(SMU ? mu * iu : mu));
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
    }
"""

# y[p, :] = act[p, :] @ W[e].T for the routes p of each distinct expert; RD rows per simdgroup,
# tokens in chunks of TVM (one weight pass per chunk); stores spread over lanes.
_DOWN2_SRC = _PROLOGUE2 + r"""
    auto wb = w + (size_t)e * NOUT * KW;
    auto sb = scales + (size_t)e * NOUT * KG;
    const float inv = spec_inv_scale<SM>(gs, GN, e);
    for (int row0 = ybeg + sg * RD; row0 < ybeg + RPC; row0 += SGS * RD) {
    int roff[RD];
#pragma unroll
    for (int r = 0; r < RD; r++) roff[r] = row0 + r;
    for (int t0 = 0; t0 < n; t0 += TVM) {
        const int nv = min(n - t0, TVM);
        int xoff[TVM];
#pragma unroll
        for (int v = 0; v < TVM; v++) xoff[v] = v < nv ? plist[t0 + v] * K : 0;
        float res[TVM * RD];
#pragma unroll
        for (int i = 0; i < TVM * RD; i++) res[i] = 0;
        qmv_multi<T, K, GS, RD, TVM, FAST>(wb, sb, roff, xoff, nv, act, lane, res, lut, slut);
        float mine = 0;
#pragma unroll
        for (int v = 0; v < TVM; v++) {
            if (v < nv) {
#pragma unroll
                for (int r = 0; r < RD; r++) {
                    const float rr = simd_sum(res[v * RD + r]);
                    if (lane == v * RD + r) mine = rr;
                }
            }
        }
        if (lane < nv * RD) {
            const int v = lane / RD, r = lane % RD;
            y[(size_t)plist[t0 + v] * NOUT + row0 + r] = static_cast<T>(SM ? mine * inv : mine);
        }
    }
    }
    }
"""

# route_topk (fast_kernels._ROUTE_SRC, per row unchanged) for all R rows + the meta kernel, one launch:
# one threadgroup of NT threads; thread t handles (row, expert) pairs t, t+NT, ... (R*E <= 4096).
_ROUTE_META_SRC = r"""
    threadgroup float keys[R * E];
    threadgroup float sc[R * E];
    threadgroup float sel_s[R * KT];
    threadgroup uint sidx[R * KT];
    threadgroup float denom_tg[R];
    constexpr int NW = (E + 31) / 32;
    threadgroup uint smask[NW];
    const int t = (int)thread_position_in_threadgroup.x;
    for (int i = t; i < R * E; i += NT) {
        const int row = i / E, ex = i % E;
        float xl = logits[(size_t)row * E + ex];
        float y = 1 / (1 + metal::precise::exp(metal::abs(xl)));
        const float s = (xl < 0) ? y : 1 - y;
        sc[i] = s;
        keys[i] = -(s + bias[ex]);
    }
    if (t < NW) smask[t] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < R * E; i += NT) {
        const int row = i / E, ex = i % E;
        const float kt = keys[i];
        const bool tn = isnan(kt);
        uint rank = 0;
        for (int j = 0; j < E; ++j) {
            float kj = keys[row * E + j];
            bool jn = isnan(kj);
            bool before = tn ? (jn ? j < ex : true) : (!jn && (kj < kt || (kj == kt && j < ex)));
            rank += before ? 1u : 0u;
        }
        if (rank < (uint)KT) { sel_s[row * KT + rank] = sc[i]; sidx[row * KT + rank] = (uint)ex; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < R) {
        float total = 0.0f;
        for (int k = 0; k < KT; ++k) total = total + sel_s[t * KT + k];
        denom_tg[t] = total + 1e-20f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < P; i += NT) {
        const int row = i / KT;
        idx[i] = sidx[i];
        w[i] = sel_s[i] / denom_tg[row] * scale[0];
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&smask[sidx[i] >> 5], 1u << (sidx[i] & 31u),
                                 memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int z = t; z < P; z += NT) {
        int e = -1, cnt = 0;
        for (int wd = 0; wd < NW && e < 0; wd++) {
            const uint m = smask[wd];
            const int c = (int)popcount(m);
            if (z < cnt + c) {
                uint mm = m;
                for (int j = 0; j < z - cnt; j++) mm &= mm - 1u;
                e = wd * 32 + (int)ctz(mm);
            }
            cnt += c;
        }
        device int* out = meta + z * (2 + R);
        out[0] = e;
        if (e < 0) continue;
        int n = 0;
        for (int p = 0; p < P; p++)
            if ((int)sidx[p] == e) out[2 + n++] = p;
        out[1] = n;
    }
"""

# language._combine_routes in one launch: out[row, c] = T(sum_k f32(y[row*KT+k, c]) * w[row*KT+k]),
# products rounded to f32 then summed in k order from 0 (MLX multiply kernel + sequential col reduce).
_COMBINE_SRC = r"""
    const int c = (int)thread_position_in_grid.x;
    const int row = (int)thread_position_in_grid.y;
    if (c >= D) return;
    float acc = 0.0f;
    for (int k = 0; k < KT; k++) {
        const float pr = static_cast<float>(y[(size_t)(row * KT + k) * D + c]) * w[row * KT + k];
        acc = spec_add_nc(acc, pr);
    }
    out[(size_t)row * D + c] = static_cast<T>(acc);
"""
_COMBINE_HDR = r"""
#pragma clang fp contract(off)
inline float spec_add_nc(float a, float b) { return a + b; }
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
V2 = int(os.environ.get("INKLING_SMOE_V2", "1"))  # one-pass multi-token kernels
G2_RG = int(os.environ.get("INKLING_SMOE_G2_RG", "2"))
G2_SGS = int(os.environ.get("INKLING_SMOE_G2_SGS", "4"))
G2_TV = int(os.environ.get("INKLING_SMOE_G2_TV", "2"))
D2_RD = int(os.environ.get("INKLING_SMOE_D2_RD", "4"))
D2_SGS = int(os.environ.get("INKLING_SMOE_D2_SGS", "2"))
D2_TV = int(os.environ.get("INKLING_SMOE_D2_TV", "2"))
RM = int(os.environ.get("INKLING_SMOE_RM", "0"))  # fused route_topk+meta: slower at B=16 (1 TG)
RT2 = int(os.environ.get("INKLING_SMOE_RT2", "1"))  # router_rows2 (stored-dtype weights, RR rows/TG)
F32IO = int(os.environ.get("INKLING_SMOE_F32IO", "0"))  # v2: x, act as exact f32 (no per-token bf16 unpack): no gain, off
MC = int(os.environ.get("INKLING_SMOE_MC", "0"))  # v2: meta2 token chunks of G2_TV per TG slot (no gain; off)
ROUTER_RR = int(os.environ.get("INKLING_SMOE_ROUTER_RR", "4"))
G2_CH = int(os.environ.get("INKLING_SMOE_G2_CH", "1"))  # row tiles per threadgroup (amortizes the LUT build)
D2_CH = int(os.environ.get("INKLING_SMOE_D2_CH", "1"))
CMB = int(os.environ.get("INKLING_SMOE_CMB", "1"))  # one-launch combine  # fused route_topk + meta (one launch)

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


# Router logits for R rows, RR rows per threadgroup (weights loaded once for RR rows, all K-loads of a
# thread issued up front), reading the router weight in its own dtype (bf16 -> f32 widening is exact, so
# the values equal the cached f32 copy). Per (row, expert): exactly mega_kernels._ROUTER_SRC's sequence
# (same per-thread K positions and FMA expression, shuffle_down tree, then SG partials summed in order).
_ROUTER2_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const int out_row = (int)threadgroup_position_in_grid.x * 4;
    const int r0 = (int)threadgroup_position_in_grid.y * RR;
    threadgroup float tgp[RR * 64];
    const device WT* mat = w + (size_t)out_row * K;
    float result[RR][4];
#pragma unroll
    for (int r = 0; r < RR; r++)
#pragma unroll
        for (int tm = 0; tm < 4; tm++) result[r][tm] = 0;
    const int bn0 = ((int)sg * 32 + (int)lane) * 4;
    constexpr int NIT = K / 1024;
    float inter[NIT + 1][4][4];
#pragma unroll
    for (int it = 0; it < NIT; ++it)
#pragma unroll
        for (int tm = 0; tm < 4; tm++)
#pragma unroll
            for (int t = 0; t < 4; t++) inter[it][tm][t] = static_cast<float>(mat[tm * K + bn0 + it * 1024 + t]);
    const int bnr = bn0 + NIT * 1024;
    const bool full = bnr + 4 <= K;
    if (K % 1024 > 0) {
#pragma unroll
        for (int tm = 0; tm < 4; tm++)
#pragma unroll
            for (int t = 0; t < 4; t++)
                inter[NIT][tm][t] = (full || bnr + t < K) ? static_cast<float>(mat[tm * K + bnr + t]) : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < RR; r++) {
        if (r0 + r < R) {
            const device T* xr = x + (size_t)(r0 + r) * K;
#pragma unroll
            for (int it = 0; it < NIT; ++it) {
                float vc[4];
#pragma unroll
                for (int t = 0; t < 4; t++) vc[t] = static_cast<float>(xr[bn0 + it * 1024 + t]);
#pragma unroll
                for (int tm = 0; tm < 4; tm++)
#pragma unroll
                    for (int t = 0; t < 4; t++) result[r][tm] += inter[it][tm][t] * vc[t];
            }
            if (K % 1024 > 0) {
                float vc[4];
#pragma unroll
                for (int t = 0; t < 4; t++) vc[t] = (full || bnr + t < K) ? static_cast<float>(xr[bnr + t]) : 0.0f;
#pragma unroll
                for (int tm = 0; tm < 4; tm++)
#pragma unroll
                    for (int t = 0; t < 4; t++) result[r][tm] += inter[NIT][tm][t] * vc[t];
            }
        }
#pragma unroll
        for (int tm = 0; tm < 4; tm++)
            for (ushort s = 16; s >= 1; s >>= 1) result[r][tm] += simd_shuffle_down(result[r][tm], s);
        if (lane == 0)
#pragma unroll
            for (int tm = 0; tm < 4; tm++) tgp[r * 64 + sg * 8 + tm] = result[r][tm];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)lid < RR && r0 + (int)lid < R) {
        const int r = (int)lid;
        float acc[4];
        for (int tm = 0; tm < 4; tm++) acc[tm] = tgp[r * 64 + tm];
        for (int sgn = 1; sgn < 8; sgn++)
            for (int tm = 0; tm < 4; tm++) acc[tm] += tgp[r * 64 + sgn * 8 + tm];
        for (int tm = 0; tm < 4; tm++) logits[(size_t)(r0 + r) * E + out_row + tm] = acc[tm];
    }
#if XO
    if (threadgroup_position_in_grid.x == 0)
        for (int r = 0; r < RR; r++)
            if (r0 + r < R)
                for (int i = (int)lid; i < K; i += 256)
                    x32[(size_t)(r0 + r) * K + i] = static_cast<float>(x[(size_t)(r0 + r) * K + i]);
#endif
"""


def router_rows2(x2, w, x32=False):
    """router_rows with the router weight in its stored dtype (bf16/f16/f32), RR rows per threadgroup.
    x32=True also returns x2 widened to f32 (exact), written by the first expert-block threadgroups."""
    R, K = x2.shape
    E = w.shape[0]
    rr = min(R, ROUTER_RR)
    k = _kernel("router2x" if x32 else "router2", ["x", "w"], ["logits", "x32"] if x32 else ["logits"],
                ("#define XO 1\n" if x32 else "#define XO 0\n") + _ROUTER2_SRC, "")
    out = k(
        inputs=[x2, w],
        template=[("T", x2.dtype), ("WT", w.dtype), ("K", K), ("E", E), ("R", R), ("RR", rr)],
        grid=(256 * (E // 4), (R + rr - 1) // rr, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, E), (R, K)] if x32 else [(R, E)],
        output_dtypes=[mx.float32, mx.float32] if x32 else [mx.float32],
    )
    return (out[0], out[1]) if x32 else out[0]


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
    if V2:
        if (NH % (G2_RG * G2_SGS * G2_CH) or D % (D2_RD * D2_SGS * D2_CH) or G2_TV * G2_RG > 32
                or D2_TV * D2_RD > 32):
            return False
        return fused is None or (fused.get("mode") == "nvfp4" and fused.get("n") == NH)
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
    gw = moe.gate_weight
    f32io = bool(V2 and F32IO and x.dtype != mx.float32)
    xg = x2
    if RT2 and gw.dtype in (mx.bfloat16, mx.float16, mx.float32) and gw.shape == w32.shape:
        if f32io:
            logits, xg = router_rows2(x2, gw, x32=True)
        else:
            logits = router_rows2(x2, gw)
    else:
        logits = router_rows(x2, w32)
        if f32io:
            xg = x2.astype(mx.float32)
    if stages == "upto_logits":
        return logits
    zs = max(1, min(P, E, int(math.ceil(P * ZFRAC))))
    if RM and R * E <= 2048:
        nt = min(1024, max(64, (R * E + 31) // 32 * 32))
        idx, wts, meta = _kernel("route_meta", ["logits", "bias", "scale"], ["idx", "w", "meta"],
                                 _ROUTE_META_SRC, "")(
            inputs=[logits, moe.e_score_correction_bias.astype(mx.float32), fk._scalar(moe.route_scale)],
            template=[("E", E), ("KT", KT), ("R", R), ("P", P), ("NT", nt)],
            grid=(nt, 1, 1), threadgroup=(nt, 1, 1),
            output_shapes=[(P,), (P,), (P * (2 + R),)],
            output_dtypes=[mx.uint32, mx.float32, mx.int32])
        if stages in ("upto_route", "upto_meta"):
            return meta
    else:
        meta = None
        idx, wts = fk.route_topk(logits, moe.e_score_correction_bias, KT, moe.route_scale)
        idx = idx.reshape(P)
    if stages == "upto_route":
        return idx
    mt = max(32, (P + 31) // 32 * 32)
    rmax = R
    if V2 and MC and G2_TV == D2_TV and meta is None:
        rmax, zs = G2_TV, P
        meta = _kernel("meta2", ["idx"], ["meta"], _META2_SRC, "")(
            inputs=[idx], template=[("E", E), ("P", P), ("TV", G2_TV)],
            grid=(mt, 1, 1), threadgroup=(mt, 1, 1),
            output_shapes=[(P * (2 + G2_TV),)], output_dtypes=[mx.int32])[0]
    meta = meta if meta is not None else _kernel("meta", ["idx"], ["meta"], _META_SRC, _QMV_HEADER + _SPEC_FN)(
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
    if V2:
        gua2 = _kernel(
            "gua2", ["meta", "x", "wg", "sgs", "gsg", "wu", "sus", "gsu", "caps"], ["act"],
            _GUA2_SRC, _perf_hdr + _QMV_HEADER + _SPEC_FN + _V2_FN,
        )
        gch2 = NH // (G2_RG * G2_SGS * G2_CH)
        act = gua2(
            inputs=[meta, xg, wg, sg, gsg, wu, su, gsu, caps],
            template=[("T", x.dtype), ("K", D), ("NOUT", NH), ("E", E), ("KT", KT), ("P", P),
                      ("RMAX", rmax), ("GS", 16), ("RG", G2_RG), ("SGS", G2_SGS), ("TVM", G2_TV),
                      ("FAST", _fast_variant(D)), ("WR", wr), ("UOFF", uoff), ("SAMEW", samew),
                      ("SMG", smg), ("SMU", smu), ("GNG", gng), ("GNU", gnu), ("RPC", G2_RG * G2_SGS * G2_CH),
                      ("ZS", zs)],
            grid=(32, G2_SGS * gch2, zs),
            threadgroup=(32, G2_SGS, 1),
            output_shapes=[(P, NH)],
            output_dtypes=[mx.float32 if f32io else x.dtype],
        )[0]
    else:
        act = _gua_v1(meta, x, x2, wg, sg, gsg, wu, su, gsu, caps, D, NH, E, KT, P, R, wr, uoff,
                      samew, smg, smu, gng, gnu, gch, zs)
    if stages == "upto_gua":
        return act
    if stages == "act_in":
        act = stages_act
    smd, gsd, gnd = _scale_mode(dn)
    if V2:
        down2 = _kernel(
            "down2", ["meta", "act", "w", "scales", "gs"], ["y"],
            _DOWN2_SRC, _perf_hdr + _QMV_HEADER + _SPEC_FN + _V2_FN,
        )
        dch2 = D // (D2_RD * D2_SGS * D2_CH)
        y = down2(
            inputs=[meta, act, dn.weight, dn.scales, gsd],
            template=[("T", x.dtype), ("K", NH), ("NOUT", D), ("E", E), ("P", P), ("RMAX", rmax),
                      ("GS", 16), ("RD", D2_RD), ("SGS", D2_SGS), ("TVM", D2_TV),
                      ("FAST", _fast_variant(NH)), ("SM", smd), ("GN", gnd), ("RPC", D2_RD * D2_SGS * D2_CH),
                      ("ZS", zs)],
            grid=(32, D2_SGS * dch2, zs),
            threadgroup=(32, D2_SGS, 1),
            output_shapes=[(P, D)],
            output_dtypes=[x.dtype],
        )[0]
    else:
        y = _down_v1(meta, act, dn, gsd, smd, gnd, x, D, NH, E, P, R, dch, zs)
    return _tail(stages, idx, logits, wts, act, y, B, L, KT, D, E, P, R, routed)


def _gua_v1(meta, x, x2, wg, sg, gsg, wu, su, gsu, caps, D, NH, E, KT, P, R, wr, uoff, samew,
            smg, smu, gng, gnu, gch, zs):
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
    return act


def _down_v1(meta, act, dn, gsd, smd, gnd, x, D, NH, E, P, R, dch, zs):
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
    return y


def _tail(stages, idx, logits, wts, act, y, B, L, KT, D, E, P, R, routed):
    from .language import _combine_routes

    if stages in ("act_in", "upto_down"):
        return y
    if stages:
        probe = _kernel("probe", ["idx"], ["out"], _PROBE_SRC, _QMV_HEADER + _SPEC_FN)(
            inputs=[idx], template=[("E", E), ("P", P), ("RMAX", R)],
            grid=(32, 1, P), threadgroup=(32, 1, 1),
            output_shapes=[(P, 2 + R)], output_dtypes=[mx.int32])[0]
        return logits, idx, wts, act, y, probe
    if routed:  # per-route outputs + weights, as SparseMoE._routed (sconv_tail combines)
        return y.reshape(B, L, KT, D), wts.reshape(B, L, KT)
    if CMB:
        out = _kernel("combine", ["y", "w"], ["out"], _COMBINE_SRC, _COMBINE_HDR)(
            inputs=[y, wts.reshape(P)], template=[("T", y.dtype), ("D", D), ("KT", KT)],
            grid=((D + 255) // 256 * 256, R, 1), threadgroup=(256, 1, 1),
            output_shapes=[(R, D)], output_dtypes=[y.dtype])[0]
        return out.reshape(B, L, D)
    return _combine_routes(y.reshape(B, L, KT, D), wts.reshape(B, L, KT))
