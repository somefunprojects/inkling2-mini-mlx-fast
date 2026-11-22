# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for MLX from Transformers' Inkling2 implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2026 somefunprojects: fast decode, batch and speculative-verify
# paths (see fast_decode.py and friends). Licensed under the Apache License, Version 2.0, as above.


import math

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import (
    ArraysCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from ..inkling.language import InklingShortConvolution
from ..inkling.language import LanguageModel as InklingLanguageModel
from ..inkling.language import (
    _cache_padding_mask,
    _next_conv_state,
    banded_additive_mask,
)
from ..linear import DECODE_BLOCK_SIZE as _DECODE_BLOCK_SIZE
from ..switch_layers import SwitchGLU
from . import fast_kernels as _fast
from .config import TextConfig, validate_sharding

_FAST_GLUE = _fast.enabled("glue")
_FAST_FUSEMOE = _fast.enabled("fusemoe")
_FAST_ROUTE = _fast.enabled("route")
_FAST_MOESORT = _fast.enabled("moesort")
_FAST_CTRLBATCH = _fast.enabled("ctrlbatch")
_FAST_MSGPACK = _fast.enabled("msgpack")
_FAST_QKVCLIP = _fast.enabled("qkvclip")
_FAST_ATTNRES = _fast.enabled("attnres")
# xpass = cached block keys + cross-pass depth-logit reuse; xpass_keys = keys only.
_FAST_XPASS_LOGITS = _fast.enabled("xpass")
_FAST_XPASS = _FAST_XPASS_LOGITS or _fast.enabled("xpass_keys")
# "mask" (bitwise vs stock, decode L=1): one fused mask+sink kernel, and the
# stock [temporal keys; zero sink row] K/V built without the two stock copies:
# a view of the cache buffer when its next slot is free (zeroed), else a
# single concatenate. SDPA sees exactly the stock operands.
# Opt-in, NOT bitwise (SDPA reduction order changes; see MASKFUSE_NOTES):
# mask_ring: physical ring order (no _temporal_order copy once wrapped).
# mask_sink: SDPA sinks= for B=1 (sink enters the online softmax first, in
# simdgroup/block 0, instead of as the last key).
_FAST_MASK_EXACT = _fast.enabled("mask")
_FAST_MASK_RING = _fast.enabled("mask_ring")
_FAST_MASK_SINK = _fast.enabled("mask_sink")
_FAST_MASK_ANY = _FAST_MASK_RING or _FAST_MASK_SINK
_FAST_ATTN = (
    _FAST_MASK_ANY
    or _FAST_MASK_EXACT
    or _fast.enabled("fuse")
    or _fast.enabled("glue")
    or _FAST_QKVCLIP
)


def _exact_sink_kv(kv, k, v):
    """Stock decode K/V operands: concat([temporal(k), 0], axis=2), bitwise.

    ``k``/``v`` come from ``kv.update_and_fetch`` (KVCache or keep=0
    RotatingKVCache). Returns (k_ext, v_ext) with S + 1 rows.
    """
    b, h, size, d = k.shape
    if type(kv) is RotatingKVCache and kv._idx != size:
        # Wrapped ring: one copy instead of _temporal_order + zero concat.
        i = kv._idx
        zk = mx.zeros((b, h, 1, d), k.dtype)
        zv = mx.zeros((b, h, 1, v.shape[3]), v.dtype)
        return (
            mx.concatenate([k[..., i:, :], k[..., :i, :], zk], axis=2),
            mx.concatenate([v[..., i:, :], v[..., :i, :], zv], axis=2),
        )
    # Storage order is temporal and k == kv.keys[..., :size, :].
    if kv.keys.shape[2] > size:
        # The next free slot becomes the zero sink row (the next update
        # overwrites it); no copy of the cache.
        kv.keys[..., size : size + 1, :] = 0
        kv.values[..., size : size + 1, :] = 0
        return kv.keys[..., : size + 1, :], kv.values[..., : size + 1, :]
    zk = mx.zeros((b, h, 1, d), k.dtype)
    zv = mx.zeros((b, h, 1, v.shape[3]), v.dtype)
    return mx.concatenate([k, zk], axis=2), mx.concatenate([v, zv], axis=2)


@mx.compile
def l2_clip(x, threshold, scale=None):
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


@mx.compile
def _softcapped_swiglu(up, gate, gate_cap, up_cap):
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    return (
        mx.tanh(gate / gate_cap)
        * gate_cap
        * mx.sigmoid(gate)
        * mx.tanh(up / up_cap)
        * up_cap
    ).astype(dtype)


@mx.compile
def _route(x, weight, bias, top_k, route_scale):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


@mx.compile
def _combine_routes(out, weights):
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


@mx.compile
def _attention_residual(states, query, normalize_values):
    states = states.astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (keys @ query.reshape(-1, states.shape[-1]).T) * states.shape[-1] ** -0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ (keys if normalize_values else states)


_conv_history_kernel = mx.fast.metal_kernel(
    name="inkling_masked_conv_history",
    input_names=["state", "inputs", "mask"],
    output_names=["out"],
    ensure_row_contiguous=False,
    source=r"""
    uint index = thread_position_in_grid.x;
    int B = inputs_shape[0], L = inputs_shape[1], C = inputs_shape[2];
    int W = state_shape[1];
    if (index >= B * L * W * C) return;
    int c = index % C, w = (index / C) % W;
    int t = (index / (C * W)) % L, b = index / (C * W * L);
    int count = 0, start = 0;
    for (int j = 0; j <= t; ++j) {
        if (mask[b * mask_strides[0] + j * mask_strides[1]]) {
            if (count == 0) start = j;
            ++count;
        }
    }
    int logical = count - W + w;
    out[index] = logical < 0
        ? state[b * state_strides[0] + (W + logical) * state_strides[1]
                + c * state_strides[2]]
        : inputs[b * inputs_strides[0] + (start + logical) * inputs_strides[1]
                 + c * inputs_strides[2]];
    """,
)


@mx.compile
def _conv_state_history_fallback(state, inputs, mask):
    width = state.shape[1]
    count = mx.cumsum(mask.astype(mx.int32), axis=1)
    start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(width)[None, None, :] + count[:, :, None] - width
    indices = mx.where(
        logical >= 0,
        width + start[:, None, None] + mx.maximum(logical, 0),
        width + logical,
    )
    combined = mx.concatenate([state, inputs], axis=1)
    return mx.take_along_axis(combined[:, None], indices[..., None], axis=2)


def _conv_state_history(state, inputs, mask):
    """Select every masked prefix's state without repeated prefix forwards."""
    batch, length, channels = inputs.shape
    width = state.shape[1]
    if (
        width > 0
        and length <= 8
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        return _conv_history_kernel(
            inputs=[state, inputs, mask],
            grid=(batch * length * width * channels, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, length, width, channels)],
            output_dtypes=[state.dtype],
        )[0]
    return _conv_state_history_fallback(state, inputs, mask)


def _update_conv_cache(cache, index, state, inputs, mask):
    if cache is None:
        return
    speculating = getattr(cache, "is_speculating", False)
    if speculating and mask is not None:
        # Physical prefixes can retain different numbers of valid tokens.
        # Avoid rebuilding a separate reduction, concat and gather per prefix.
        states = _conv_state_history(state, inputs, mask)
        final = states[:, -1]
        cache.record_speculative_states(
            index, states[:, :-1] if inputs.shape[1] > 1 else None, final
        )
    else:
        final = _next_conv_state(state, inputs, mask)
        if speculating:
            cache.record_speculative_window(
                index, mx.concatenate([state, inputs], axis=1), state.shape[1]
            )
    cache[index] = final


class ShortConvolution(InklingShortConvolution):
    def __call__(self, x, cache=None, mask=None):
        # Keep the convolution, residual, and scaling in FP32. The checkpoint
        # stores BF16 taps, but rounding the convolution before adding the
        # residual changes both reference parity and prefill/decode consistency.
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]))
        if (
            not self.training
            and x.dtype == mx.bfloat16
            and self.kernel_size == 4
            and x.shape[0] <= 4
            and x.shape[1] <= 8
            and x.shape[-1] % 128 == 0
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            from .kernels import short_convolution

            if (
                _FAST_GLUE
                and mask is None
                and not getattr(cache, "is_speculating", False)
            ):
                # One kernel writes both the output and the next FP32 state
                # (stock: concat + slice, then the convolution kernel).
                out, nstate = _fast.short_convolution_with_state(
                    state, self.conv.weight, x, self.kernel_size
                )
                if cache is not None:
                    cache[self.conv_idx] = nstate
                return out
            # Temporal selection promotes selected BF16 values into FP32 state.
            # Invalid positions never enter that state. The convolution itself
            # reads masked inputs directly, without materializing a concat.
            _update_conv_cache(cache, self.conv_idx, state, x, mask)
            return short_convolution(state, self.conv.weight, x, mask, self.kernel_size)
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        context = mx.concatenate([state, inputs], axis=1)
        _update_conv_cache(cache, self.conv_idx, state, inputs, mask)
        out = mx.conv1d(
            context, self.conv.weight.astype(mx.float32), groups=x.shape[-1]
        )
        return ((out + xf) / math.sqrt(self.kernel_size + 1)).astype(x.dtype)


class DynamicShortConvolution(nn.Module):
    def __init__(self, config, channels, num_heads, conv_idx):
        super().__init__()
        self.kernel_size = config.sconv_kernel_size
        self.num_heads = num_heads
        self.conv_idx = conv_idx
        self.static_weight = mx.zeros((self.kernel_size, channels))
        self.weight_proj = nn.Linear(
            config.hidden_size, num_heads * self.kernel_size, bias=False
        )

    def __call__(self, x, dynamic_input, cache=None, mask=None):
        b, length, channels = x.shape
        k = self.kernel_size
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((b, k - 1, channels), mx.float32)
        context = mx.concatenate([state, inputs], axis=1)
        _update_conv_cache(cache, self.conv_idx, state, inputs, mask)
        dynamic = (
            self.weight_proj(dynamic_input)
            .astype(mx.float32)
            .reshape(b, length, self.num_heads, 1, k)
        )
        weights = self.static_weight.T.reshape(self.num_heads, -1, k) + dynamic
        out = mx.zeros_like(xf).reshape(b, length, self.num_heads, -1)
        for lag in range(k):
            values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
            out = out + values * weights[..., lag]
        return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = self.sliding or config.rel_extent
        self.sharding_group = None
        self.head_start = 0
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.r_proj = nn.Linear(
            config.hidden_size, self.n_heads * config.d_rel, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.sink_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.rel_scale = mx.ones((self.n_heads, config.d_rel))
        self.rel_proj = mx.zeros((config.d_rel, self.rel_extent))
        for idx, name in enumerate(("k_sconv", "v_sconv")):
            channels = self.n_kv * self.head_dim
            self[name] = (
                ShortConvolution(channels, config.sconv_kernel_size, idx)
                if self.is_sliding
                else DynamicShortConvolution(config, channels, self.n_kv, idx)
            )

    def shard(self, group):
        """Split complete heads and output rows without splitting dot products."""
        if self.sharding_group is not None:
            return
        projections = ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "o_proj")
        # Preserve the dense kernels' output-row tiles. Other configurations
        # retain replicated attention while the experts are still partitioned.
        if (
            group.size() != 2
            or self.n_heads % 2
            or self.n_kv % 2
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or any(
                type(self[name]) is not nn.Linear
                or "bias" in self[name]
                or self[name].weight.shape[0] % 256
                for name in projections
            )
            or (
                not self.is_sliding
                and any(
                    type(self[name].weight_proj) is not nn.Linear
                    or "bias" in self[name].weight_proj
                    for name in ("k_sconv", "v_sconv")
                )
            )
        ):
            return
        rank = group.rank()
        for name in projections:
            projection = self[name]
            width = projection.weight.shape[0] // 2
            projection.weight = mx.contiguous(
                projection.weight[rank * width : (rank + 1) * width]
            )
        for name in ("q_scale", "rel_scale"):
            value = self[name]
            width = value.shape[0] // 2
            self[name] = mx.contiguous(value[rank * width : (rank + 1) * width])
        for name in ("k_sconv", "v_sconv"):
            conv = self[name]
            if self.is_sliding:
                width = conv.conv.weight.shape[0] // 2
                conv.conv.weight = mx.contiguous(
                    conv.conv.weight[rank * width : (rank + 1) * width]
                )
            else:
                conv.num_heads //= 2
                width = conv.static_weight.shape[1] // 2
                conv.static_weight = mx.contiguous(
                    conv.static_weight[:, rank * width : (rank + 1) * width]
                )
                width = conv.weight_proj.weight.shape[0] // 2
                conv.weight_proj.weight = mx.contiguous(
                    conv.weight_proj.weight[rank * width : (rank + 1) * width]
                )
        self.n_heads //= 2
        self.n_kv //= 2
        self.head_start = rank * self.n_heads
        self.sharding_group = group

    def _gather_features(self, x):
        if self.sharding_group is None:
            return x
        shape = x.shape
        return (
            mx.distributed.all_gather(
                x.reshape(1, -1, shape[-1]), group=self.sharding_group
            )
            .transpose(1, 0, 2)
            .reshape(*shape[:-1], shape[-1] * self.sharding_group.size())
        )

    def _project(self, projection, x):
        if self.sharding_group is None:
            return projection(x)
        # Both GEMV and GEMM dispatch can depend on N, including short batches
        # on older GPUs. Preserve the original N with zero rows for the other
        # rank, then select our complete rows. No remote weights are retained.
        width = projection.weight.shape[0]
        start = self.sharding_group.rank() * width
        weight = mx.pad(projection.weight, ((start, width - start), (0, 0)))
        return (x @ weight.T)[..., start : start + width]

    # ---- INKLING_FAST paths (mask / fuse / glue); stock path below is intact.

    # sink_proj is NOT fused: alone (N = n_heads = 24, K = 2304 >= 16 * N) MLX
    # dispatches the split-K GEMV (bn = 8), while inside a wide fused GEMV its
    # rows use bn = 1, a different FP32 reduction order -> occasional 1-ulp
    # BF16 differences in the sink logit (not bitwise vs stock). q/r/gate/k/v
    # all have K < 16 * N, so they reduce identically fused or alone.
    _FUSE_GROUPS = (("qrg", ("q_proj", "r_proj", "gate_proj")),
                    ("kv", ("k_proj", "v_proj")))

    def _fast_fused_valid(self, entry):
        # The fused buffer is only valid while every source module and its
        # weight are the exact objects we fused (load_weights / LoRA wraps
        # replace them).
        return entry is not None and all(
            self.get(n) is m and m.get("weight") is w for n, m, w in entry["src"]
        )

    def fast_prepare(self):
        """Concatenate projection weights once (INKLING_FAST=fuse).

        Re-validated on every call: if a source module or weight changed, the
        group is re-fused when still fusable, otherwise dropped (stock path).
        """
        cache = self.__dict__.get("_fast_fused_state")
        if cache is not None and all(self._fast_fused_valid(e) for e in cache.values()):
            return
        cache = {}
        if self.sharding_group is None:
            for key, names in self._FUSE_GROUPS:
                if key == "kv" and not self.is_sliding:
                    continue
                if not all(
                    type(self[n]) is nn.Linear
                    and "bias" not in self[n]
                    and _fast.gemv_reduction_matches_fused(self[n].weight)
                    for n in names
                ):
                    continue
                entry = _fast.fuse_linears([self[n] for n in names])
                cache[key] = {
                    "fused": entry,
                    "src": [(n, self[n], self[n].weight) for n in names],
                }
        self.__dict__["_fast_fused_state"] = cache
        self.__dict__["_fast_fused"] = {k: e["fused"] for k, e in cache.items()}

    def _fast_rel_proj(self, dtype):
        cached = self.__dict__.get("_fast_rel_proj_cache")
        if cached is None or cached[0] is not self.rel_proj or cached[1] != dtype:
            cached = (self.rel_proj, dtype, self.rel_proj.astype(dtype))
            self.__dict__["_fast_rel_proj_cache"] = cached
        return cached[2]

    def _fast_call(self, x, key_input, value_input, cache, conv_mask):
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        glue = _fast.enabled("glue")
        fused = None
        if _fast.enabled("fuse") and b * length == 1:
            self.fast_prepare()
            fused = self.__dict__["_fast_fused"]
        same_kv_input = key_input is None and value_input is None
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        if fused and "kv" in fused and same_kv_input:
            w, splits = fused["kv"]
            k, v = mx.split(x @ w.T, splits, axis=-1)
        else:
            k = self.k_proj(key_input)
            v = self.v_proj(value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        if fused and "qrg" in fused:
            w, splits = fused["qrg"]
            q, r, gate = mx.split(x @ w.T, splits, axis=-1)
        else:
            q, r = self.q_proj(x), self.r_proj(x)
            gate = None
        sinks = None
        if _FAST_QKVCLIP and _fast.qkv_clip_supported(self, cfg, x):
            # INKLING_FAST=qkvclip: clips, value norm and tau in one kernel.
            tau = None
            if not self.is_sliding and cfg.log_scaling_n_floor is not None:
                offset = kv.offset if kv is not None else 0
                if glue and isinstance(offset, int):
                    tau = _fast.log_n_tau(
                        offset, length, cfg.log_scaling_alpha, cfg.log_scaling_n_floor
                    )
                else:
                    positions = mx.arange(length) + mx.array(offset)[..., None] + 1
                    tau = 1 + cfg.log_scaling_alpha * mx.log(
                        mx.maximum(positions / cfg.log_scaling_n_floor, 1)
                    )
                tau = mx.broadcast_to(
                    tau[..., None, None].astype(mx.float32), (b, length, 1, 1)
                ).reshape(-1)
            q, k, v, r = _fast.qkv_clip(q, k, v, r, self, cfg, tau)
        else:
            q = l2_clip(
                q.reshape(b, length, self.n_heads, self.head_dim),
                cfg.qk_l2_clip_threshold,
                self.q_scale,
            )
            k = l2_clip(
                k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold
            )
            v = v.reshape(b, length, self.n_kv, self.head_dim)
            if cfg.use_value_norm_no_gamma:
                v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(
                    x.dtype
                )
            r = l2_clip(
                r.reshape(b, length, self.n_heads, cfg.d_rel),
                cfg.rel_l2_clip_threshold,
                self.rel_scale,
            )
            if not self.is_sliding and cfg.log_scaling_n_floor is not None:
                offset = kv.offset if kv is not None else 0
                if glue and isinstance(offset, int):
                    tau = _fast.log_n_tau(
                        offset, length, cfg.log_scaling_alpha, cfg.log_scaling_n_floor
                    )
                else:
                    positions = mx.arange(length) + mx.array(offset)[..., None] + 1
                    tau = 1 + cfg.log_scaling_alpha * mx.log(
                        mx.maximum(positions / cfg.log_scaling_n_floor, 1)
                    )
                q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
                r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        rel_proj = (
            self._fast_rel_proj(x.dtype) if glue else self.rel_proj.astype(x.dtype)
        )
        if sinks is None:
            sinks = self.sink_proj(x)
        physical = False
        exact = False
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            size = k.shape[2]
            if (
                _FAST_MASK_EXACT
                and not _FAST_MASK_ANY
                and length == 1
                and getattr(kv, "left_padding", None) is None
                and (
                    type(kv) is KVCache
                    or (type(kv) is RotatingKVCache and kv.keep == 0)
                )
            ):
                exact = True
                # Mask over the S real keys (temporal order: idx == S), with the
                # sink logit as column S -- same values as banded mask + concat.
                mask = _fast.ring_mask(
                    r, rel_proj, mx.array([size], mx.int32), k, self.sliding,
                    sinks=sinks,
                )
                k, v = _exact_sink_kv(kv, k, v)
            elif (
                _FAST_MASK_ANY
                and length == 1
                and getattr(kv, "left_padding", None) is None
                and (
                    type(kv) is KVCache
                    or (
                        type(kv) is RotatingKVCache
                        and kv.keep == 0
                        and (kv._idx == size or size == kv.max_size)
                    )
                )
            ):
                physical = True
                if _FAST_MASK_RING and type(kv) is RotatingKVCache:
                    # Keys stay in physical ring order; mask maps slot -> distance.
                    idx = kv._idx
                else:
                    if type(kv) is RotatingKVCache:
                        k, v = kv._temporal_order(k), kv._temporal_order(v)
                    idx = size
            elif isinstance(kv, RotatingKVCache):
                k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        size = k.shape[2]
        scale = self.head_dim**-0.5
        if exact:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask
            )
        elif physical:
            idx = mx.array([idx], mx.int32)
            if b == 1 and _FAST_MASK_SINK:
                # SDPA sinks are per head [H]; for B=1, L=1 that is per query.
                mask = _fast.ring_mask(r, rel_proj, idx, k, self.sliding)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask, sinks=sinks.reshape(-1)
                )
            else:
                mask = _fast.ring_mask(r, rel_proj, idx, k, self.sliding, sinks=sinks)
                zeros = mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    mx.concatenate([k, zeros], axis=2),
                    mx.concatenate([v, zeros], axis=2),
                    scale=scale,
                    mask=mask,
                )
        else:
            padding = getattr(kv, "left_padding", None)
            mask = banded_additive_mask(
                r,
                rel_proj,
                size - length,
                size,
                self.sliding,
                self.rel_extent,
                left_padding=padding,
                shape_ref=k,
            )
            mask = mx.concatenate([mask, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
            zeros = mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)
            out = mx.fast.scaled_dot_product_attention(
                q,
                mx.concatenate([k, zeros], axis=2),
                mx.concatenate([v, zeros], axis=2),
                scale=scale,
                mask=mask,
            )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        if gate is None:
            gate = self.gate_proj(x)
        if glue:
            out = _fast.output_gate(out, gate)
        else:
            out = (
                out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
            ).astype(x.dtype)
        return self.o_proj(out)

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        if _FAST_ATTN and self.sharding_group is None and not self.training:
            return self._fast_call(x, key_input, value_input, cache, conv_mask)
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        k = self._project(self.k_proj, key_input)
        v = self._project(self.v_proj, value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        q = l2_clip(
            self._project(self.q_proj, x).reshape(
                b, length, self.n_heads, self.head_dim
            ),
            cfg.qk_l2_clip_threshold,
            self.q_scale,
        )
        k = l2_clip(
            k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold
        )
        v = v.reshape(b, length, self.n_kv, self.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(
                x.dtype
            )
        r = l2_clip(
            self._project(self.r_proj, x).reshape(b, length, self.n_heads, cfg.d_rel),
            cfg.rel_l2_clip_threshold,
            self.rel_scale,
        )
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset if kv is not None else 0
            positions = mx.arange(length) + mx.array(offset)[..., None] + 1
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1)
            )
            q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
            r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            # Relative logits index backward distance, so ring buffers must be
            # read in chronological order (ordinary RoPE attention does not care).
            if isinstance(kv, RotatingKVCache):
                k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        size = k.shape[2]
        padding = getattr(kv, "left_padding", None)
        mask = banded_additive_mask(
            r,
            self.rel_proj.astype(x.dtype),
            size - length,
            size,
            self.sliding,
            self.rel_extent,
            left_padding=padding,
            shape_ref=k,
        )
        # A query-dependent sink is represented by a zero K/V and its own logit.
        # Keep this tiny projection's original width: splitting its 48 output
        # rows changes the prefill GEMM reduction and hence BF16 rounding.
        sinks = self.sink_proj(x)
        if self.sharding_group is not None:
            sinks = sinks[..., self.head_start : self.head_start + self.n_heads]
        sinks = sinks.transpose(0, 2, 1)[..., None]
        mask = mx.concatenate([mask, sinks], axis=-1)
        k = mx.concatenate(
            [k, mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)], axis=2
        )
        v = mx.concatenate(
            [v, mx.zeros((b, self.n_kv, 1, self.head_dim), v.dtype)], axis=2
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        out = (
            out.astype(mx.float32)
            * mx.sigmoid(self._project(self.gate_proj, x).astype(mx.float32))
        ).astype(x.dtype)
        # Every output dot still consumes all heads on one rank. Gather the
        # heads first, project complete output rows, then restore hidden order.
        return self._gather_features(
            self._project(self.o_proj, self._gather_features(out))
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap

    def __call__(self, up, gate):
        return _softcapped_swiglu(up, gate, self.gate_cap, self.up_cap)


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.sharding_group = None
        self.expert_start = 0
        self.expert_end = config.n_routed_experts
        self.gate_weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.intermediate_size,
            config.n_routed_experts,
            activation=SoftcappedSwiGLU(config),
        )

    def _fast_router_weight(self):
        cached = self.__dict__.get("_fast_router")
        if cached is None or cached[0] is not self.gate_weight:
            cached = (self.gate_weight, self.gate_weight.astype(mx.float32))
            self.__dict__["_fast_router"] = cached
        return cached[1]

    def _fast_gate_up_key(self):
        # Identity of both modules and every array the fused buffer derives from.
        sm = self.switch_mlp
        g, u = sm.get("gate_proj"), sm.get("up_proj")
        names = ("weight", "scales", "biases", "bias", "pre_quant_scale",
                 "weight_scale_2", "weight_global_scale")
        return (g, u) + tuple(m.get(n) for m in (g, u) for n in names)

    def fast_prepare(self):
        """Fuse gate+up expert weights once (INKLING_FAST=fusemoe).

        Re-validated on every call by identity of the two modules and every
        weight/scale array (load_weights / LoRA wraps replace them); on a
        change the pair is re-fused if still fusable, else the stock path runs.
        """
        state = self.__dict__.get("_fast_gate_up_state")
        if state is not None:
            key = self._fast_gate_up_key()
            if len(key) == len(state[0]) and all(a is b for a, b in zip(key, state[0])):
                return state[1]
        fused = (
            _fast.fuse_switch_gate_up(self.switch_mlp)
            if self.sharding_group is None
            else None
        )
        # Key taken after fusing: gate/up weights are now views of the buffer.
        self.__dict__["_fast_gate_up_state"] = (self._fast_gate_up_key(), fused)
        return fused

    def _fast_decode_experts(self, x, indices, fused):
        # Mirrors SwitchGLU.__call__ for L=1 (8 routes: no sort), with gate and
        # up computed by one gather_qmm over the fused [gate; up] weight.
        sm = self.switch_mlp
        xe = mx.expand_dims(x, (-2, -3))
        up, gate = _fast.fused_gate_up(fused, xe, indices)
        out = sm.down_proj(sm.activation(up, gate), indices, sorted_indices=False)
        return out.squeeze(-2)

    def _fast_sorted_experts(self, x, indices, fused):
        # Mirrors SwitchGLU.__call__'s sorted path (indices.size >= 64, L not
        # in 2..8: prefill and wide batched decode). Once fusemoe has run,
        # gate/up are strided views of the fused [E, 2n, K] buffer and two
        # sorted gather_qmm calls over those views are ~15% slower at 1k
        # prefill. One sorted gather_qmm over the contiguous fused buffer gives
        # the same per-column K reduction (bitwise) in one launch.
        from ..switch_layers import _gather_sort, _scatter_unsort

        sm = self.switch_mlp
        xs, idx, inv_order = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
        up, gate = _fast.fused_gate_up(fused, xs, idx, sorted_indices=True)
        out = sm.down_proj(sm.activation(up, gate), idx, sorted_indices=True)
        return _scatter_unsort(out, inv_order, indices.shape).squeeze(-2)

    def __call__(self, x):
        return _combine_routes(*self._routed(x))

    def _routed(self, x):
        """Per-route expert outputs [..., K, D] and route weights [..., K]."""
        if _FAST_ROUTE and not self.training and _fast._on_gpu():
            # Stock logits GEMV, then one kernel for sigmoid/bias/top-k/norm.
            indices, weights = _fast.route(
                x,
                self._fast_router_weight(),
                self.e_score_correction_bias,
                self.top_k,
                self.route_scale,
            )
        else:
            indices, weights = _route(
                x,
                self._fast_router_weight() if _FAST_GLUE else self.gate_weight,
                self.e_score_correction_bias,
                self.top_k,
                self.route_scale,
            )
        if self.sharding_group is None:
            if (
                not self.training
                and x.shape[0] <= 4
                and 1 < x.shape[1] <= 8
                and getattr(self.switch_mlp.down_proj, "mode", None) == "nvfp4"
            ):
                out = self.switch_mlp.masked(x, indices.astype(mx.int32))
            elif (
                _FAST_MOESORT
                and not self.training
                and x.shape[1] == 1
                and x.shape[0] >= 4
                # At >= 4 routes per expert stock GatherQMM already switches to
                # its sorted matrix kernel (each expert read once): keep stock.
                and indices.size < 4 * self.switch_mlp.num_experts
                and _fast._on_gpu()
                and _fast.moesort_supported(self.switch_mlp)
            ):
                # Batched decode: expert-sorted routes, grouped native matvecs.
                out = _fast.sorted_switch_glu(
                    self.switch_mlp,
                    x,
                    indices,
                    self.fast_prepare() if _FAST_FUSEMOE else None,
                )
            elif (
                _FAST_FUSEMOE
                and not self.training
                and x.shape[1] == 1
                and indices.size < 64
                and self.fast_prepare() is not None
            ):
                out = self._fast_decode_experts(x, indices, self.fast_prepare())
            elif (
                _FAST_FUSEMOE
                and not self.training
                and x.ndim == 3
                # Exactly where stock SwitchGLU takes its sorted gather path.
                and not (1 < x.shape[1] <= _DECODE_BLOCK_SIZE)
                and indices.size >= 64
                and getattr(self.switch_mlp.gate_proj, "num_experts", None)
                == self.switch_mlp.num_experts
                and self.fast_prepare() is not None
            ):
                out = self._fast_sorted_experts(x, indices, self.fast_prepare())
            else:
                out = self.switch_mlp(x, indices)
        else:
            owned = (indices >= self.expert_start) & (indices < self.expert_end)
            local = mx.where(owned, indices.astype(mx.int32) - self.expert_start, -1)
            out = self.switch_mlp.masked(x, local)
            # Exactly one rank owns each route. Reduce route outputs before
            # weighting to retain the unsharded FP32 accumulation order.
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out, weights


class AttentionResidual(nn.Module):
    def __init__(self, config, num_queries=1):
        super().__init__()
        self.query = mx.zeros(
            (config.hidden_size,)
            if num_queries == 1
            else (num_queries, config.hidden_size)
        )
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks, partial):
        if (
            _FAST_ATTNRES
            and isinstance(blocks, _fast.KeyedBlocks)
            and self.normalize_values
            and self.query.ndim == 1
            and _fast.attn_res_supported(partial, self.query, len(blocks.keys))
        ):
            # INKLING_FAST=attnres (with xpass keys): one kernel.
            return _fast.attn_res(blocks.keys, partial, self.query)
        if isinstance(blocks, _fast.KeyedBlocks) and self.normalize_values:
            # INKLING_FAST=xpass: cached block keys + cross-pass depth logits
            # (not always bitwise in batches: see DETAILS.md "Known limits").
            return _fast.attention_residual_keyed(
                self, blocks, partial, reuse_logits=_FAST_XPASS_LOGITS
            )
        return _attention_residual(
            mx.stack([*blocks, partial], axis=-2), self.query, self.normalize_values
        )


class MessageController(nn.Module):
    """Blend a previous pass's message into one sublayer input."""

    def __init__(self, config):
        super().__init__()
        rank, width = config.loop_config.gate_rank, config.hidden_size
        self.down_weight = mx.zeros((rank, 2 * width))
        self.forget_weight = mx.zeros((width, rank))
        self.gain_weight = mx.zeros((width, rank))
        self.eps = config.rms_norm_eps

    def __call__(self, x, message):
        if _FAST_GLUE:
            return _fast.message_controller(
                x,
                message,
                self.down_weight,
                self.forget_weight,
                self.gain_weight,
                self.eps,
            )
        pair = mx.concatenate(
            [mx.fast.rms_norm(v, None, self.eps) for v in (x, message)], axis=-1
        )
        hidden = (pair.astype(self.down_weight.dtype) @ self.down_weight.T).astype(
            mx.float32
        )
        forget = mx.tanh(hidden @ self.forget_weight.T)
        gain = mx.tanh(hidden @ self.gain_weight.T)
        return (1 - forget) * x + gain * message


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.attention_input = (
            AttentionResidual(config, 1 if self.is_sliding else 3)
            if layer_idx
            else None
        )
        self.mlp_input = AttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoE(config)
        names = (
            ("input_layernorm",)
            if self.is_sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        for name in (*names, "post_attention_layernorm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 2
        )
        self.mlp_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 3
        )
        loop = config.loop_config
        if loop is not None and loop.core_start <= layer_idx < loop.core_end:
            self.controllers = {
                name: MessageController(config) for name in ("attn", "k", "v", "mlp")
            }

    def __call__(self, x, blocks, cache=None, messages=None):
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = (
            self.attention_input(blocks, x)
            if self.attention_input is not None
            else x[..., None, :]
        )
        if self.is_sliding and messages is None:
            attn = self.self_attn(
                self.input_layernorm(mixed[..., 0, :]).astype(x.dtype),
                cache=cache,
                conv_mask=mask,
            )
        else:
            parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            if (
                messages is not None
                and _FAST_CTRLBATCH
                and not self.training
                and x.shape[0] * x.shape[1] <= 8
            ):
                # One batched computation over stacked [3, ...] weights;
                # bitwise vs three controllers up to M=8 (not at M=16).
                stk = _fast.stacked_controllers(self)
                fn = _fast.controllers3_compiled if _FAST_GLUE else _fast.controllers3
                out3 = fn(
                    mx.stack(parts),
                    mx.moveaxis(messages[..., :3, :], -2, 0),
                    stk["down_weight"],
                    stk["forget_weight"],
                    stk["gain_weight"],
                    self.controllers["attn"].eps,
                )
                parts = [out3[i] for i in range(3)]
            elif messages is not None:
                parts = [
                    self.controllers[name](part, messages[..., i, :])
                    for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))
                ]
            norms = (
                (self.input_layernorm,) * 3
                if self.is_sliding
                else (
                    self.q_input_layernorm,
                    self.k_input_layernorm,
                    self.v_input_layernorm,
                )
            )
            parts = [norm(part).astype(x.dtype) for norm, part in zip(norms, parts)]
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        if messages is not None:
            hidden = self.controllers["mlp"](hidden, messages[..., 3, :])
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(
            x.dtype
        )
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.layer_indices = config.layer_indices
        self.loop = config.loop_config
        if self.loop is not None:
            self.message_queries = mx.zeros(
                (self.loop.core_end - self.loop.core_start, 4, config.hidden_size)
            )
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        skip_final_norm=False,
        capture_layer_ids=None,
        speculative_verify=False,
    ):
        captures = {}
        if capture_layer_ids is not None and (
            len(capture_layer_ids) != len(set(capture_layer_ids))
            or any(
                type(i) is not int or not 0 <= i <= len(self.layer_indices)
                for i in capture_layer_ids
            )
        ):
            raise ValueError("Invalid Inkling 2 capture_layer_ids")
        requested = set(capture_layer_ids or [])
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        xpass = _FAST_XPASS and not self.training
        blocks = _fast.KeyedBlocks() if xpass else []
        bank = None
        if cache and len(cache) != len(self.layer_indices):
            raise ValueError(
                "Inkling 2 requires a separate cache for every layer visit"
            )
        for visit, index in enumerate(self.layer_indices):
            if self.loop is not None and index == self.loop.core_start:
                if visit == index:
                    if xpass:
                        # Prelude blocks + the prelude partial (appended at
                        # core_start) repeat in every pass.
                        blocks.stable = len(blocks) + 1
                    prelude, prelude_blocks = x, blocks.copy()
                else:
                    # The first block appended in the core is the prelude partial.
                    # Only completed core blocks and the last partial supply messages.
                    first = len(prelude_blocks) + 1
                    if xpass:
                        # Same rows as rms_norm(stack(...)): row-wise, expected to match.
                        bank = mx.concatenate(
                            [*blocks.keys[first:], _fast._block_key(x)], axis=-2
                        )
                    else:
                        states = mx.stack([*blocks[first:], x], axis=-2)
                        bank = mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)
                    x, blocks = prelude, prelude_blocks.copy()
            if xpass and self.loop is not None and index == self.loop.core_end:
                blocks.stable = 0  # past the loop: nothing repeats any more
            messages = None
            if bank is not None and index < self.loop.core_end:
                if _FAST_MSGPACK and not self.training and x.size // x.shape[-1] <= 16:
                    # INKLING_FAST=msgpack: one pass's messages for every core
                    # layer at once (the bank is fixed within a pass).
                    if index == self.loop.core_start:
                        pass_messages = _fast.packed_messages(
                            self.message_queries, bank
                        )
                    messages = pass_messages[..., index - self.loop.core_start, :, :]
                else:
                    query = self.message_queries[index - self.loop.core_start]
                    scores = (query @ bank.swapaxes(-1, -2)) * x.shape[-1] ** -0.5
                    messages = mx.softmax(scores, axis=-1) @ bank
            # DSpark taps the partial block entering the layer, not its output.
            if visit in requested:
                captures[visit] = x
            x = self.layers[index](x, blocks, cache[visit] if cache else None, messages)
            if speculative_verify and (visit + 1) % 4 == 0:
                mx.async_eval(x)
        h = self.final_input(blocks, x)[..., 0, :]
        normalized = self.norm(h).astype(x.dtype)
        if len(self.layer_indices) in requested:
            captures[len(self.layer_indices)] = normalized
        result = h if skip_final_norm else normalized
        if capture_layer_ids is not None:
            return result, [captures[i] for i in capture_layer_ids]
        return result


class LanguageModel(InklingLanguageModel):
    def __init__(self, config: TextConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sharding_group = None

    def chunked_prefill_policy(self, **kwargs):
        # Captures are per-token partial blocks; the shared prefill collector
        # retains them across chunks. Preserve the caller's target chunk size.
        return True

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        cache=None,
        return_hidden=False,
        return_shared_kv=False,
        skip_logits=False,
        capture_layer_ids=None,
        speculative_verify=False,
        **kwargs,
    ):
        from ...speculative.cache_state import start_speculative_cache

        transaction = None
        if speculative_verify:
            length = inputs.shape[1] if inputs is not None else inputs_embeds.shape[1]
            transaction = start_speculative_cache(cache, length)
        try:
            result = self.model(
                inputs,
                cache=cache,
                input_embeddings=inputs_embeds,
                skip_final_norm=True,
                capture_layer_ids=capture_layer_ids,
                speculative_verify=speculative_verify,
            )
            pre_norm, captured = (
                result if capture_layer_ids is not None else (result, None)
            )
            return LanguageModelOutput(
                logits=(
                    None
                    if skip_logits
                    else self._logits_from_norm(self.model.norm(pre_norm))
                ),
                hidden_states=(
                    captured
                    if capture_layer_ids is not None
                    else ([pre_norm] if return_hidden else None)
                ),
                shared_kv_states={} if return_shared_kv else None,
                gdn_states=transaction,
            )
        except BaseException:
            if transaction is not None:
                transaction.abort()
            raise

    def shard(self, group=None):
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        count = self.config.n_routed_experts
        validate_sharding(vars(self.config), size)
        if not 0 <= rank < size:
            raise ValueError("Each distributed rank must own at least one expert")
        if self.sharding_group is not None:
            raise ValueError("Inkling 2 is already sharded")
        if size == 1:
            return
        start, end = count * rank // size, count * (rank + 1) // size
        scalar_scales = {"weight_scale_2", "weight_global_scale"}
        # Check every layer before changing any weights, including native
        # packed weights, group scales, and per-expert FP32 outer scales.
        for layer in self.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer.mlp.switch_mlp, name)
                for key, value in projection.parameters().items():
                    if not isinstance(value, mx.array):
                        continue
                    if key in scalar_scales and value.size == 1:
                        continue
                    if not value.ndim or value.shape[0] != count:
                        raise ValueError(
                            f"Invalid expert partition for {name}.{key}: "
                            f"expected {count} experts, got {value.shape}"
                        )
        for layer in self.layers:
            moe = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(moe.switch_mlp, name)
                projection.update(
                    {
                        key: mx.contiguous(value[start:end])
                        for key, value in projection.parameters().items()
                        if isinstance(value, mx.array)
                        and not (key in scalar_scales and value.size == 1)
                    }
                )
            # Materialize one layer at a time and release the full expert
            # buffers; lazy slices alone retain the unpartitioned allocation.
            mx.eval(moe.switch_mlp.parameters())
            mx.clear_cache()
            moe.expert_start, moe.expert_end = start, end
            moe.sharding_group = group
            layer.self_attn.shard(group)
            mx.eval(layer.self_attn.parameters())
            mx.clear_cache()
        self.sharding_group = group

    def _logits_from_norm(self, h):
        # hf_quant_config.json stores BF16 unembedding weights but requires
        # fwd_output_dtype=float32. Promote the product before multiplication;
        # casting BF16 scores afterwards loses the accumulator's low bits.
        h = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
        head_ok = (
            not self.training
            and type(self.lm_head) is nn.Linear
            and "bias" not in self.lm_head
            and self.lm_head.weight.dtype == mx.bfloat16
            and self.sharding_group is None
            and h.ndim == 3
            and _fast._on_gpu()
        )
        if (
            head_ok
            and _fast.enabled("head")
            and h.shape[0] * h.shape[1] == 1
            and _fast.head_exact_supported(1, h.shape[-1])
        ):
            # Bitwise twin of the cached-FP32 M=1 GEMV below (same per-row
            # FMA order and simd_shuffle_down tree as MLX 0.32.4's gemv),
            # reading the BF16 weight directly. Softcap and slice fused.
            return _fast.lm_head_gemv_exact(
                h,
                self.lm_head.weight,
                self.config.unpadded_vocab_size,
                self.config.final_logit_softcapping or 0.0,
            )
        if (
            head_ok
            and _fast.enabled("head_approx")
            and h.shape[0] * h.shape[1] <= 15
            and h.shape[-1] % 8 == 0
        ):
            # NOT bitwise: FP32 accumulation in a different order than stock's
            # GEMV/GEMM (ordering-only; operands are the same exact FP32 values).
            # BF16 weight read with FP32 activation and FP32 accumulation; the
            # BF16->FP32 widening is exact, so only the reduction order differs
            # from the cached-FP32 path. Softcap and vocab slice are fused.
            return _fast.lm_head_gemv(
                h,
                self.lm_head.weight,
                self.config.unpadded_vocab_size,
                self.config.final_logit_softcapping or 0.0,
            )
        if (
            not self.training
            and type(self.lm_head) is nn.Linear
            and "bias" not in self.lm_head
            and h.ndim == 3
            and h.shape[0] * h.shape[1] <= 15
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            # Retain the exact conversion instead of reading and promoting the
            # entire vocabulary matrix again on every decode step. Whole output
            # rows can be shared across the two expert ranks without splitting
            # any dot product or changing its FP32 accumulation.
            weight = self.lm_head.weight
            group = self.sharding_group
            partition = (
                group is not None
                and group.size() == 2
                and weight.shape[0] % group.size() == 0
            )
            size, rank = (group.size(), group.rank()) if partition else (1, 0)
            cached = self.get("_head_projection_cache")
            if cached is None or cached[0] is not weight or cached[1:3] != (size, rank):
                width = weight.shape[0] // size
                projected = weight[rank * width : (rank + 1) * width].astype(mx.float32)
                self._head_projection_cache = (weight, size, rank, projected)
            projected = self._head_projection_cache[-1]
            logits = h @ projected.T
            if partition:
                width = projected.shape[0]
                logits = (
                    mx.distributed.all_gather(logits.reshape(1, -1, width), group=group)
                    .transpose(1, 0, 2)
                    .reshape(*h.shape[:-1], weight.shape[0])
                )
        else:
            logits = self.lm_head(h)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    def make_cache(self):
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                    if self.model.layers[i].is_sliding
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for i in self.model.layer_indices
        ]

    def speculative_verify_hidden(self, inputs, cache):
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            skip_logits=True,
            speculative_verify=True,
        )
        return out.hidden_states[-1], {}, out.gdn_states

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        from ...speculative.cache_state import rollback_speculative_cache

        return rollback_speculative_cache(caches, gdn_states, accepted, block_size)
