# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""Static-shape, mx.compile'd single-token decode for Inkling 2 (PLAN item #4).

Prefill stays on the normal mlx-vlm path (INKLING_FAST kernels stay on).  ``FastDecoder.from_cache`` converts the stock
prompt cache into fixed-shape state arrays, and ``FastDecoder.step`` runs one
decode token (L=1) for all 48 layer visits as ONE compiled graph:

* sliding visits: K/V ring buffers [B, n_kv, W + 1, D]; token at position p is
  written to physical slot p % W.  Slot W is a permanent zero K/V row (sink).
  One shared ``swa_pos`` [B, W] int32 array records the logical position held
  in each slot (-1 = empty); the rel-bias mask is built in physical slot order
  from it, so the ring is never reordered (attention is permutation-invariant
  over keys once the bias is in the mask).
* global visits: K/V buffers [B, n_kv, cap + 1, D] (slot j holds position j);
  ``cap`` comes from ``buckets``.  When the position reaches ``cap`` the
  buffers are copied into the next bucket (one new compiled graph per bucket).
* conv states: per visit the four [B, 3, C] FP32 arrays of the ArraysCache(4).
* position: ``q_pos`` [B] int32, passed as an array, so nothing in the traced
  function depends on a Python value that changes per step.

In the approximate path (``exact=False`` / INKLING_FAST_DECODE_APPROX=1) the
arithmetic mirrors ``language.py`` op for op (same submodules, same FP32
islands and casts).  Its deliberate differences, all reduction-order only:
key order inside SDPA (physical vs temporal), masked padding slots in the
global buffer, the rel-bias dot as an FP32 matmul instead of the mask kernel's
serial loop, and (B == 1) the sink via SDPA ``sinks=`` instead of a zero K/V
column.  The default exact path instead gives decode attention the stock SDPA
operands (see ``exact`` in ``__init__``).  Flags: INKLING_FAST_SINKS=0 forces
the zero-slot sink path.

Composes with INKLING_FAST (fast_kernels.py): ``mask`` -> slot_mask kernel,
``fuse`` -> fused [q;r;gate] / [k;v] GEMVs (B == 1), ``glue`` / ``fusemoe``
/ ``head`` -> used through the stock submodules.  Every lazily cached array
those paths create is built in ``_prepare`` BEFORE tracing (an array cached
from inside a trace would be a tracer).

Restructurings of our own (on by default, kwargs to disable):
``reuse_keys``: rms_norm(block) is computed once when a block is appended and
reused by every later AttentionResidual / message bank (a row-wise op, so we
expect it to be bitwise; it was in our B=1 runs, but not in some batched
tests, see DETAILS.md "Known limits"); ``batch_ctrl``: the attn/k/v
MessageControllers of a core visit run as one batched computation over
stacked [3, ...] weights, only for B*L <= 8.

Batched decode: B rows with different (left-padded) lengths are refused
in the default exact mode unless INKLING_FAST_DECODE_PADDED=1, and then not bitwise; in that mode
each row keeps its own logical positions (q_pos[b]); pads never enter the
buffers (slot_pos = -1).

Falls back to the normal mlx-vlm path, INKLING_FAST kernels on (``Unsupported``): sharded models are refused at
construction; a prompt beyond the largest bucket is refused by from_cache;
decoding past the largest bucket converts back with ``to_cache`` and
continues on that path inside ``step``.  The compiled graph captures the
weights at trace time: build a new FastDecoder after load_weights / LoRA.

Planned: compiled block verify (DSpark, L = 2..8) on the same state.
* ``ring_extra = Lmax - 1`` (already supported): the ring holds W + L - 1
  slots, so a block's first query still sees the tokens its later writes
  would overwrite; the window test masks the extra slots.
* verify(tokens [B, L]): positions q_pos + arange(L); write L slots per visit
  (slice_update at q_pos for B=1, put_along_axis otherwise); mask
  [B, H, L, S] from slot_pos with per-query q (causal inside the block comes
  free from dist < 0); conv states for every prefix length (the stock
  _conv_state_history kernel) returned as [B, L, 3, C].
* rollback(accepted [B]) is pure array work, no copies: q_pos += accepted;
  rejected KV slots need no undo (slot_pos > q_pos -> dist < 0 -> masked, and
  global slots > q_pos are masked); swa_pos entries > new q_pos set to -1;
  conv state = take(prefix_states, accepted - 1).  One compiled graph per
  (L, bucket).
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn

from ..cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache
from . import fast_kernels as fk
from . import mega_kernels as mk
from .language import l2_clip

NEG = -1e30


_LOUD_SEEN = set()


def _loud(msg):
    """Fallbacks to non-exact or stock paths are printed once per process (review F4)."""
    if msg not in _LOUD_SEEN:
        _LOUD_SEEN.add(msg)
        import sys

        print("[inkling2.fast_decode] " + msg, file=sys.stderr, flush=True)


class Unsupported(Exception):
    """FastDecoder cannot take this model/cache; callers use the normal mlx-vlm path."""

DEFAULT_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def _sdpa_cap_limits(n_q, n_kv):
    """Largest global-bucket caps that keep decode SDPA on the kernel config
    stock picks (MLX 0.32 metal dispatch, scaled_dot_product_attention.cpp).

    Exact decode feeds SDPA cap + 1 keys where stock feeds S + 1 (S = keys in
    cache). Trailing masked keys are exact no-ops inside one kernel (exp(-1e30
    - max) == 0, factor == 1), but the kernel itself (1-pass vs 2-pass, number
    of 2-pass blocks) is chosen from N, so every N threshold below gets a
    bucket boundary: a cap of t - 2 serves exactly the positions whose stock N
    is < t.
    """
    arch = mx.device_info().get("architecture", "") if fk._on_gpu() else ""
    devc = arch[-1:] if arch else ""
    gqa = max(1, n_q // max(1, n_kv))
    n_simds = gqa  # q length 1
    limits = set()
    if devc in ("d", "s"):
        limits.add(1024)  # 1-pass below, 2-pass at/above
        if devc == "s" and n_simds > 4:
            limits.update((1025, 8193, 32769, 65537))  # blocks grow for N > t-1
        if devc == "d":
            if n_simds <= 2:
                limits.add(8193)
            elif n_simds >= 6:
                limits.update((16384, 65536))
    elif n_kv < n_q:
        limits.add(4096)
    return sorted(t - 2 for t in limits)


def _rel_slot_mask(r, proj, slot_pos, q_pos, window, rel_extent, dtype):
    """Pure-MLX physical-order banded rel-bias mask.

    r: [B, L, H, d_rel]; proj: [d_rel, R] (already cast like stock);
    slot_pos: [B, S] int32 (-1 empty); q_pos: [B] (L=1) or [B, L] positions of
    the queries; returns [B, H, L, S] additive mask in ``dtype``.
    """
    B, L, H, _ = r.shape
    if q_pos.ndim == 1:
        q_pos = q_pos[:, None]
    # Stock kernel: fp32 sum over d_rel of float(rel)*float(proj), one round to T.
    rl = (r.astype(mx.float32) @ proj.astype(mx.float32)).transpose(0, 2, 1, 3)
    dist = q_pos[:, :, None] - slot_pos[:, None, :]  # [B, L, S]
    invalid = (slot_pos[:, None, :] < 0) | (dist < 0)
    if window > 0:
        invalid = invalid | (dist >= window)
    gidx = mx.clip(dist, 0, rel_extent - 1)
    S = slot_pos.shape[-1]
    bias = mx.take_along_axis(
        rl, mx.broadcast_to(gidx[:, None], (B, H, L, S)), axis=-1
    )
    if rel_extent < (window if window > 0 else 1 << 30):
        bias = mx.where((dist >= rel_extent)[:, None], 0.0, bias)
    bias = bias.astype(dtype)
    return mx.where(invalid[:, None], mx.array(NEG, dtype), bias)


def _write_slots(buf, new, slots):
    """buf [B, n, S, D] <- new [B, n, L, D] at per-row slots [B, L] int32.
    ``contiguous``: slots[b] = slots[b, 0] + arange(L) for every row."""
    B, _, L, _ = new.shape
    if B == 1 and L == 1:
        return mx.slice_update(buf, new, slots.reshape(1), axes=(2,))
    idx = mx.broadcast_to(slots[:, None, :, None], new.shape)
    return mx.put_along_axis(buf, idx, new, axis=2)


def _grouped_qmv(x, weight, scales, factor, ids, n, V=2):
    """masked_quantized.grouped_masked_qmv with explicit weight/N/V: all routes
    of one expert (sorted ids) share each weight read; per-vector reduction is
    the native QMV's (bitwise, verified at K=2304/N=1152 and tiny dims)."""
    from ..masked_quantized import _grouped_qmv as kern

    k = x.shape[-1]
    return kern()(
        inputs=[x, weight, scales, factor, ids],
        template=[("T", x.dtype), ("K", k), ("N", n), ("V", V)],
        grid=(32, n // 8 * 2, ids.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(*ids.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


def _attn_res_keys(bkeys, partial, query, normalize_values, states=None):
    """_attention_residual with cached per-block rms_norm keys (meant to match; batches: see DETAILS.md "Known limits")."""
    pk = mx.fast.rms_norm(partial.astype(mx.float32)[..., None, :], None, 1e-6)
    keys = mx.concatenate([*bkeys, pk], axis=-2)
    D = keys.shape[-1]
    scores = (keys @ query.reshape(-1, D).T) * D**-0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ (keys if normalize_values else states)


_attn_res_keys_c = mx.compile(_attn_res_keys)


def _controllers3(x3, m3, dw, fw, gw, eps):
    """Three MessageControllers at once. x3, m3: [3, B, 1, D]; dw [3, r, 2D];
    fw, gw [3, D, r].  Same op sequence as MessageController.__call__."""
    pair = mx.concatenate(
        [mx.fast.rms_norm(v, None, eps) for v in (x3, m3)], axis=-1
    )
    # Fold [B, 1] into M, as the per-controller matmuls do: [3, B, 2D] @ [3, 2D, r].
    shape = pair.shape
    pair = pair.reshape(shape[0], -1, shape[-1])
    hidden = (pair.astype(dw.dtype) @ dw.swapaxes(-1, -2)).astype(mx.float32)
    forget = mx.tanh(hidden @ fw.swapaxes(-1, -2)).reshape(x3.shape)
    gain = mx.tanh(hidden @ gw.swapaxes(-1, -2)).reshape(x3.shape)
    return (1 - forget) * x3 + gain * m3


_controllers3_c = mx.compile(_controllers3)


def _kv_rows(kv):
    """(keys, values) in temporal order [B, n, P, D] and logical lengths per row."""
    if isinstance(kv, RotatingKVCache):
        k, v = kv._temporal_order(kv.keys), kv._temporal_order(kv.values)
        return k, v, [kv.offset] * k.shape[0]
    if isinstance(kv, KVCache):
        T = kv.offset
        return kv.keys[:, :, :T], kv.values[:, :, :T], [T] * kv.keys.shape[0]
    if isinstance(kv, BatchRotatingKVCache):
        k, v = kv.keys, kv.values
        if kv.rotated:
            k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        else:
            k, v = k[:, :, : kv._idx], v[:, :, : kv._idx]
        return k, v, [int(o) for o in kv.offset.tolist()]
    if isinstance(kv, BatchKVCache):
        return (
            kv.keys[:, :, : kv._idx],
            kv.values[:, :, : kv._idx],
            [int(o) for o in kv.offset.tolist()],
        )
    raise TypeError(f"unsupported cache {type(kv).__name__}")


_BMOE_WARNED = set()  # (B, dtype, routed) shapes already warned about a stock MoE fallback


class FastDecoder:
    def __init__(self, language_model, buckets=DEFAULT_BUCKETS, use_sinks=None,
                 reuse_keys=True, batch_ctrl=True, use_kernels=True, ring_extra=0,
                 exact=None):
        if getattr(language_model, "sharding_group", None) is not None:
            raise Unsupported("sharded model: use the stock decode path")
        self.lm = language_model
        self.cfg = language_model.config
        self.model = language_model.model
        self.visits = list(self.model.layer_indices)
        self.sliding = [self.model.layers[i].is_sliding for i in self.visits]
        self.window = self.cfg.sliding_window_size
        # Sliding ring modulus.  Decode needs W slots; a block verify of L
        # tokens needs W + L - 1 (the block's first query still sees the
        # tokens its later writes would overwrite).  Extra slots are masked by
        # the window test (dist >= W), so decode results are unchanged.
        self.ring = self.window + int(ring_extra)
        self.fallback = None  # stock cache once we left the static path
        self.buckets = tuple(sorted(buckets))
        if use_sinks is None:
            use_sinks = os.environ.get("INKLING_FAST_SINKS", "1") != "0"
        self.use_sinks = use_sinks
        self.reuse_keys = reuse_keys and self.cfg.attn_res_normalize_values
        self.batch_ctrl = batch_ctrl
        self.xpass = fk.enabled("xpass")
        self.msgpack = fk.enabled("msgpack")
        self.qkvclip = fk.enabled("qkvclip") and use_kernels
        self.xmask = fk.enabled("xmask") and use_kernels
        self.stail = fk.enabled("sconvtail") and fk._on_gpu() and use_kernels
        self.attnres = fk.enabled("attnres") and use_kernels
        # INKLING_FAST=mega_ctrl: MessageControllers in two kernels (bitwise).
        self.mega_ctrl = fk.enabled("mega_ctrl") and fk._on_gpu() and use_kernels
        # INKLING_FAST=mega_norm: RMSNorm(w) + bf16 cast (all parts of a
        # sublayer) in one kernel instead of 3 launches per norm (bitwise).
        self.mega_norm = fk.enabled("mega_norm") and fk._on_gpu() and use_kernels
        # INKLING_FAST=mega_dsconv: global k/v DynamicShortConvolution (incl.
        # weight_proj GEMVs and the conv-state update) in one kernel (bitwise).
        self.mega_dsconv = fk.enabled("mega_dsconv") and fk._on_gpu() and use_kernels
        # INKLING_FAST=mega_attnres: attnres kernel reads the bf16 partial (no cast launch).
        self.mega_attnres = fk.enabled("mega_attnres")
        # INKLING_FAST=mega_post: see mega_kernels.post_block.
        self.mega_post = fk.enabled("mega_post") and fk._on_gpu() and use_kernels
        # INKLING_FAST=mega_qsm: qkvclip + sink_proj + exact decode mask in
        # one launch (B*L == 1, exact + xmask + qkvclip).
        self.mega_qsm = fk.enabled("mega_qsm") and self.qkvclip and self.xmask
        # needs the glue sconv path (state-writing kernel) it replaces
        self.mega_oproj = fk.enabled("mega_oproj") and fk._on_gpu() and use_kernels
        # INKLING_FAST=mega_kvip (with mega_qsm + mega_kvz): KV rows written in
        # place by the qsm kernel, no slice_update (no launches, no buffer copy).
        self.mega_kvip = fk.enabled("mega_kvip")
        # INKLING_FAST=mega_pre: see mega_kernels.pre_block.
        self.mega_pre = fk.enabled("mega_pre") and fk._on_gpu() and use_kernels
        self._pre, self._next = None, None
        # INKLING_FAST=mega_sdpa: exact decode SDPA replica reading the KV
        # buffers in place (no ring_gather), trailing masked keys skipped,
        # loads issued ahead (mega_kernels.sdpa_exact). Needs mega_qsm.
        self.mega_sdpa = fk.enabled("mega_sdpa") and fk._on_gpu() and use_kernels
        if self.mega_sdpa:
            # replica vs mx.fast once per process.
            c = self.cfg
            dims = {(c.num_attention_heads, c.num_key_value_heads, c.head_dim),
                    (c.swa_num_attention_heads, c.swa_num_key_value_heads,
                     getattr(c, "swa_head_dim", c.head_dim))}
            if not all(mk.sdpa_selftest(*d, self.window, self.ring) for d in dims):
                _loud("FastDecoder: mega_sdpa self-test vs mx.fast SDPA FAILED (MLX/compiler drift?); "
                      "mega_sdpa disabled, using the stock exact SDPA path")
                self.mega_sdpa = False
        self.mega_sconv2 = (fk.enabled("mega_sconv2") and fk.enabled("glue")
                            and fk._on_gpu() and use_kernels)
        # INKLING_FAST=mega_kvz: in the sliced exact regime qkvclip emits k/v
        # with the zero sink row attached, so each KV write is one 2-row
        # slice_update (no separate zero-row writes). Needs qkvclip + xmask.
        self.mega_kvz = fk.enabled("mega_kvz") and self.qkvclip and self.xmask
        # INKLING_FAST=mega_moe: router GEMV reads BF16 x (no AsType launch),
        # constant lhs indices for both gather_qmm (no Arange launches). Needs
        # route + fusemoe + glue (the stock decode path it mirrors).
        self.mega_moe = (
            fk.enabled("mega_moe") and fk.enabled("route") and fk.enabled("fusemoe")
            and fk.enabled("glue") and fk._on_gpu() and use_kernels
        )
        self._xp_memo, self._xp_stable = {}, 0
        on_gpu = fk._on_gpu() and use_kernels
        self.k_mask = on_gpu and (fk.enabled("mask") or fk.enabled("mask_ring"))
        self.k_fuse = on_gpu and fk.enabled("fuse")
        # exact (default): decode (L=1) attention gets the stock SDPA operands
        # -- temporal key order, the sink as a zero key at index S, an N that
        # selects the stock kernel config -- so B=1 decode was bitwise vs the
        # stock language path in the gated runs. INKLING_FAST_DECODE_APPROX=1 (or exact=False)
        # restores the old physical-order / sinks= path (not bitwise).
        if exact is None:
            exact = os.environ.get("INKLING_FAST_DECODE_APPROX", "0") != "1"
        if exact and not on_gpu:
            # The exact path needs the GPU kernels; never drop silently to the
            # non-bitwise approx path. Pass exact=False to opt into it.
            raise Unsupported("exact FastDecoder needs the GPU kernels (use_kernels=True on GPU)")
        if exact:
            fk.check_exact_runtime("FastDecoder exact")
        self.exact = bool(exact)
        if self.exact:
            cfg = self.cfg
            limits = set()
            for nq, nkv in ((cfg.num_attention_heads, cfg.num_key_value_heads),
                            (cfg.swa_num_attention_heads, cfg.swa_num_key_value_heads)):
                limits.update(_sdpa_cap_limits(nq, nkv))
            # Boundaries inside the configured range only (the largest bucket
            # still decides when to leave for the normal mlx-vlm path).
            top = max(self.buckets)
            self.buckets = tuple(sorted(set(self.buckets) | {t for t in limits if t < top}))
            # The sliced sliding regime (cap < W) must not wrap.
            self._slice_cap = max([t for t in limits if t < self.window], default=0)
        self._compiled = mx.compile(self._step_flat)
        self._kvip_pending = False  # in-place KV writes MLX cannot see (F1)
        self.state = None
        self._known, self._rounds = None, []  # host position bookkeeping
        self.cap = None
        self._spec_compiled = {}
        self._prefixes = None
        self.dedup = os.environ.get("INKLING_FAST_VERIFY_DEDUP", "1") != "0"
        self.dedup_v = int(os.environ.get("INKLING_FAST_VERIFY_V", "2"))
        self.tokenwise_moe = os.environ.get("INKLING_FAST_VERIFY_TOKENWISE", "1") != "0"
        # expert-grouped verify MoE extensions (verify_dedup.py), meant to match decode rows.
        self.ds_dedup_mxfp8 = os.environ.get("INKLING_DS_DEDUP_MXFP8", "0") == "1"
        self.ds_dedup_down = os.environ.get("INKLING_DS_DEDUP_DOWN", "0") == "1"
        self.ds_route_tokenwise = os.environ.get("INKLING_DS_ROUTE_TOKENWISE", "0") == "1"
        # L>1 verify speedups (INKLING_SPECFAST=all or a comma list). They change
        # only the verify (L>1) graph; the L=1 decode step is untouched. NONBITWISE-OK class.
        sf = {t for t in os.environ.get("INKLING_SPECFAST", "").split(",") if t}
        if "all" in sf:
            sf |= {"prefix", "fuse", "head", "mask", "moe", "sdpa", "post", "pre", "dattn", "kvsc"}
        self.sf = sf
        # B>1 decode (L=1) speedups, INKLING_BATCHFAST=all or a comma list.
        # They change only the B*L>1, L==1 graph; B=1 decode is untouched. NONBITWISE-OK class
        # vs the stock batched path (each kernel computes every row with the B=1 decode arithmetic).
        bf = {t for t in os.environ.get("INKLING_BATCHFAST", "").split(",") if t}
        if "all" in bf:
            bf |= {"moe", "head", "fuse", "sdpa", "post", "pre", "dattn"}
        self.bf = bf
        self._prepare()

    # ------------------------------------------------------------ prepare
    def _prepare(self):
        """Build every lazily cached array before tracing."""
        arrays = []
        model, lm = self.model, self.lm
        for layer in model.layers:
            attn, moe = layer.self_attn, layer.mlp
            if self.k_fuse and hasattr(attn, "fast_prepare"):
                attn.fast_prepare()
                arrays += [w for w, _ in attn.__dict__["_fast_fused"].values()]
            if (fk.enabled("glue") or fk.enabled("route")) and hasattr(
                moe, "_fast_router_weight"
            ):
                arrays.append(moe._fast_router_weight())
            if fk.enabled("glue"):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    lin = getattr(moe.switch_mlp, name)
                    if hasattr(lin, "global_scale") and lin.global_scale.size in (
                        1,
                        lin.num_experts,
                    ) and "pre_quant_scale" not in lin:
                        arrays.append(fk.global_scale_factor(lin))
                arrays.append(fk.sconv_divisor(self.cfg.sconv_kernel_size))
            if fk.enabled("fusemoe") and hasattr(moe, "fast_prepare"):
                f = moe.fast_prepare()
                if f is not None:
                    arrays += [f["weight"], f["scales"]]
                    if f["factor"] is not None:
                        arrays.append(f["factor"])
            if self.batch_ctrl and hasattr(layer, "controllers") and not hasattr(
                layer, "_fd_ctrl3"
            ):
                cs = [layer.controllers[n] for n in ("attn", "k", "v")]
                stk = {}
                for wname in ("down_weight", "forget_weight", "gain_weight"):
                    w = mx.stack([getattr(c, wname) for c in cs])
                    mx.eval(w)
                    for i, c in enumerate(cs):  # originals become views (memory +0)
                        setattr(c, wname, w[i])
                    stk[wname] = w
                layer.__dict__["_fd_ctrl3"] = stk
                arrays += list(stk.values())
        if fk.enabled("mega_moe"):
            k = self.cfg.num_experts_per_tok
            mk.lhs_const(1), mk.lhs_const(k)
        head_kernel = (
            fk.enabled("head")
            and fk._on_gpu()
            and type(lm.lm_head) is nn.Linear
            and lm.lm_head.weight.dtype == mx.bfloat16
        )
        # ``head`` (bitwise) covers only M = B*L == 1; any other M takes stock's
        # cached-FP32 head (``head_approx`` covers M <= 15). That cache must exist
        # before tracing, so it is built here, or later in _ensure_head_cache.
        self._head_m1 = head_kernel and lm.config.hidden_size % 128 == 0
        if not head_kernel:
            arrays += self._head_cache_arrays()
        if "sdpa" in self.sf or "sdpa" in self.bf:  # rel_proj^T for the L-row / B-row SDPA kernels
            from . import spec_attn as sa

            arrays += sa.prepare(model)
        mx.eval(arrays)

    def _head_cache_arrays(self):
        lm = self.lm
        if type(lm.lm_head) is not nn.Linear or "bias" in lm.lm_head:
            return []
        cached = lm.get("_head_projection_cache")
        w = lm.lm_head.weight
        if cached is None or cached[0] is not w:
            lm._head_projection_cache = (w, 1, 0, w.astype(mx.float32))
        return [lm._head_projection_cache[-1]]

    def _ensure_head_cache(self, m):
        if m == 1 and self._head_m1:
            return
        if m <= 15 and fk.enabled("head_approx"):
            return
        arrs = self._head_cache_arrays()
        if arrs:
            mx.eval(arrs)

    # ------------------------------------------------------------------ state
    def _bucket_for(self, n):
        for b in self.buckets:
            if b >= n:
                return b
        raise Unsupported(f"context {n} exceeds largest bucket {self.buckets[-1]}")

    def from_cache(self, cache, materialize=False):
        """Convert a stock prompt cache (after prefill) into static state.
        Accepts single-row caches and batch (left-padded) caches.

        Lazy by default: the cache work the prompt logits
        did not need (last chunk K/V stores, conv states; ~18 ms at 1k) and
        the copies here run inside the first decode step, as they do for the
        stock cache. materialize=True evaluates the state now."""
        W, R = self.window, self.ring
        rows = [_kv_rows(c[0]) for c in cache]
        lengths = rows[0][2]
        if any(r[2] != lengths for r in rows):
            raise ValueError("inconsistent cache offsets across visits")
        if (self.exact and len(set(lengths)) > 1
                and os.environ.get("INKLING_FAST_DECODE_PADDED", "0") != "1"):
            # Review F2: on left-padded batches the exact path is not bitwise
            # vs stock's padded layout (key indices and SDPA N differ), and
            # FP32-exactness is unproven. Callers use the normal path (INKLING_FAST kernels stay on).
            # INKLING_FAST_DECODE_PADDED=1 opts in (non-bitwise).
            # Equal-length batches were only checked in limited configurations
            # (see DETAILS.md "Known limits").
            raise Unsupported("left-padded batch: exact FastDecoder is bitwise only for equal row lengths")
        B = len(lengths)
        cap = self._bucket_for(max(lengths) + 1)  # raises Unsupported if too long
        kbufs, vbufs, convs = [], [], []
        swa_pos = [[-1] * R for _ in range(B)]
        for visit, ((k, v, _), c) in enumerate(zip(rows, cache)):
            P = k.shape[2]
            _, nkv, _, D = k.shape
            S = R if self.sliding[visit] else cap
            kb = mx.zeros((B, nkv, S + 1, D), k.dtype)
            vb = mx.zeros((B, nkv, S + 1, D), v.dtype)
            for b, n in enumerate(lengths):
                m = min(n, P, W) if self.sliding[visit] else min(n, P)
                if m <= 0:
                    continue
                if self.sliding[visit] and visit == self.sliding.index(True):
                    for p in range(n - m, n):
                        swa_pos[b][p % R] = p
                # Slots of positions n-m..n-1 form at most two contiguous runs
                # (ring wrap; m <= W <= R): slice updates, not an index-array
                # scatter (same values; conversion time).
                start = (n - m) % R if self.sliding[visit] else n - m
                r1 = min(m, (R if self.sliding[visit] else S) - start)
                src = P - m
                kb[b, :, start : start + r1] = k[b, :, src : src + r1]
                vb[b, :, start : start + r1] = v[b, :, src : src + r1]
                if r1 < m:
                    kb[b, :, : m - r1] = k[b, :, src + r1 : P]
                    vb[b, :, : m - r1] = v[b, :, src + r1 : P]
            kbufs.append(kb)
            vbufs.append(vb)
            states = list(c[1].cache)
            if any(s is None for s in states):
                raise ValueError("conv state missing; prefill at least 1 token")
            convs.append([s.astype(mx.float32) for s in states])
        self.state = {
            "q_pos": mx.array(lengths, mx.int32),
            "swa_pos": mx.array(swa_pos, mx.int32),
            "k": kbufs,
            "v": vbufs,
            "conv": convs,
        }
        self._known, self._rounds, self.cap = list(lengths), [], cap
        if materialize:
            mx.eval(self.state)
        return self

    def _settle_kvip(self):
        """mega_kvip writes KV rows in place, so the state's
        k/v arrays do not depend (in MLX's graph) on the qsm kernels that
        wrote them.  Before anything outside a qsm step reads or copies the
        buffers (bucket growth, to_cache, a verify round), run the pending
        steps to completion.  Their conv states depend on every visit's qsm
        (qsm -> SDPA -> mlp sconv), so evaluating the state is enough.  Rare
        (growth, fallback, step->verify switch): no steady-state cost."""
        if self._kvip_pending and self.state is not None:
            mx.eval(self.state)
        self._kvip_pending = False

    def _fit_bounded(self, L):
        """Fit the bucket for a call of L tokens when the true position determines it.
        Each unobserved verify round advanced the true position by 1..L_k, so
        need lies in [lo, hi]; if both ends map to the same bucket, that bucket
        is exact (shrinking if an earlier upper bound over-grew it). Otherwise
        the caller grows to the upper bound (unavoidable without a host sync)."""
        lo = max(self._known) + len(self._rounds) + L
        hi = self.pos + L
        try:
            if self._bucket_for(lo) == self._bucket_for(hi):
                self._fit(hi)
        except Unsupported:
            pass

    def _fit(self, need):
        """Set cap to the smallest bucket >= need, shrinking global buffers if a
        verify grew them from an upper-bound position (review F1). Only call
        when the host position is exact (no unobserved rounds). Slots >= the new
        cap hold no live keys (pos + 1 <= need <= new cap); the last slot (the
        permanent zero row) is kept."""
        try:
            cap = self._bucket_for(need)
        except Unsupported:
            return
        if cap >= self.cap:
            return
        # integ: shrinking copies global KV rows (as _grow), so settle pending
        # in-place KVIP writes first.  Only here, not on every step's fit check.
        self._settle_kvip()
        st = self.state
        for i, s in enumerate(self.sliding):
            if s:
                continue
            for key in ("k", "v"):
                b = st[key][i]
                old = b.shape[2] - 1
                st[key][i] = mx.concatenate([b[:, :, :cap], b[:, :, old:]], axis=2)
        self.cap = cap

    def _grow(self, need):
        self._settle_kvip()
        cap = self._bucket_for(need)
        st = self.state
        for i, s in enumerate(self.sliding):
            if s:
                continue
            for key in ("k", "v"):
                b = st[key][i]
                old = b.shape[2] - 1
                pad = mx.zeros((*b.shape[:2], cap - old, b.shape[3]), b.dtype)
                st[key][i] = mx.concatenate([b[:, :, :old], pad, b[:, :, old:]], axis=2)
        self.cap = cap

    # ------------------------------------------------------------- flatten
    def _flatten(self, st):
        flat = [st["q_pos"], st["swa_pos"], *st["k"], *st["v"]]
        for c in st["conv"]:
            flat.extend(c)
        return flat

    def _unflatten(self, flat):
        n = len(self.visits)
        q_pos, swa_pos = flat[0], flat[1]
        k = list(flat[2 : 2 + n])
        v = list(flat[2 + n : 2 + 2 * n])
        rest = flat[2 + 2 * n :]
        conv = [list(rest[4 * i : 4 * i + 4]) for i in range(n)]
        return {"q_pos": q_pos, "swa_pos": swa_pos, "k": k, "v": v, "conv": conv}

    # ------------------------------------------------------------- the step
    def _sconv(self, module, x, conv, extra=None):
        """Call a (Dynamic)ShortConvolution; in verify mode also record the
        conv state after every prefix (stock _conv_state_history, no mask):
        state_j = concat([state, x], 1)[:, j+1 : j+4], as [B, L, W-1, C]."""
        state = conv[module.conv_idx]
        out = module(x, conv, None) if extra is None else module(x, extra, conv, None)
        if self._prefixes is not None:
            L = x.shape[1]
            ctx = mx.concatenate([state, x.astype(mx.float32)], axis=1)
            w = state.shape[1]
            if "prefix" in self.sf:
                # INKLING_SPECFAST=prefix: keep ctx [B, w+L, C]; the commit gathers rows
                # accepted+1 .. accepted+w (same values as the stacked prefixes).
                self._prefixes[-1][module.conv_idx] = ("ctx", ctx, w)
            else:
                self._prefixes[-1][module.conv_idx] = mx.stack(
                    [ctx[:, j + 1 : j + 1 + w] for j in range(L)], axis=1
                )
        return out

    def _attention(self, attn, x, key_input, value_input, conv, kb, vb, st, tau, dep=None):
        """Mirror of Attention.__call__ / _fast_call with static buffers (L>=1)."""
        b, length, _ = x.shape
        cfg = self.cfg
        fused = (
            attn.__dict__.get("_fast_fused")
            if (self.k_fuse and (b * length == 1
                                 or ("fuse" in self.sf and self._prefixes is not None
                                     and b * length <= 8)
                                 or ("fuse" in self.bf and length == 1 and b <= 16)))
            else None
        )
        same_kv_input = key_input is None and value_input is None
        kvsc = False
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        if fused and "kv" in fused and same_kv_input:
            w, splits = fused["kv"]
            k, v = mx.split(x @ w.T, splits, axis=-1)
        else:
            k = attn.k_proj(key_input)
            v = attn.v_proj(value_input)
        if (
            attn.is_sliding
            and self.mega_sconv2
            and self._prefixes is None
            and mk.sconv2_supported(
                attn.k_sconv, attn.v_sconv, k, v,
                conv[attn.k_sconv.conv_idx], conv[attn.v_sconv.conv_idx],
            )
        ):
            # INKLING_FAST=mega_sconv2: k and v sconv (+ state) in one launch.
            ki, vi = attn.k_sconv.conv_idx, attn.v_sconv.conv_idx
            k, v, conv[ki], conv[vi] = mk.sconv_pair(
                attn.k_sconv, attn.v_sconv, k, v, conv[ki], conv[vi]
            )
        elif attn.is_sliding and self._sa_kvsc_ok(attn, x, conv, b):
            kvsc = True  # INKLING_SPECFAST=kvsc: sconv inside the clip kernel below
        elif attn.is_sliding:
            k = self._sconv(attn.k_sconv, k, conv)
            v = self._sconv(attn.v_sconv, v, conv)
        elif (
            self.mega_dsconv
            and self._prefixes is None
            and mk.dsconv_supported(attn.k_sconv, k, key_input, conv[attn.k_sconv.conv_idx])
            and mk.dsconv_supported(attn.v_sconv, v, value_input, conv[attn.v_sconv.conv_idx])
        ):
            ki, vi = attn.k_sconv.conv_idx, attn.v_sconv.conv_idx
            k, v, conv[ki], conv[vi] = mk.dsconv_pair(
                attn.k_sconv, attn.v_sconv, k, v, key_input, value_input, conv[ki], conv[vi]
            )
        else:
            k = self._sconv(attn.k_sconv, k, conv, key_input)
            v = self._sconv(attn.v_sconv, v, conv, value_input)
        if fused and "qrg" in fused:
            w, splits = fused["qrg"]
            q, r, gate = mx.split(x @ w.T, splits, axis=-1)
            sinks = None  # sink_proj stays unfused (see Attention._FUSE_GROUPS)
        else:
            q, r, gate, sinks = attn.q_proj(x), attn.r_proj(x), None, None
        kvz = (
            self.mega_kvz
            and self.exact
            and b * length == 1
            and self._prefixes is None
            and (not attn.is_sliding or self.cap <= self._slice_cap)
            and fk.qkv_clip_supported(attn, cfg, x)
        )
        qsm_mask, kvip = None, False
        if (
            self.mega_qsm and self.exact and b * length == 1
            and self._prefixes is None and sinks is None
            and mk.qsm_supported(attn, cfg, x, attn.rel_proj.astype(x.dtype))
        ):
            if attn.is_sliding and self.cap > self._slice_cap:
                ncol, gather, win = self.window + 1, True, self.window
            else:
                ncol, gather = (self.cap + 1 if attn.is_sliding else kb.shape[2]), False
                win = attn.sliding
            kvip = self.mega_kvip and kvz
            q, k, v, r, qsm_mask = mk.qkv_sink_mask(
                q, k, v, r, attn, cfg,
                tau.reshape(-1) if (not attn.is_sliding and tau is not None) else None,
                x, attn.rel_proj.astype(x.dtype), st["qp"][:, 0], ncol, win, gather, kvz,
                kb if kvip else None, vb if kvip else None,
                dep=dep if kvip else None,
            )
            sinks = qsm_mask  # marks the sink as already folded into the mask
        elif kvsc:
            # INKLING_SPECFAST=kvsc (specfast-attn): k/v ShortConvolutions + the qkv clip in
            # one launch; the kernel also writes the k/v rollback contexts.
            from . import spec_attn as sa

            ki, vi = attn.k_sconv.conv_idx, attn.v_sconv.conv_idx
            q, k, v, r, kctx, vctx = sa.kvsc_clip(
                q, k, v, r, attn, cfg, attn.k_sconv, attn.v_sconv, conv[ki], conv[vi])
            w = attn.k_sconv.kernel_size - 1
            self._prefixes[-1][ki] = ("ctx", kctx, w)
            self._prefixes[-1][vi] = ("ctx", vctx, w)
        elif self.qkvclip and fk.qkv_clip_supported(attn, cfg, x):
            # INKLING_FAST=qkvclip: the clips, the value norm and the tau
            # scaling in one kernel (bitwise).
            q, k, v, r = fk.qkv_clip(
                q, k, v, r, attn, cfg,
                tau.reshape(-1) if (not attn.is_sliding and tau is not None) else None,
                zrow=kvz,
            )
        else:
            q = l2_clip(
                q.reshape(b, length, attn.n_heads, attn.head_dim),
                cfg.qk_l2_clip_threshold,
                attn.q_scale,
            )
            k = l2_clip(
                k.reshape(b, length, attn.n_kv, attn.head_dim), cfg.qk_l2_clip_threshold
            )
            v = v.reshape(b, length, attn.n_kv, attn.head_dim)
            if cfg.use_value_norm_no_gamma:
                v = mx.fast.rms_norm(
                    v.astype(mx.float32), None, cfg.rms_norm_eps
                ).astype(x.dtype)
            r = l2_clip(
                r.reshape(b, length, attn.n_heads, cfg.d_rel),
                cfg.rel_l2_clip_threshold,
                attn.rel_scale,
            )
            if not attn.is_sliding and tau is not None:
                q = (q.astype(mx.float32) * tau).astype(x.dtype)
                r = (r.astype(mx.float32) * tau).astype(x.dtype)
        q_rows = q  # [B, L, H, D] before the transpose (mega_sdpa)
        if kvz:  # k, v already [1, n_kv, 2, D] (row 1 = zero sink row)
            q = q.transpose(0, 2, 1, 3)
        else:
            q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        qp = st["qp"]  # [B, L] positions of this call's tokens
        S = kb.shape[2] - 1
        if attn.is_sliding:
            slots = qp % self.ring
            slot_pos = st["swa_pos_new"]
        else:
            slots = qp
            ar = mx.arange(S, dtype=mx.int32)[None]
            slot_pos = mx.where(ar <= qp[:, -1:], ar, -1)
        if kvip:
            # written in place by the qsm kernel: ordered before SDPA via the
            # mask, after the previous step's SDPA via ``dep`` (F1)
            pass
        elif kvz:
            kb = mx.slice_update(kb, k, slots.reshape(1), axes=(2,))
            vb = mx.slice_update(vb, v, slots.reshape(1), axes=(2,))
        else:
            kb = _write_slots(kb, k, slots)
            vb = _write_slots(vb, v, slots)
        if sinks is None:
            sinks = attn.sink_proj(x)  # [B, L, H]
        proj = attn.rel_proj.astype(x.dtype)
        scale = attn.head_dim**-0.5
        decode = length == 1
        gather = attn.is_sliding and self.cap > self._slice_cap
        if not decode and "sdpa" in self.sf and self._prefixes is not None:
            from . import spec_attn as sa

            if sa.sdpa_supported(attn, q_rows, kb, r, sinks):
                # INKLING_SPECFAST=sdpa (specfast-attn): rel-bias mask + SDPA + sink + output
                # gate for the L verify rows in one launch (spec_attn.sdpa_rows).
                if gate is None:
                    gate = attn.gate_proj(x)
                out = sa.sdpa_rows(attn, q_rows, kb, vb, r, slot_pos if attn.is_sliding else None,
                                   qp, sinks, gate if fk.enabled("glue") else None,
                                   self.ring, scale, self.cap)
                if not fk.enabled("glue"):
                    out = (out.astype(mx.float32)
                           * mx.sigmoid(gate.astype(mx.float32))).astype(x.dtype)
                return attn.o_proj(out), kb, vb
        if decode and b > 1 and "sdpa" in self.bf and self._prefixes is None:
            from . import batch_kernels as bk

            if bk.sdpa_supported(attn, q_rows, kb, r, sinks):
                # INKLING_BATCHFAST=sdpa: rel-bias mask + SDPA + sink + output gate for the B decode
                # rows in one launch (+ a combine launch for long buffers); only each row's live
                # positions are read (no mask tensor, no zero sink row, no ring gather).
                if gate is None:
                    gate = attn.gate_proj(x)
                glue = fk.enabled("glue")
                out = bk.sdpa_brows(attn, q_rows, kb, vb, r, slot_pos if attn.is_sliding else None,
                                    qp[:, 0], sinks, gate if glue else None, self.ring, scale, self.cap)
                if not glue:
                    out = (out.astype(mx.float32)
                           * mx.sigmoid(gate.astype(mx.float32))).astype(x.dtype)
                return attn.o_proj(out), kb, vb
        if (
            decode and self.exact and self.mega_sdpa and b == 1
            and qsm_mask is not None and (kvz or gather)
            and mk.sdpa_supported(q_rows, kb, vb, qsm_mask, qsm_mask.shape[-1],
                                  gather=gather, ring=self.ring)
        ):
            # INKLING_FAST=mega_sdpa: sliced regime needs the zero sink row
            # already in the buffer (kvz); gather regime reads the ring in
            # temporal order through the slot remap (no ring_gather copy).
            out = mk.sdpa_exact(
                q_rows, kb, vb, qsm_mask, qp[:, 0], qsm_mask.shape[-1], gather,
                self.window, self.ring, scale,
            ).reshape(b, length, -1)
        elif decode and self.exact:
            k_ext, v_ext, kb, vb, mask = self._exact_operands(
                attn, r, proj, sinks, kb, vb, qp[:, 0], zero_written=kvz,
                mask=qsm_mask,
            )
            out = mx.fast.scaled_dot_product_attention(
                q, k_ext, v_ext, scale=scale, mask=mask
            )
        elif decode and self.use_sinks and b == 1:
            if self.k_mask:
                mask = fk.slot_mask(r, proj, slot_pos, qp[:, 0], attn.sliding)
            else:
                mask = _rel_slot_mask(
                    r, proj, slot_pos, qp, attn.sliding, attn.rel_extent, x.dtype
                )
            out = mx.fast.scaled_dot_product_attention(
                q, kb[:, :, :S], vb[:, :, :S], scale=scale, mask=mask,
                sinks=sinks.reshape(-1),
            )
        else:
            if self.k_mask and decode:
                mask = fk.slot_mask(
                    r, proj, slot_pos, qp[:, 0], attn.sliding, sinks=sinks
                )
            elif "mask" in self.sf and fk._on_gpu():
                # INKLING_SPECFAST=mask: L-row mask kernel (serial FP32 dot per row, as the
                # decode mask kernel) with the sink column; one launch.
                from . import spec_kernels as sk

                mask = sk.slot_mask_rows(r, proj, slot_pos, qp, attn.sliding, sinks, x.dtype)
            else:
                # Sink logits are per query, so L>1 keeps the zero K/V slot.
                mask = _rel_slot_mask(
                    r, proj, slot_pos, qp, attn.sliding, attn.rel_extent, x.dtype
                )
                mask = mx.concatenate(
                    [mask, sinks.transpose(0, 2, 1)[..., None]], axis=-1
                )
            out = mx.fast.scaled_dot_product_attention(q, kb, vb, scale=scale, mask=mask)
        if out.ndim == 4:
            out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        if gate is None:
            gate = attn.gate_proj(x)
        if (
            self.mega_oproj and b * length == 1 and fk.enabled("glue")
            and mk.oproj_supported(attn.o_proj, out, gate)
        ):
            # INKLING_FAST=mega_oproj: output gate applied inside the o_proj GEMV.
            return mk.gate_oproj(attn.o_proj, out, gate), kb, vb
        if fk.enabled("glue"):
            out = fk.output_gate(out, gate)
        else:
            out = (
                out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
            ).astype(x.dtype)
        return attn.o_proj(out), kb, vb

    def _exact_operands(self, attn, r, proj, sinks, kb, vb, p, zero_written=False,
                        mask=None):
        """Stock decode operands from the static buffers (L=1).

        Stock SDPA sees keys [temporal(S keys); zero] and mask
        [banded(S); sink], S = keys in cache. Returned here:
        * global, and sliding while cap <= self._slice_cap (no wrap possible):
          buffer[:cap + 1] (slot j = position j); slot p + 1 is zeroed and
          carries the sink logit; slots > p + 1 are masked (-1e30): trailing
          masked keys are exact no-ops in both SDPA kernels.
        * sliding otherwise: a gather into temporal order, oldest first,
          [W + 1] rows; rows >= S read the permanent zero slot, row S carries
          the sink logit, rows > S are masked. N = W + 1 = stock's N once the
          window is full (and both >= 1024 -> same 2-pass config before that).
        Rel bias: slot_mask kernel (same serial FP32 loop as the stock mask
        kernel).
        """
        B = kb.shape[0]
        n_kv, D = kb.shape[1], kb.shape[3]
        sink_col = None if mask is not None else sinks.transpose(0, 2, 1)[..., None]  # [B, H, 1, 1]
        p_col = p[:, None]  # [B, 1]
        sliced = (not attn.is_sliding) or self.cap <= self._slice_cap
        if self.xmask:
            # INKLING_FAST=xmask: one mask kernel (positions, window, sink
            # column worked out from p), and in the sliced regime the zero
            # sink row goes straight into the full buffer, so the sliding
            # ring needs no sliced copy and write-back (SDPA reads the view).
            if sliced:
                n = self.cap + 1
                if not zero_written:
                    zeros = mx.zeros((B, n_kv, 1, D), kb.dtype)
                    zslot = p_col + 1  # <= cap
                    kb = _write_slots(kb, zeros, zslot)
                    vb = _write_slots(vb, zeros, zslot)
                k_ext = kb[:, :, :n] if attn.is_sliding else kb
                v_ext = vb[:, :, :n] if attn.is_sliding else vb
                if mask is None:
                    mask = fk.exact_mask(r, proj, p, sinks, k_ext, attn.sliding, False)
                assert mask.shape[-1] == k_ext.shape[2]
                return k_ext, v_ext, kb, vb, mask
            W, R = self.window, self.ring
            k_ext, v_ext = fk.ring_gather(kb, vb, p, W, R)
            if mask is None:
                mask = fk.exact_mask(r, proj, p, sinks, k_ext, W, True)
            assert mask.shape[-1] == k_ext.shape[2]
            return k_ext, v_ext, kb, vb, mask
        if sliced:
            n = self.cap + 1
            if attn.is_sliding:
                kb_full, vb_full = kb, vb
                kb, vb = kb[:, :, :n], vb[:, :, :n]
            zeros = mx.zeros((B, n_kv, 1, D), kb.dtype)
            zslot = p_col + 1  # <= cap
            kb = _write_slots(kb, zeros, zslot)
            vb = _write_slots(vb, zeros, zslot)
            ar = mx.arange(n, dtype=mx.int32)[None]
            slot_pos = mx.where(ar <= p_col, ar, -1)
            mask = fk.slot_mask(r, proj, slot_pos, p, attn.sliding)
            mask = mx.where((ar == zslot)[:, None, None, :], sink_col, mask)
            k_ext, v_ext = kb, vb
            if attn.is_sliding:
                # Write the zeroed rows back into the full ring state.
                kb = mx.slice_update(kb_full, kb, mx.array([0]), axes=(2,))
                vb = mx.slice_update(vb_full, vb, mx.array([0]), axes=(2,))
            return k_ext, v_ext, kb, vb, mask
        W, R = self.window, self.ring
        t = mx.arange(W + 1, dtype=mx.int32)[None]
        S = mx.minimum(p_col + 1, W)
        base = mx.maximum(p_col - W + 1, 0)
        pos = base + t
        valid = t < S
        idx = mx.where(valid, pos % R, R)  # slot R is the permanent zero row
        slot_pos = mx.where(valid, pos, -1)
        if B == 1:
            k_ext = mx.take(kb, idx[0], axis=2)
            v_ext = mx.take(vb, idx[0], axis=2)
        else:
            gi = mx.broadcast_to(idx[:, None, :, None], (B, n_kv, W + 1, D))
            k_ext = mx.take_along_axis(kb, gi, axis=2)
            v_ext = mx.take_along_axis(vb, gi, axis=2)
        mask = fk.slot_mask(r, proj, slot_pos, p, attn.sliding)
        mask = mx.where((t == S)[:, None, None, :], sink_col, mask)
        return k_ext, v_ext, kb, vb, mask

    def _attn_res(self, module, blocks, bkeys, partial):
        if ("dattn" in self.bf and self._prefixes is None and self.reuse_keys
                and partial.ndim == 3 and partial.shape[1] == 1 and partial.shape[0] > 1):
            from . import batch_kernels as bk

            if bk.tail_brows_supported(None, None, partial, len(bkeys), module.query):
                # INKLING_BATCHFAST=dattn: depth attention for the B decode rows in one launch.
                return bk.tail_brows(2, None, None, partial, None, list(bkeys), module.query)["hid"]
        if ("dattn" in self.sf and self._prefixes is not None and self.reuse_keys
                and partial.ndim == 3 and partial.shape[1] > 1):
            from . import spec_attn as sa

            if sa.tail_supported(None, None, partial, len(bkeys), module.query):
                # INKLING_SPECFAST=dattn (specfast-attn): depth attention for the verify
                # rows (1 or NQ queries) in one launch.
                return sa.tail_rows(2, None, None, partial, None, list(bkeys), module.query)["hid"]
        if (
            self.attnres
            and self.reuse_keys
            and module.query.ndim == 1
            and fk.attn_res_supported(partial, module.query, len(bkeys))
        ):
            # INKLING_FAST=attnres: the whole depth attention in one kernel.
            return fk.attn_res(bkeys, partial, module.query, raw_partial=self.mega_attnres)
        if self.reuse_keys:
            P = min(self._xp_stable, len(bkeys)) if self.xpass else 0
            if P and partial.size // partial.shape[-1] <= 16:
                # INKLING_FAST=xpass: depth logits of the blocks that repeat in
                # every loop pass, computed once per step (not always bitwise in batches; see
                # fast_kernels.attention_residual_keyed).
                entry = self._xp_memo.get(id(module))
                prefix = tuple(bkeys[:P])
                if entry is None or len(entry[0]) != P or any(
                    a is not b for a, b in zip(entry[0], prefix)
                ):
                    entry = (prefix, fk.depth_scores(bkeys[:P], module.query))
                    self._xp_memo[id(module)] = entry
                return fk.attn_res_prefix(
                    entry[1], bkeys[:P], bkeys[P:], partial, module.query
                )
            return _attn_res_keys_c(bkeys, partial, module.query, True)
        return module(blocks, partial)

    # MoE for verify (L>1): group all B*L*k routes by expert so each expert's
    # NVFP4 gate/up weights are read once per group (meant to match decode's QMV).
    def _moe_dedup_ok(self, moe):
        from ..masked_quantized import supports_masked_qmv

        sm = moe.switch_mlp
        return (
            self.dedup
            and moe.sharding_group is None
            and fk._on_gpu()
            and supports_masked_qmv(sm.gate_proj, input_alignment=256)
            and supports_masked_qmv(sm.up_proj, input_alignment=256)
        )

    def _batch_moe(self, moe, x, routed=False):
        """INKLING_BATCHFAST=moe: B>1 decode rows through spec_moe (decode-row arithmetic over the
        distinct experts of the step). routed=True returns (per-route outputs, weights) like
        SparseMoE._routed (the sconv_tail path combines). None -> caller's normal path (MXFP8 layers)."""
        if not ("moe" in self.bf and x.shape[1] == 1 and x.shape[0] > 1 and self._prefixes is None
                and fk.enabled("route") and fk.enabled("fusemoe") and fk.enabled("glue")):
            return None
        from . import spec_moe

        out = spec_moe.moe_verify(moe, x, moe.fast_prepare(), routed=routed)
        if out is None:
            # make the stock fallback visible (B > spec_moe.MAX_ROWS, MXFP8 layers, ...)
            key = (x.shape[0], str(x.dtype), routed)
            if key not in _BMOE_WARNED:
                _BMOE_WARNED.add(key)
                import sys
                print(f"[inkling batchfast] moe: B={x.shape[0]} falls back to the stock MoE on some layers "
                      f"(spec_moe.supported refused; MAX_ROWS={spec_moe.MAX_ROWS}, MXFP8 layers always stock)",
                      file=sys.stderr, flush=True)
        return out

    def _moe(self, moe, x):
        if x.shape[1] == 1:
            out = self._batch_moe(moe, x)
            return out if out is not None else moe(x)
        if ("moe" in self.sf and fk.enabled("route") and fk.enabled("fusemoe")
                and fk.enabled("glue")):
            # INKLING_SPECFAST=moe: decode-row arithmetic on an expert grid (spec_moe.py).
            from . import spec_moe

            out = spec_moe.moe_verify(moe, x, moe.fast_prepare())
            if out is not None:
                return out
        if not self._moe_dedup_ok(moe):
            if (self.ds_dedup_mxfp8 and moe.sharding_group is None and fk._on_gpu()
                    and x.dtype in (mx.bfloat16, mx.float16, mx.float32)):
                from . import verify_dedup as vd
                from .language import _combine_routes

                if vd.supported(moe.switch_mlp):
                    indices, weights = vd.route_tokenwise(moe, x)
                    fused = moe.__dict__.get("_fast_gate_up_state")
                    fused = fused[1] if (fused is not None and fk.enabled("fusemoe")) else None
                    out = vd.dedup_routes(moe.switch_mlp, x, indices, self.dedup_v, fused)
                    return _combine_routes(out, weights)
            # MXFP8 etc.: run each position through the decode path so verify
            # rows equal decode rows exactly (stock's L>1 verifier is not
            # bitwise at every shape).
            if self.tokenwise_moe:
                return mx.concatenate(
                    [moe(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1
                )
            return moe(x)
        from ..masked_quantized import _global_scale, supports_masked_qmv
        from .language import _combine_routes, _route

        if self.ds_route_tokenwise:
            from . import verify_dedup as vd

            indices, weights = vd.route_tokenwise(moe, x)
        else:
            indices, weights = _route(
                x,
                moe._fast_router_weight() if fk.enabled("glue") else moe.gate_weight,
                moe.e_score_correction_bias,
                moe.top_k,
                moe.route_scale,
            )
        sm = moe.switch_mlp
        B, L, D = x.shape
        k = indices.shape[-1]
        flat = indices.reshape(-1).astype(mx.int32)
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        sel = flat[order]
        values = x.reshape(-1, D)[order // k]
        n = sm.gate_proj.output_dims
        fused = moe.__dict__.get("_fast_gate_up_state")
        fused = fused[1] if (fused is not None and fk.enabled("fusemoe")) else None
        if fused is not None and fused["mode"] == "nvfp4":
            g = fused["factor"] if fused["factor"] is not None else _global_scale(sm.gate_proj)
            if g.size < 64:
                g = mx.concatenate([g, mx.zeros((64 - g.size,), mx.float32)])
            gu = _grouped_qmv(values, fused["weight"], fused["scales"], g, sel, 2 * n,
                              self.dedup_v)
            gate, up = gu[..., :n], gu[..., n:]
        else:
            up = _grouped_qmv(values, sm.up_proj.weight, sm.up_proj.scales,
                              _global_scale(sm.up_proj), sel, n, self.dedup_v)
            gate = _grouped_qmv(values, sm.gate_proj.weight, sm.gate_proj.scales,
                                _global_scale(sm.gate_proj), sel, n, self.dedup_v)
        act = sm.activation(up, gate)
        if self.ds_dedup_down:
            from . import verify_dedup as vd

            if vd.kind_of(sm.down_proj) is not None:
                out = vd.grouped_qmv(act, sm.down_proj, sel, self.dedup_v)
                out = out[inv].reshape(B, L, k, D)
                return _combine_routes(out, weights)
        if supports_masked_qmv(sm.down_proj, input_alignment=256):
            out = _grouped_qmv(act, sm.down_proj.weight, sm.down_proj.scales,
                               _global_scale(sm.down_proj), sel, D, self.dedup_v)
        else:
            out = sm.down_proj(
                act[:, None, None], sel[:, None], sorted_indices=False
            ).reshape(sel.size, D)
        out = out[inv].reshape(B, L, k, D)
        return _combine_routes(out, weights)

    def _layer(self, layer, x, blocks, bkeys, conv, kb, vb, st, tau, messages):
        """Mirror of DecoderLayer.__call__ (mask=None)."""
        pre, self._pre = self._pre, None
        if pre is not None and pre[0] is layer:
            mixed = pre[1]  # computed by the previous visit's mega_pre kernel
        else:
            pre = None
            mixed = (
                self._attn_res(layer.attention_input, blocks, bkeys, x)
                if layer.attention_input is not None
                else x[..., None, :]
            )
        if layer.is_sliding and messages is None:
            attn, kb, vb = self._attention(
                layer.self_attn,
                pre[2] if (pre is not None and pre[2] is not None)
                else self._norms(x, [mixed[..., 0, :]], (layer.input_layernorm,))[0],
                None, None, conv, kb, vb, st, tau,
                dep=conv[layer.mlp_sconv.conv_idx],
            )
        else:
            parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            if messages is not None:
                # Bitwise vs per-controller matmuls verified up to M=8 (M4 Pro);
                # at M=16 the batched GEMM tiles differently.
                stk = (
                    layer.__dict__.get("_fd_ctrl3")
                    if self.batch_ctrl and x.shape[0] * x.shape[1] <= 8
                    else None
                )
                if stk is not None and self._mega_ctrl_ok(
                    x, mixed, messages, stk["down_weight"], stk["forget_weight"],
                    stk["gain_weight"], 3,
                ):
                    D = x.shape[-1]
                    nq = mixed.shape[-2]
                    out3 = mk.controllers(
                        mixed.reshape(nq, D), messages.reshape(-1, D), 0,
                        stk["down_weight"], stk["forget_weight"], stk["gain_weight"],
                        layer.controllers["attn"].eps, 3, nq,
                    )
                    # out3[i:i+1], not out3[i]: a leading int index is a Gather (copy)
                    parts = [out3[i : i + 1].reshape(x.shape[:-1] + (D,)) for i in range(3)]
                elif stk is not None:
                    fn = _controllers3_c if fk.enabled("glue") else _controllers3
                    out3 = fn(
                        mx.stack(parts),
                        mx.moveaxis(messages[..., :3, :], -2, 0),
                        stk["down_weight"], stk["forget_weight"], stk["gain_weight"],
                        layer.controllers["attn"].eps,
                    )
                    parts = [out3[i] for i in range(3)]
                else:
                    parts = [
                        layer.controllers[name](part, messages[..., i, :])
                        for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))
                    ]
            norms = (
                (layer.input_layernorm,) * 3
                if layer.is_sliding
                else (
                    layer.q_input_layernorm,
                    layer.k_input_layernorm,
                    layer.v_input_layernorm,
                )
            )
            parts = self._norms(x, parts, norms)
            q_in, k_in, v_in = parts
            attn, kb, vb = self._attention(
                layer.self_attn, q_in, k_in, v_in, conv, kb, vb, st, tau,
                dep=conv[layer.mlp_sconv.conv_idx],
            )
        stail = self.stail and self._prefixes is None and self._stail_ok(attn)
        spost = self._sa_post_ok(layer, conv, attn, x, bkeys)
        bpost = not spost and self._b_post_ok(layer, conv, attn, x, bkeys)
        spost = spost or bpost
        ac = layer.attn_sconv
        boundary = layer.is_block_boundary
        post = (
            stail
            and not spost
            and self.mega_post
            and self.attnres
            and self.reuse_keys
            and x.shape[0] * x.shape[1] == 1
            and mk.post_supported(
                ac, conv[ac.conv_idx], attn, x,
                len(bkeys) + (1 if boundary else 0), layer.mlp_input.query,
            )
        )
        hn = None
        if spost:
            # INKLING_SPECFAST=post (specfast-attn): attn sconv + residual (+ boundary key)
            # + mlp_input depth attention (+ norm) per verify row in one launch.
            from . import spec_attn as sa

            ac, pn = layer.attn_sconv, layer.post_attention_layernorm
            use_norm = messages is None and sa.norm_ok(pn)
            if bpost:
                # INKLING_BATCHFAST=post: the same per B decode row (own conv state).
                from . import batch_kernels as bk

                r = bk.tail_brows(0, ac, conv[ac.conv_idx], attn, x, list(bkeys),
                                  layer.mlp_input.query, boundary=boundary,
                                  norm=pn if use_norm else None, want_hid=not use_norm)
                conv[ac.conv_idx] = r["ctx"]
            else:
                r = sa.tail_rows(0, ac, conv[ac.conv_idx], attn, x, list(bkeys),
                                 layer.mlp_input.query, boundary=boundary,
                                 norm=pn if use_norm else None, want_hid=not use_norm)
                self._prefixes[-1][ac.conv_idx] = ("ctx", r["ctx"], ac.kernel_size - 1)
            residual = r["res"]
            if boundary:
                blocks.append(x)
                bkeys.append(r["bkey"])
            hidden = r["hid"][..., 0, :] if not use_norm else None
            hn = r.get("hn")
        elif post:
            # INKLING_FAST=mega_post: attn sconv tail, boundary key, mlp_input
            # depth attention and (no controller) the post-attention norm in
            # one kernel (bitwise).
            pn = layer.post_attention_layernorm
            use_norm = (
                messages is None and self.mega_norm
                and pn.weight.dtype == mx.bfloat16 and pn.weight.ndim == 1
            )
            r = mk.post_block(
                ac, conv[ac.conv_idx], attn, x, boundary, list(bkeys),
                layer.mlp_input.query, pn if use_norm else None,
            )
            residual, conv[ac.conv_idx] = r["residual"], r["nstate"]
            if boundary:
                blocks.append(x)
                bkeys.append(r["bkey"])
            hidden = r["hidden"][..., 0, :]
            if use_norm:
                hn = r["hn"]
        elif stail:
            # INKLING_FAST=sconvtail: sconv + residual add in one kernel.
            ac = layer.attn_sconv
            residual, conv[ac.conv_idx] = fk.sconv_tail(
                ac, conv[ac.conv_idx], attn, None if layer.is_block_boundary else x
            )
        else:
            residual = self._sconv(layer.attn_sconv, attn, conv).astype(mx.float32)
        if layer.is_block_boundary and not (post or spost):
            blocks.append(x)
            if self.reuse_keys:
                bkeys.append(
                    mx.fast.rms_norm(x.astype(mx.float32)[..., None, :], None, 1e-6)
                )
        elif not (stail or spost):
            residual = residual + x.astype(mx.float32)
        if not (post or spost):
            hidden = self._attn_res(layer.mlp_input, blocks, bkeys, residual)[..., 0, :]
        if messages is not None:
            mc = layer.controllers["mlp"]
            if self._mega_ctrl_ok(
                x, hidden, messages, mc.down_weight, mc.forget_weight,
                mc.gain_weight, 1,
            ):
                D = x.shape[-1]
                hidden = mk.controllers(
                    hidden.reshape(1, D), messages.reshape(-1, D), 3,
                    mc.down_weight[None], mc.forget_weight[None], mc.gain_weight[None],
                    mc.eps, 1, 1,
                ).reshape(hidden.shape)
            else:
                hidden = mc(hidden, messages[..., 3, :])
        if hn is None:
            hn = self._norms(x, [hidden], (layer.post_attention_layernorm,))[0]
        if stail and not self._b_pre_want(x):
            # expert combine + mlp sconv + residual add in one kernel.
            r = self._mega_moe(layer.mlp, hn)
            if r is None:
                r = self._batch_moe(layer.mlp, hn, routed=True)
            routes, weights = r if r is not None else layer.mlp._routed(hn)
            mc = layer.mlp_sconv
            nxt = self._next
            if (
                nxt is not None
                and self.attnres and self.reuse_keys
                and nxt[0].attention_input is not None
                and nxt[0].attention_input.query.ndim == 1
                and x.shape[0] * x.shape[1] == 1
                and mk.pre_supported(mc, conv[mc.conv_idx], routes, residual,
                                     len(bkeys), nxt[0].attention_input.query)
            ):
                # INKLING_FAST=mega_pre: expert combine + mlp sconv + residual,
                # then the next visit's depth attention (+ its input norm).
                nl, nmsg = nxt
                nn_ = nl.input_layernorm if (nl.is_sliding and not nmsg) else None
                use_norm = (
                    nn_ is not None and self.mega_norm
                    and nn_.weight.dtype == mx.bfloat16 and nn_.weight.ndim == 1
                )
                r = mk.pre_block(
                    mc, conv[mc.conv_idx], routes, residual, weights, list(bkeys),
                    nl.attention_input.query, nn_ if use_norm else None,
                )
                conv[mc.conv_idx] = r["nstate"]
                self._pre = (nl, r["mixed"], r.get("hn"))
                return r["x"], kb, vb
            out, conv[mc.conv_idx] = fk.sconv_tail(
                mc, conv[mc.conv_idx], routes, residual, weights
            )
            return out, kb, vb
        hidden = self._moe(layer.mlp, hn)
        spre = self._sa_pre(layer, conv, hidden, residual, bkeys, x)
        if spre is None:
            spre = self._b_pre(layer, conv, hidden, residual, bkeys, x)
        if spre is not None:
            return spre, kb, vb
        out = (
            residual + self._sconv(layer.mlp_sconv, hidden, conv).astype(mx.float32)
        ).astype(x.dtype)
        return out, kb, vb

    def _norms(self, x, parts, norms):
        """[norm(part).astype(x.dtype)]; INKLING_FAST=mega_norm: one launch."""
        if (
            self.mega_norm
            and x.dtype == mx.bfloat16
            and len({n.eps for n in norms}) == 1
            and 4 * len(parts) + 2 <= 31  # Metal buffer slots (x, w: buffer + strides)
            and all(mk.norm_supported(p, n.weight) for p, n in zip(parts, norms))
        ):
            out = mk.rms_norm_cast(parts, [n.weight for n in norms], norms[0].eps)
            return [out[i : i + 1].reshape(x.shape) for i in range(len(parts))]
        return [norm(part).astype(x.dtype) for norm, part in zip(norms, parts)]

    def _mega_moe(self, moe, x):
        if not (
            self.mega_moe
            and self._prefixes is None
            and x.shape[0] * x.shape[1] == 1
            and moe.sharding_group is None
        ):
            return None
        return mk.moe_routed(moe, x, moe.fast_prepare(), fk.route_topk)

    def _mega_ctrl_ok(self, x, xin, messages, dw, fw, gw, n_ctrl):
        return (
            self.mega_ctrl
            and self._prefixes is None
            and x.shape[0] * x.shape[1] == 1
            and xin.dtype == mx.float32
            and messages.dtype == mx.float32
            and messages.shape[-2] >= 4
            and mk.ctrl_supported(x, dw, fw, gw, n_ctrl)
        )

    def _sa_kvsc_ok(self, attn, x, conv, b):
        if "kvsc" not in self.sf or self._prefixes is None or not self.qkvclip:
            return False
        from . import spec_attn as sa

        ks, vs = attn.k_sconv, attn.v_sconv
        return sa.kvsc_supported(attn, self.cfg, x, ks, vs, conv[ks.conv_idx],
                                 conv[vs.conv_idx], b)

    def _sa_post_ok(self, layer, conv, attn, x, bkeys):
        if "post" not in self.sf or self._prefixes is None or not self.reuse_keys:
            return False
        from . import spec_attn as sa

        ac, q = layer.attn_sconv, layer.mlp_input.query
        nk = len(bkeys) + (1 if layer.is_block_boundary else 0)
        return (q.ndim == 1 and x.dtype == attn.dtype
                and sa.tail_supported(ac, conv[ac.conv_idx], attn, nk, q))

    def _b_post_ok(self, layer, conv, attn, x, bkeys):
        if ("post" not in self.bf or self._prefixes is not None or not self.reuse_keys
                or x.shape[0] < 2 or x.shape[1] != 1):
            return False
        from . import batch_kernels as bk

        ac, q = layer.attn_sconv, layer.mlp_input.query
        nk = len(bkeys) + (1 if layer.is_block_boundary else 0)
        return (q.ndim == 1 and attn.dtype == x.dtype
                and bk.tail_brows_supported(ac, conv[ac.conv_idx], attn, nk, q))

    def _b_pre_want(self, x):
        return ("pre" in self.bf and self._prefixes is None and self.reuse_keys and self._next is not None
                and x.shape[0] >= 2 and x.shape[1] == 1)

    def _b_pre(self, layer, conv, hidden, residual, bkeys, x):
        """INKLING_BATCHFAST=pre: mlp sconv + residual, then the next visit's attention_input depth
        attention (+ its input norm) for the B decode rows in one launch (hands over like mega_pre)."""
        if not self._b_pre_want(x) or residual.dtype != mx.float32 or hidden.dtype != x.dtype:
            return None
        from . import batch_kernels as bk, spec_attn as sa

        nl, nmsg = self._next
        ai = nl.attention_input
        mc = layer.mlp_sconv
        if ai is None or not bk.tail_brows_supported(mc, conv[mc.conv_idx], hidden, len(bkeys), ai.query):
            return None
        nn_ = (nl.input_layernorm if (nl.is_sliding and not nmsg and ai.query.ndim == 1) else None)
        use_norm = nn_ is not None and sa.norm_ok(nn_)
        r = bk.tail_brows(1, mc, conv[mc.conv_idx], hidden, residual, list(bkeys), ai.query,
                          norm=nn_ if use_norm else None, want_hid=True, out_dtype=x.dtype)
        conv[mc.conv_idx] = r["ctx"]
        self._pre = (nl, r["hid"], r.get("hn"))
        return r["res"]

    def _sa_pre(self, layer, conv, hidden, residual, bkeys, x):
        """INKLING_SPECFAST=pre (specfast-attn): mlp sconv + residual, then the next visit's
        attention_input depth attention (+ its input norm) per verify row in one launch;
        hands the result over like mega_pre (self._pre). None when not applicable."""
        if ("pre" not in self.sf or self._prefixes is None or not self.reuse_keys
                or self._next is None or residual.dtype != mx.float32
                or hidden.dtype != x.dtype):
            return None
        from . import spec_attn as sa

        nl, nmsg = self._next
        ai = nl.attention_input
        mc = layer.mlp_sconv
        if ai is None or not sa.tail_supported(
                mc, conv[mc.conv_idx], hidden, len(bkeys), ai.query):
            return None
        nn_ = (nl.input_layernorm if (nl.is_sliding and not nmsg and ai.query.ndim == 1)
               else None)
        use_norm = nn_ is not None and sa.norm_ok(nn_)
        r = sa.tail_rows(1, mc, conv[mc.conv_idx], hidden, residual, list(bkeys), ai.query,
                         norm=nn_ if use_norm else None, want_hid=True, out_dtype=x.dtype)
        self._prefixes[-1][mc.conv_idx] = ("ctx", r["ctx"], mc.kernel_size - 1)
        self._pre = (nl, r["hid"], r.get("hn"))
        return r["res"]

    def _stail_ok(self, a):
        # Same conditions as ShortConvolution's kernel path (else stock runs
        # mx.conv1d, a different reduction).
        return (
            a.shape[1] == 1
            and a.shape[0] <= 4
            and a.dtype == mx.bfloat16
            and a.shape[-1] % 128 == 0
            and self.cfg.sconv_kernel_size == 4
        )

    def _forward(self, tokens, st, captures=()):
        """tokens [B, L] at positions q_pos + arange(L).  Returns logits
        [B, L, V], the state with all L tokens written (q_pos NOT advanced),
        and the requested captures (stock capture_layer_ids semantics)."""
        cfg, model = self.cfg, self.model
        L = tokens.shape[1]
        q_pos = st["q_pos"]
        qp = q_pos[:, None] + mx.arange(L, dtype=mx.int32)[None]
        st["qp"] = qp
        # One shared slot->position update for all sliding visits.
        st["swa_pos_new"] = mx.put_along_axis(
            st["swa_pos"], qp % self.ring, qp, axis=1
        ) if (L > 1 or q_pos.shape[0] > 1) else mx.slice_update(
            st["swa_pos"], qp, (q_pos % self.ring).reshape(1), axes=(1,)
        )
        tau = None
        if cfg.log_scaling_n_floor is not None:
            positions = qp + 1
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1)
            )
            tau = tau.reshape(*qp.shape, 1, 1)
        x = model.embed(tokens)
        blocks, bkeys, bank = [], [], None
        loop = model.loop
        self._xp_memo, self._xp_stable = {}, 0  # per-step xpass state
        self._pre, self._next = None, None  # INKLING_FAST=mega_pre hand-over
        new_k, new_v = list(st["k"]), list(st["v"])
        new_conv = [list(c) for c in st["conv"]]
        caps = {}
        want = set(captures)
        for visit, index in enumerate(self.visits):
            if loop is not None and index == loop.core_start:
                if visit == index:
                    prelude, prelude_blocks, prelude_keys = x, blocks.copy(), bkeys.copy()
                    # prelude blocks + the prelude partial repeat in every pass
                    self._xp_stable = len(bkeys) + 1
                else:
                    first = len(prelude_blocks) + 1
                    if self.reuse_keys:
                        xk = mx.fast.rms_norm(
                            x.astype(mx.float32)[..., None, :], None, 1e-6
                        )
                        bank = mx.concatenate([*bkeys[first:], xk], axis=-2)
                    else:
                        states = mx.stack([*blocks[first:], x], axis=-2)
                        bank = mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)
                    x, blocks, bkeys = prelude, prelude_blocks.copy(), prelude_keys.copy()
            if loop is not None and index == loop.core_end:
                self._xp_stable = 0
            messages = None
            if bank is not None and index < loop.core_end:
                if self.msgpack:
                    if index == loop.core_start:
                        pass_messages = fk.packed_messages(model.message_queries, bank)
                    messages = pass_messages[..., index - loop.core_start, :, :]
                else:
                    query = model.message_queries[index - loop.core_start]
                    scores = (query @ bank.swapaxes(-1, -2)) * x.shape[-1] ** -0.5
                    messages = mx.softmax(scores, axis=-1) @ bank
            if visit in want:
                caps[visit] = x
            if self._prefixes is not None:
                self._prefixes.append([None] * 4)
            self._next = None
            if self.mega_pre and visit + 1 < len(self.visits):
                ni = self.visits[visit + 1]
                reset = loop is not None and ni == loop.core_start and visit + 1 != ni
                if not reset:
                    nb = bank
                    self._next = (
                        model.layers[ni],
                        nb is not None and ni < loop.core_end if loop is not None else False,
                    )
            x, new_k[visit], new_v[visit] = self._layer(
                model.layers[index], x, blocks, bkeys, new_conv[visit],
                new_k[visit], new_v[visit], st, tau, messages,
            )
        h = self._attn_res(model.final_input, blocks, bkeys, x)[..., 0, :]
        if len(self.visits) in want:
            caps[len(self.visits)] = model.norm(h).astype(x.dtype)
        if (((("head" in self.sf and self._prefixes is not None and L > 1 and h.shape[0] * L <= 15)
              or ("head" in self.bf and L == 1 and 1 < h.shape[0] <= 32)))
                and fk._on_gpu()
                and type(self.lm.lm_head).__name__ == "Linear" and self.lm.sharding_group is None
                and self.lm.lm_head.weight.dtype == mx.bfloat16):
            # INKLING_SPECFAST=head: multi-row BF16-weight / FP32-accumulate head GEMV
            # (fast_kernels.lm_head_gemv, the head_approx kernel; ordering-only vs the
            # cached-FP32 GEMM; the BF16->FP32 widening is exact). Verify rows only.
            hn = model.norm(h)
            hn = hn.astype(model.embed_norm.weight.dtype).astype(mx.float32)
            c = self.lm.config
            from . import batch_kernels as bk

            hk = (bk.head_rows if ("head" in self.bf and L == 1
                                   and bk.head_supported(h.shape[0], h.shape[-1], self.lm.lm_head.weight))
                  else fk.lm_head_gemv)
            logits = hk(hn, self.lm.lm_head.weight, c.unpadded_vocab_size,
                        c.final_logit_softcapping or 0.0)
        else:
            logits = self.lm._logits_from_norm(model.norm(h))
        new = {
            "q_pos": q_pos,
            "swa_pos": st["swa_pos_new"],
            "k": new_k,
            "v": new_v,
            "conv": new_conv,
        }
        return logits, new, [caps[i] for i in captures]

    def _step_flat(self, tokens, *flat):
        self._prefixes = None
        logits, new, _ = self._forward(tokens, self._unflatten(list(flat)))
        logits = logits[:, -1]
        new["q_pos"] = new["q_pos"] + 1
        nxt = mx.argmax(logits, axis=-1).astype(mx.int32)
        return (logits, nxt, *self._flatten(new))

    def _spec_flat(self, captures, tokens, *flat):
        """Greedy verify of tokens = [anchor, d_1..d_n] plus on-device accept
        and commit.  Rollback is array-only: q_pos += accepted + 1, conv
        state = state after that prefix; rejected K/V slots stay in place and
        are masked (dist < 0) until overwritten."""
        self._prefixes = []
        try:
            logits, new, caps = self._forward(
                tokens, self._unflatten(list(flat)), captures
            )
            prefixes = self._prefixes
        finally:
            self._prefixes = None
        B, L = tokens.shape
        target = mx.argmax(logits, axis=-1).astype(mx.int32)  # [B, L]
        drafts = tokens[:, 1:]
        mism = mx.concatenate(
            [drafts != target[:, :-1], mx.ones((B, 1), mx.bool_)], axis=1
        )
        accepted = mx.argmax(mism.astype(mx.int32), axis=1).astype(mx.int32)  # [B]
        bonus = mx.take_along_axis(target, accepted[:, None], axis=1)[:, 0]
        pos = mx.arange(L, dtype=mx.int32)[None]
        pad = mx.concatenate([drafts, mx.zeros((B, 1), mx.int32)], axis=1)
        emitted = mx.where(
            pos < accepted[:, None], pad, mx.where(pos == accepted[:, None], bonus[:, None], -1)
        )
        new["q_pos"] = new["q_pos"] + accepted + 1
        sel = accepted[:, None, None, None]
        gidx = {}

        def commit(pv):
            if isinstance(pv, tuple):  # INKLING_SPECFAST=prefix
                _, ctx, w = pv
                gi = gidx.get(w)
                if gi is None:
                    gi = gidx[w] = (accepted[:, None] + 1
                                    + mx.arange(w, dtype=mx.int32)[None])[:, :, None]
                return mx.take_along_axis(ctx, mx.broadcast_to(gi, (B, w, ctx.shape[-1])), axis=1)
            return mx.take_along_axis(pv, mx.broadcast_to(sel, (B, 1, *pv.shape[2:])), axis=1)[:, 0]

        new["conv"] = [[commit(pv) for pv in prefixes[visit]] for visit in range(len(self.visits))]
        return (logits, target, accepted, emitted, bonus, *caps, *self._flatten(new))

    # ------------------------------------------------------------- public
    def step(self, tokens, compiled=True):
        """tokens: [B, 1] int.  Returns (logits [B, V] fp32, argmax [B] int32).

        Past the largest bucket the state is converted back into a stock
        cache (``to_cache``) and decoding continues on the normal path (INKLING_FAST kernels stay on)."""
        tokens = tokens.astype(mx.int32)
        if self.fallback is None:
            self._fit_bounded(1)  # review F1: a cap grown by verify past p+1 changes N
        if self.fallback is None and self.pos + 1 > self.cap:
            try:
                self._grow(self.pos + 1)
            except Unsupported:
                _loud(f"FastDecoder: position {self.pos + 1} past the largest bucket; continuing on the stock path")
                self.fallback = self.to_cache()
                self.state = None
        if self.fallback is not None:
            logits = self.lm(inputs=tokens, cache=self.fallback).logits[:, -1]
            self._known = [k + 1 for k in self._known]
            return logits, mx.argmax(logits, axis=-1).astype(mx.int32)
        self._ensure_head_cache(tokens.shape[0])
        fn = self._compiled if compiled else self._step_flat
        out = fn(tokens, *self._flatten(self.state))
        self._kvip_pending = self._kvip_pending or self.mega_kvip
        logits, nxt = out[0], out[1]
        self.state = self._unflatten(list(out[2:]))
        self._known = [k + 1 for k in self._known]
        return logits, nxt

    def spec_step(self, tokens, captures=(), compiled=True):
        """One greedy speculative round: verify tokens [B, L] = [anchor,
        drafts...] and commit the accepted prefix, all on device.

        Returns dict(logits [B,L,V], target [B,L], accepted [B], emitted [B,L]
        (accepted drafts then the bonus, -1 padded), bonus [B] (next anchor),
        captures list of [B,L,D]).  Nothing here syncs the host.  The host
        position estimate becomes an upper bound (+L per round); refresh it
        with ``observe(accepted)`` when accepted counts reach the host.
        Requires ring_extra >= L - 1.
        """
        tokens = tokens.astype(mx.int32)
        B, L = tokens.shape
        if L - 1 > self.ring - self.window:
            raise ValueError(f"verify L={L} needs ring_extra >= {L - 1}")
        if self.fallback is not None:
            raise Unsupported("static state handed off to stock")
        self._settle_kvip()  # verify reads/copies rows a kvip step wrote in place
        self._fit_bounded(L)  # review F1 / D5: bucket from the true position when it is determined
        if self.pos + L > self.cap:
            self._grow(self.pos + L)  # Unsupported past the largest bucket
        self._ensure_head_cache(B * L)
        captures = tuple(captures)
        key = captures
        fn = self._spec_compiled.get(key)
        if fn is None:
            fn = self._spec_compiled[key] = mx.compile(
                lambda *a: self._spec_flat(captures, *a)
            )
        if not compiled:
            fn = lambda *a: self._spec_flat(captures, *a)
        out = fn(tokens, *self._flatten(self.state))
        nc = len(captures)
        logits, target, accepted, emitted, bonus = out[:5]
        caps = list(out[5 : 5 + nc])
        self.state = self._unflatten(list(out[5 + nc :]))
        self._rounds.append(L)
        return dict(logits=logits, target=target, accepted=accepted,
                    emitted=emitted, bonus=bonus, captures=caps)

    @property
    def pos(self):
        """Host upper bound on max(q_pos): exact rows + L per unobserved round."""
        if self._known is None:
            return None
        return max(self._known) + sum(self._rounds)

    @pos.setter
    def pos(self, value):
        self._known = None if value is None else [int(value)] * max(1, len(self._known or [0]))
        self._rounds = []

    def observe(self, accepted):
        """The host learned one round's accepted counts (oldest unobserved
        round first): tighten the position bound used for bucket growth."""
        acc = accepted.tolist() if hasattr(accepted, "tolist") else list(accepted)
        acc = [int(x) for x in (acc if isinstance(acc, list) else [acc])]
        self._rounds.pop(0)
        self._known = [k + a + 1 for k, a in zip(self._known, acc)]

    def to_cache(self):
        """Static state -> stock mlx-vlm prompt cache (inverse of from_cache).

        Equal row lengths give RotatingKVCache / KVCache (as make_cache);
        unequal lengths give left-padded BatchRotatingKVCache / BatchKVCache.
        Also the hand-off point for anything the static path does not cover."""
        self._settle_kvip()
        import numpy as np

        from ..cache import ArraysCache, CacheList

        st, W, R = self.state, self.window, self.ring
        lengths = [int(x) for x in np.array(st["q_pos"]).tolist()]
        B, Tmax = len(lengths), max(lengths)
        same = len(set(lengths)) == 1
        out = []
        for visit in range(len(self.visits)):
            kb, vb = st["k"][visit], st["v"][visit]
            if self.sliding[visit]:
                size = min(Tmax, W)
                rows_k, rows_v, pads = [], [], []
                for b, n in enumerate(lengths):
                    m = min(n, W)
                    slots = [p % R for p in range(n - m, n)]
                    pad = size - m
                    idx = mx.array(slots, mx.int32)
                    rk, rv = mx.take(kb[b], idx, axis=1), mx.take(vb[b], idx, axis=1)
                    if pad:
                        z = mx.zeros((rk.shape[0], pad, rk.shape[2]), rk.dtype)
                        rk, rv = mx.concatenate([z, rk], 1), mx.concatenate([z, rv], 1)
                    rows_k.append(rk); rows_v.append(rv); pads.append(pad)
                k, v = mx.stack(rows_k), mx.stack(rows_v)
                if same:
                    kv = RotatingKVCache(max_size=W, keep=0)
                    kv.keys, kv.values = k, v
                    kv.offset, kv._idx = Tmax, size
                else:
                    kv = BatchRotatingKVCache(W, pads)
                    kv.keys, kv.values = k, v
                    kv.offset = mx.array(lengths)
                    kv._offset, kv._idx, kv.rotated = size, size, False
            else:
                rows_k, rows_v, pads = [], [], []
                for b, n in enumerate(lengths):
                    pad = Tmax - n
                    rk, rv = kb[b, :, :n], vb[b, :, :n]
                    if pad:
                        z = mx.zeros((rk.shape[0], pad, rk.shape[2]), rk.dtype)
                        rk, rv = mx.concatenate([z, rk], 1), mx.concatenate([z, rv], 1)
                    rows_k.append(rk); rows_v.append(rv); pads.append(pad)
                k, v = mx.stack(rows_k), mx.stack(rows_v)
                if same:
                    kv = KVCache()
                    kv.keys, kv.values, kv.offset = k, v, Tmax
                else:
                    kv = BatchKVCache(pads)
                    kv.keys, kv.values = k, v
                    kv.offset, kv._idx = mx.array(lengths), Tmax
            conv = ArraysCache(4)  # decode sees no padding: all rows valid
            conv.cache = list(st["conv"][visit])
            out.append(CacheList(kv, conv))
        return out


# ---------------------------------------------------------------- harness glue
def enabled():
    return os.environ.get("INKLING_FAST_DECODE", "0") == "1"


def generate_fast(model, processor, prompt, max_tokens=256, prefill_step=2048,
                  logits_hook=None, eos_token_ids=None, decoder_kwargs=None):
    """Greedy: normal-path prefill, convert cache, loop the compiled step with
    async_eval pipelining.  ``logits_hook(step, logits)`` sees each step's
    logits (forces a sync per step when given).  Returns list of token ids."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if isinstance(prompt, str):
        ids = tok.encode(prompt, add_special_tokens=False)
    else:
        ids = list(prompt)
    lm = model.language_model if hasattr(model, "language_model") else model
    inputs = mx.array(ids)[None]
    cache = lm.make_cache()
    T = inputs.shape[1]
    for s in range(0, T, prefill_step):
        e = min(s + prefill_step, T)
        if e == T:
            logits = lm(inputs=inputs[:, s:e], cache=cache).logits[:, -1]
            mx.eval(logits)
        else:
            out = lm(inputs=inputs[:, s:e], cache=cache, return_hidden=True,
                     skip_logits=True)
            mx.eval(out.hidden_states[0])
            mx.clear_cache()
    try:
        dec = FastDecoder(lm, **(decoder_kwargs or {})).from_cache(cache)
        del cache
        step = dec.step
    except Unsupported:  # sharded model or prompt beyond the largest bucket

        def step(t):
            lg = lm(inputs=t, cache=cache).logits[:, -1]
            return lg, mx.argmax(lg, axis=-1).astype(mx.int32)

    y = mx.argmax(logits, axis=-1).astype(mx.int32)
    if logits_hook is not None:
        logits_hook(0, logits)
    eos = set(eos_token_ids or [])
    toks = []
    nxt_logits, nxt = step(y[:, None])
    mx.async_eval(nxt)
    for i in range(max_tokens):
        cur = int(y.item())
        toks.append(cur)
        if cur in eos or i == max_tokens - 1:
            break
        y, cur_logits = nxt, nxt_logits
        nxt_logits, nxt = step(y[:, None])
        mx.async_eval(nxt)
        if logits_hook is not None:
            logits_hook(i + 1, cur_logits)
    return toks


def verify_enabled():
    return os.environ.get("INKLING_FAST_VERIFY", "0") == "1"


def speculative_loop(dec, anchor, draft_fn, n, max_tokens, captures=(),
                     eos_token_ids=None, on_round=None):
    """Greedy speculative decoding on a FastDecoder with no blocking host sync
    inside a round.

    anchor: [B] int32 device array (the last emitted, not yet fed, token).
    draft_fn(round_state) -> drafts [B, n] int32 (device).  ``round_state``
    holds device arrays only: anchor, q_pos (anchor's position), captures and
    accepted of the previous round (None in round 0).  n may be an int or a
    callable(B) (draft length per batch size; n == 0 means plain decode).

    Round k+1 is built and queued before round k's emitted tokens are copied
    to the host (one round of lag), so graph building overlaps GPU work.
    Returns per-row token lists (anchor excluded) truncated to max_tokens.
    """
    B = anchor.shape[0]
    n = n(B) if callable(n) else int(n)
    eos = set(eos_token_ids or [])
    out = [[] for _ in range(B)]
    done = [False] * B
    rs = dict(anchor=anchor, q_pos=dec.state["q_pos"], captures=None, accepted=None)
    pending = None
    rounds = 0

    def drain(res):
        acc, em = res["accepted"].tolist(), res["emitted"].tolist()
        if n:
            dec.observe(acc)
        for b in range(B):
            if done[b]:
                continue
            for t in em[b][: acc[b] + 1]:
                out[b].append(t)
                if t in eos or len(out[b]) >= max_tokens:
                    done[b] = True
                    break

    while True:
        if n == 0:
            logits, nxt = dec.step(rs["anchor"][:, None])
            res = dict(accepted=mx.zeros((B,), mx.int32), emitted=nxt[:, None],
                       bonus=nxt, captures=[], logits=logits[:, None])
        else:
            drafts = draft_fn(rs).astype(mx.int32)
            tokens = mx.concatenate([rs["anchor"][:, None], drafts], axis=1)
            res = dec.spec_step(tokens, captures)
        mx.async_eval(res["emitted"], res["accepted"], res["bonus"])
        rounds += 1
        if on_round is not None:
            on_round(rounds, res)
        rs = dict(anchor=res["bonus"], q_pos=dec.state["q_pos"],
                  captures=res["captures"], accepted=res["accepted"])
        if pending is not None:
            drain(pending)  # previous round: already finished or nearly so
        pending = res
        if all(done):
            break
    # Review D5: the last queued round is committed in the decoder state but was
    # never observed; observe it so the host position is exact again (later
    # step()/spec_step() then fit the bucket from the true position). The state
    # then includes that round's tokens, which may run past EOS / max_tokens
    # (they are not in the returned lists).
    if n and pending is not None and dec._rounds:
        dec.observe(pending["accepted"].tolist())
    return [o[:max_tokens] for o in out], rounds


def draft_len_for_batch(B, table=None):
    """Draft length n per batch size from INKLING_FAST_VERIFY_N, e.g.
    "1:7,4:3,6:0" = n=7 for B<4, 3 for 4<=B<6, plain decode for B>=6.
    Default keeps n=7 everywhere until spec_bench measures the trade-off."""
    spec = table or os.environ.get("INKLING_FAST_VERIFY_N", "1:7")
    pairs = sorted(
        (int(b), int(n)) for b, n in (item.split(":") for item in spec.split(",") if item)
    )
    n = pairs[0][1]
    for b, v in pairs:
        if B >= b:
            n = v
    return n
