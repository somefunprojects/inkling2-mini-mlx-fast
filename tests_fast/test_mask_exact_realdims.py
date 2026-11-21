"""INKLING_FAST=mask exact path at real Inkling 2 Mini attention dims (no weights).

For KVCache and RotatingKVCache(1024): prefill P tokens, then decode steps; at each
step compare the stock operands/output (temporal order + zero row, banded mask +
sink column) with _exact_sink_kv + ring_mask. Must be bitwise. Also counts the
three exact sub-cases (view / wrapped copy / concat fallback) to show coverage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: F401  (sets up the package path without importing mlx_vlm.__init__)
import mlx.core as mx
from mlx_vlm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.inkling.language import banded_additive_mask
from mlx_vlm.models.inkling2 import language as L, fast_kernels as F

H, KV, D, DREL, W, REL = 24, 8, 128, 16, 1024, 1024
mx.random.seed(0)
proj = (mx.random.normal((DREL, REL)) * 0.3).astype(mx.bfloat16)
cases = {"view": 0, "wrapped": 0, "concat": 0}
OLD = {"sink": 0, "sink_elems": 0, "n": 0, "ring": 0, "ring_elems": 0, "nw": 0}
orig = L._exact_sink_kv


def counting(kv, k, v):
    size = k.shape[2]
    if type(kv) is RotatingKVCache and kv._idx != size:
        cases["wrapped"] += 1
    elif kv.keys.shape[2] > size:
        cases["view"] += 1
    else:
        cases["concat"] += 1
    return orig(kv, k, v)


def run(kind, prompt, steps, check_every=1):
    window = W if kind == "rot" else 0
    mk = (lambda: RotatingKVCache(max_size=W, keep=0)) if kind == "rot" else KVCache
    a, b = mk(), mk()
    kp = mx.random.normal((1, KV, prompt, D)).astype(mx.bfloat16)
    vp = mx.random.normal((1, KV, prompt, D)).astype(mx.bfloat16)
    for c in (a, b):
        c.update_and_fetch(kp, vp)
    bad = 0
    for t in range(steps):
        k1 = mx.random.normal((1, KV, 1, D)).astype(mx.bfloat16)
        v1 = mx.random.normal((1, KV, 1, D)).astype(mx.bfloat16)
        q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
        r = (mx.random.normal((1, 1, H, DREL)) * 0.5).astype(mx.bfloat16)
        sinks = (mx.random.normal((1, 1, H)) * 2).astype(mx.bfloat16)
        # stock
        ks, vs = a.update_and_fetch(k1, v1)
        if isinstance(a, RotatingKVCache):
            ks, vs = a._temporal_order(ks), a._temporal_order(vs)
        S = ks.shape[2]
        m = banded_additive_mask(r, proj, S - 1, S, window, window or REL, shape_ref=ks)
        m = mx.concatenate([m, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
        z = mx.zeros((1, KV, 1, D), mx.bfloat16)
        os_ = mx.fast.scaled_dot_product_attention(
            q, mx.concatenate([ks, z], 2), mx.concatenate([vs, z], 2), scale=D**-0.5, mask=m)
        # exact
        kf, vf = b.update_and_fetch(k1, v1)
        mf = F.ring_mask(r, proj, mx.array([kf.shape[2]], mx.int32), kf, window, sinks=sinks)
        kf, vf = counting(b, kf, vf)
        of = mx.fast.scaled_dot_product_attention(q, kf, vf, scale=D**-0.5, mask=mf)
        if t % check_every == 0:
            ok = mx.array_equal(m, mf).item() and mx.array_equal(os_, of).item()
            bad += not ok
            # reference only: the old non-bitwise variants on the same operands
            ns = int(mx.sum(mx.fast.scaled_dot_product_attention(
                q, kf[..., :-1, :], vf[..., :-1, :], scale=D**-0.5, mask=mf[..., :-1],
                sinks=sinks.reshape(-1)) != os_).item())
            OLD["sink"] += ns > 0; OLD["sink_elems"] += ns; OLD["n"] += 1
            if isinstance(b, RotatingKVCache) and b.offset >= W:
                raw_k, raw_v = b.keys, b.values  # physical order (zero slot not present: full ring)
                mr = F.ring_mask(r, proj, mx.array([b._idx], mx.int32), raw_k, window, sinks=sinks)
                zz = mx.zeros((1, KV, 1, D), mx.bfloat16)
                nr = int(mx.sum(mx.fast.scaled_dot_product_attention(
                    q, mx.concatenate([raw_k, zz], 2), mx.concatenate([raw_v, zz], 2),
                    scale=D**-0.5, mask=mr) != os_).item())
                OLD["ring"] += nr > 0; OLD["ring_elems"] += nr; OLD["nw"] += 1
    print(f"{kind:3s} prompt={prompt:5d} steps={steps:4d} mismatching steps={bad}")
    return bad


total = 0
for kind in ("rot", "kv"):
    for prompt, steps in ((3, 300), (250, 20), (1000, 60), (1500, 40), (4000, 20)):
        total += run(kind, prompt, steps)
print("cases", cases)
print(f"reference (old opt-in variants, not gated): sinks= differs on {OLD['sink']}/{OLD['n']} steps "
      f"({OLD['sink_elems']} of {OLD['n'] * H * D} output elems); physical ring differs on "
      f"{OLD['ring']}/{OLD['nw']} wrapped steps ({OLD['ring_elems']} elems)")
print("PASS" if total == 0 else "FAIL")
