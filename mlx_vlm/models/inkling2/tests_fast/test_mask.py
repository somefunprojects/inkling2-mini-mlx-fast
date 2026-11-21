"""Physical-order rel-bias mask + SDPA sinks vs the stock formulation."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _load import load
import mlx.core as mx

fk = load("mlx_vlm.models.inkling2.fast_kernels")
ink = load("mlx_vlm.models.inkling.language")
cache_mod = load("mlx_vlm.models.cache")
banded_additive_mask = ink.banded_additive_mask
RotatingKVCache, KVCache = cache_mod.RotatingKVCache, cache_mod.KVCache

H, KV, HD, DREL, WIN = 24, 8, 128, 16, 1024
dt = mx.bfloat16
mx.random.seed(1)


def stock_attn(q, k, v, r, proj, sinks, sliding, extent, kv):
    """Mirror of Attention.__call__ after update_and_fetch (L=1)."""
    if isinstance(kv, RotatingKVCache):
        k, v = kv._temporal_order(k), kv._temporal_order(v)
    b, length = q.shape[0], q.shape[2]
    size = k.shape[2]
    mask = banded_additive_mask(r, proj, size - length, size, sliding, extent, shape_ref=k)
    mask = mx.concatenate([mask, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
    k = mx.concatenate([k, mx.zeros((b, KV, 1, HD), k.dtype)], axis=2)
    v = mx.concatenate([v, mx.zeros((b, KV, 1, HD), v.dtype)], axis=2)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=HD**-0.5, mask=mask), mask


def fast_attn(q, k, v, r, proj, sinks, sliding, kv):
    b = q.shape[0]
    idx = kv._idx if isinstance(kv, RotatingKVCache) else k.shape[2]
    idx = mx.array([idx], mx.int32)
    if b == 1:
        mask = fk.ring_mask(r, proj, idx, k, sliding)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=HD**-0.5, mask=mask, sinks=sinks.reshape(-1))
    else:
        mask = fk.ring_mask(r, proj, idx, k, sliding, sinks=sinks)
        z = mx.zeros((b, KV, 1, HD), k.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, mx.concatenate([k, z], 2), mx.concatenate([v, z], 2), scale=HD**-0.5, mask=mask)
    return out, mask


def to_temporal(mask_phys, kv, S):
    """Reorder fast mask columns (physical) into temporal order for comparison."""
    if isinstance(kv, RotatingKVCache) and kv._idx < kv.offset and kv._idx != S:
        cols = list(range(kv._idx, S)) + list(range(0, kv._idx))
    else:
        cols = list(range(S))
    return mask_phys[..., mx.array(cols)]


def run(kind, B, prefill, steps):
    sliding = WIN if kind == "sliding" else 0
    extent = WIN
    kv = RotatingKVCache(max_size=WIN, keep=0) if kind == "sliding" else KVCache()
    proj = (mx.random.normal((DREL, extent)) * 0.5).astype(dt)
    k0 = mx.random.normal((B, KV, prefill, HD)).astype(dt)
    kv.update_and_fetch(k0, k0 * 0.5)
    mask_bitwise, out_bitwise, out_maxdiff, mask_eq_steps = True, True, 0.0, 0
    for s in range(steps):
        q = mx.random.normal((B, H, 1, HD)).astype(dt)
        kn = mx.random.normal((B, KV, 1, HD)).astype(dt)
        vn = mx.random.normal((B, KV, 1, HD)).astype(dt)
        r = mx.random.normal((B, 1, H, DREL)).astype(dt)
        sinks = (mx.random.normal((B, 1, H)) * 2).astype(dt)
        k, v = kv.update_and_fetch(kn, vn)
        ref, mref = stock_attn(q, k, v, r, proj, sinks, sliding, extent, kv)
        got, mgot = fast_attn(q, k, v, r, proj, sinks, sliding, kv)
        S = k.shape[2]
        mt = to_temporal(mgot[..., :S], kv, S)
        meq = mx.array_equal(mt, mref[..., :S]).item()
        if B > 1:
            meq = meq and mx.array_equal(mgot[..., S], mref[..., S]).item()
        mask_bitwise &= meq
        # max |diff| relative to max |ref| (bf16 ulp at the top is 2^-8..2^-7)
        d = (mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()
             / mx.abs(ref.astype(mx.float32)).max()).item()
        out_maxdiff = max(out_maxdiff, d)
        out_bitwise &= d == 0
    print(f"{kind:7s} B={B} prefill={prefill:4d} steps={steps:3d} final offset={kv.offset:5d} "
          f"mask_bitwise={mask_bitwise} attn_bitwise={out_bitwise} attn_maxdiff/max|ref|={out_maxdiff:.2e}")
    return mask_bitwise, out_maxdiff


def test_slot_mask_general():
    # slot_mask with explicit slot positions incl. empty slots, window cap, R<S.
    B, S, R = 2, 300, 200
    proj = mx.random.normal((DREL, R)).astype(dt)
    r = mx.random.normal((B, 1, H, DREL)).astype(dt)
    q_pos = mx.array([250, 120], mx.int32)
    pos = mx.array([[(j * 7) % 260 - 5 for j in range(S)], [j if j < 150 else -1 for j in range(S)]], mx.int32)
    for window in (0, 100):
        got = fk.slot_mask(r, proj, pos, q_pos, window)
        # reference in numpy-ish mlx
        dist = q_pos[:, None] - pos
        rl = (r.astype(mx.float32)[:, 0] [..., None] * proj.astype(mx.float32)[None, None]).sum(-2)  # [B,H,R]
        gathered = mx.take_along_axis(rl, mx.broadcast_to(mx.clip(dist, 0, R - 1)[:, None, :], (B, H, S)), -1).astype(dt)
        val = mx.where((dist >= R)[:, None], mx.array(0, dt), gathered)
        bad = (pos < 0) | (dist < 0)
        if window:
            bad = bad | (dist >= window)
        ref = mx.where(bad[:, None], mx.array(-1e30, dt), val)[:, :, None]
        d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        d = mx.where(mx.abs(ref.astype(mx.float32)) > 1e29, 0, d).max().item()
        eqmask = mx.array_equal(got <= -1e29, ref <= -1e29).item()
        print(f"slot_mask general window={window}: masked-pattern equal={eqmask} max|d|={d:.3g} (ref sums d_rel in a different order)")
        assert eqmask and d <= 2 ** -6


def test_compile():
    # Inside mx.compile, fixed shapes, varying idx: one trace, correct results.
    S = WIN
    proj = mx.random.normal((DREL, S)).astype(dt)
    k = mx.zeros((1, KV, S, HD), dt)
    r = mx.random.normal((1, 1, H, DREL)).astype(dt)
    traces = [0]

    def f(r, idx, k):
        traces[0] += 1
        return fk.ring_mask(r, proj, idx, k, WIN)

    cf = mx.compile(f)
    ok = True
    for i in (1, 17, 513, 1024):
        idx = mx.array([i], mx.int32)
        ok &= mx.array_equal(cf(r, idx, k), f(r, idx, k)).item()
    print(f"ring_mask under mx.compile: equal={ok} traces={traces[0] - 4}")
    assert ok and traces[0] - 4 == 1


if __name__ == "__main__":
    worst = 0.0
    all_mask = True
    for args in [("sliding", 1, 1500, 40), ("sliding", 1, 600, 500), ("sliding", 1, 1, 30),
                 ("sliding", 2, 1100, 30), ("global", 1, 37, 20), ("global", 1, 4000, 10),
                 ("global", 2, 300, 10)]:
        m, d = run(*args)
        all_mask &= m
        worst = max(worst, d)
    test_slot_mask_general()
    test_compile()
    assert all_mask, "mask not bitwise"
    assert worst <= 2 ** -7, worst
    print(f"PASS mask (all masks bitwise; worst attention max|d|/max|ref| = {worst:.2e}, i.e. <= 1 bf16 ulp at the top of the range)")
