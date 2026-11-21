"""qkvclip kernel vs the stock op sequence at real Inkling 2 Mini dims (bitwise).
Random activations with a spread of scales so the clip both triggers and not."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: F401  (package stubs)
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2.language import l2_clip

cfg = types.SimpleNamespace(d_rel=16, qk_l2_clip_threshold=float(os.environ.get("THR", "12.0")),
                            rel_l2_clip_threshold=float(os.environ.get("RTHR", "4.0")),
                            rms_norm_eps=1e-6, use_value_norm_no_gamma=True)
dt = getattr(mx, os.environ.get("DT", "bfloat16"))


def stock(q, k, v, r, attn, tau):
    b, L = q.shape[:2]
    qq = l2_clip(q.reshape(b, L, attn.n_heads, 128), cfg.qk_l2_clip_threshold, attn.q_scale)
    kk = l2_clip(k.reshape(b, L, attn.n_kv, 128), cfg.qk_l2_clip_threshold)
    vv = mx.fast.rms_norm(v.reshape(b, L, attn.n_kv, 128).astype(mx.float32), None, 1e-6).astype(q.dtype)
    rr = l2_clip(r.reshape(b, L, attn.n_heads, 16), cfg.rel_l2_clip_threshold, attn.rel_scale)
    if tau is not None:
        qq = (qq.astype(mx.float32) * tau).astype(q.dtype)
        rr = (rr.astype(mx.float32) * tau).astype(q.dtype)
    return qq, kk, vv, rr


bad = tot = 0
for seed in range(int(os.environ.get("SEEDS", "40"))):
    mx.random.seed(seed)
    for B, L in ((1, 1), (4, 1), (1, 3), (8, 1)):
        attn = types.SimpleNamespace(n_heads=24, n_kv=8, head_dim=128,
                                     q_scale=(1 + 0.2 * mx.random.normal((24, 128))).astype(dt),
                                     rel_scale=(1 + 0.2 * mx.random.normal((24, 16))).astype(dt))
        s = 10 ** mx.random.uniform(-1.5, 1.5, (1,))
        fused = (mx.random.normal((B, L, 3072 + 384 + 3072)) * s).astype(dt)
        q, r, _ = mx.split(fused, [3072, 3456], axis=-1)       # views, like fuse
        kv = (mx.random.normal((B, L, 2048)) * s).astype(dt)
        k, v = mx.split(kv, [1024], axis=-1)
        for tau in (None, (1 + 0.05 * mx.random.uniform(shape=(B, L, 1, 1))).astype(mx.float32)):
            mx.eval(q, k, v, r, attn.q_scale, attn.rel_scale)
            ref = stock(q, k, v, r, attn, tau)
            tl = None if tau is None else tau.reshape(B * L)
            got = fk.qkv_clip(q, k, v, r, attn, cfg, tl)
            for name, a, g in zip("qkvr", ref, got):
                tot += 1
                if not mx.array_equal(a, g).item():
                    bad += 1
                    d = mx.abs(a.astype(mx.float32) - g.astype(mx.float32))
                    if bad < 12:
                        print(f"seed {seed} B{B} L{L} tau={tau is not None} {name}: "
                              f"{int((d > 0).sum().item())}/{d.size} differ, max {d.max().item():.3e}")
print(f"qkvclip vs stock: {tot - bad}/{tot} tensors bitwise ({dt})")
