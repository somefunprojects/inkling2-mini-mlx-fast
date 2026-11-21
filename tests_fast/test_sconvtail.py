"""sconv_tail kernel vs stock ops at real dims (D=2304, top-8), bitwise."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: F401
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2.language import _combine_routes
from mlx_vlm.models.inkling2.kernels import short_convolution

D, K = 2304, 8
bad = tot = 0
for seed in range(30):
    mx.random.seed(seed)
    for B in (1, 2, 4):
        mod = types.SimpleNamespace(kernel_size=4, conv=types.SimpleNamespace(
            weight=(mx.random.normal((D, 4, 1)) * 0.5).astype(mx.bfloat16)))
        state = mx.random.normal((B, 3, D))
        e = (mx.random.normal((B, 1, K, D)) * 3).astype(mx.bfloat16)
        w = mx.softmax(mx.random.normal((B, 1, K)), -1) * 2.5
        res = mx.random.normal((B, 1, D)) * 4
        a = (mx.random.normal((B, 1, D)) * 2).astype(mx.bfloat16)
        x = (mx.random.normal((B, 1, D)) * 2).astype(mx.bfloat16)
        mx.eval(state, e, w, res, a, x, mod.conv.weight)
        # MLP side
        h = _combine_routes(e, w)
        s = short_convolution(state, mod.conv.weight, h, None, 4)
        ref = (res + s.astype(mx.float32)).astype(mx.bfloat16)
        rst = mx.concatenate([state, h.astype(mx.float32)], 1)[:, -3:]
        got, gst = fk.sconv_tail(mod, state, e, res, w)
        # attn side
        s2 = short_convolution(state, mod.conv.weight, a, None, 4).astype(mx.float32)
        ref2, ref3 = s2 + x.astype(mx.float32), s2
        got2, _ = fk.sconv_tail(mod, state, a, x)
        got3, _ = fk.sconv_tail(mod, state, a, None)
        for r_, g_ in ((ref, got), (rst, gst), (ref2, got2), (ref3, got3)):
            tot += 1
            if not mx.array_equal(r_, g_).item():
                bad += 1
                print("diff seed", seed, "B", B, mx.abs(r_.astype(mx.float32) - g_.astype(mx.float32)).max().item())
print(f"sconv_tail vs stock: {tot - bad}/{tot} bitwise")
