"""Dedup verify MoE: grouped_masked_qmv (all routes of one expert share weight reads)
vs the decode path (gather_qmm per route, L=1). Bitwise check + timing."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import mlx.core as mx, mlx.nn as nn
from mlx_vlm.models.switch_layers import SwitchLinear
from mlx_vlm.models import masked_quantized as mq
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear

def grouped(x, lin, idx, V):
    k, n = lin.input_dims, lin.output_dims
    return mq._grouped_qmv()(
        inputs=[x, lin.weight, lin.scales, mq._global_scale(lin), idx],
        template=[("T", x.dtype), ("K", k), ("N", n), ("V", V)],
        grid=(32, n // 8 * 2, idx.size), threadgroup=(32, 2, 1),
        output_shapes=[(*idx.shape, n)], output_dtypes=[x.dtype])[0]

def make(E, K, N):
    sl = SwitchLinear(K, N, E); sl.weight = (mx.random.normal(sl.weight.shape) * 0.05).astype(mx.bfloat16)
    q = sl.to_quantized(group_size=16, bits=4, mode="nvfp4")
    return ScaledQuantizedSwitchLinear(q, (0.5 + mx.random.uniform(shape=(E,))) / 2688 * 8, "weight_scale_2")

mx.random.seed(0)
for (E, K, N) in ((128, 2304, 1152), (8, 256, 128), (8, 512, 128)):
    lin = make(E, K, N)
    for L in (2, 4, 8):
        x = mx.random.normal((1, L, K)).astype(mx.bfloat16)
        # skewed routing like real models: 8 routes per token from a hot subset
        idx = mx.stack([mx.random.permutation(min(E, 24))[:8] for _ in range(L)])[None].astype(mx.uint32) if E >= 24 else \
              mx.stack([mx.random.permutation(E)[:min(8, E)] for _ in range(L)])[None].astype(mx.uint32)
        R = idx.shape[-1]
        # decode reference: per position gather_qmm (as SwitchGLU.__call__ at L=1)
        ref = mx.concatenate([lin(mx.expand_dims(x[:, i:i+1], (-2, -3)), idx[:, i:i+1]).squeeze(-2) for i in range(L)], axis=1)
        flat = idx.reshape(-1); order = mx.argsort(flat); inv = mx.argsort(order)
        xs = x.reshape(-1, K)[order // R]
        for V in (2, 8):
            out = grouped(xs, lin, flat[order].astype(mx.int32), V)[inv].reshape(1, L, R, N)
            mx.eval(out, ref)
            def t(f, n=50):
                f(); mx.synchronize(); t0 = time.perf_counter()
                for _ in range(n): mx.eval(f())
                return (time.perf_counter() - t0) / n * 1e3
            tg = t(lambda: grouped(xs, lin, flat[order].astype(mx.int32), V))
            tr = t(lambda: mx.concatenate([lin(mx.expand_dims(x[:, i:i+1], (-2, -3)), idx[:, i:i+1]).squeeze(-2) for i in range(L)], axis=1))
            print(f"E={E} K={K} N={N} L={L} V={V} unique={len(set(flat.tolist()))}/{flat.size} bitwise={bool(mx.array_equal(out, ref).item())} "
                  f"max|d|={mx.abs(out.astype(mx.float32)-ref.astype(mx.float32)).max().item():.1e} grouped {tg:.3f} ms vs per-pos {tr:.3f} ms")
