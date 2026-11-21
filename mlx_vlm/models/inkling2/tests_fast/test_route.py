"""INKLING_FAST=route: fused sigmoid/bias/top-k/normalise vs stock _route."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from _load import load
import mlx.core as mx

fk = load("mlx_vlm.models.inkling2.fast_kernels")
lang = load("mlx_vlm.models.inkling2.language")
mx.random.seed(5)
ok_all = True
for D in (2304, 4608):
    for E in (128, 384):
        W = (mx.random.normal((E, D)) / D ** 0.5).astype(mx.bfloat16)
        Wf = W.astype(mx.float32)
        for R in (1, 4, 16):
            for case in ("random", "ties", "bias-ties"):
                x = mx.random.normal((R, 1, D)).astype(mx.bfloat16)
                bias = mx.random.normal((E,)) * 0.01
                if case == "ties":
                    # identical expert rows -> exactly equal scores
                    W2 = mx.concatenate([W[: E // 2], W[: E // 2]], 0)
                    Wf2, Wu = W2.astype(mx.float32), W2
                    bias = mx.zeros((E,))
                elif case == "bias-ties":
                    Wu, Wf2 = W, Wf
                    bias = mx.zeros((E,))
                    x = mx.zeros((R, 1, D), mx.bfloat16)  # all scores 0.5: full tie
                else:
                    Wu, Wf2 = W, Wf
                i0, w0 = lang._route(x, Wu, bias, 8, 8.0)
                i1, w1 = fk.route(x, Wf2, bias, 8, 8.0)
                ie = mx.array_equal(i0, i1).item()
                we = mx.array_equal(w0, w1).item()
                d = mx.abs(w0 - w1).max().item()
                ok_all &= ie and we
                if not (ie and we) or case != "random" or R == 1:
                    print(f"D={D} E={E} R={R:2d} {case:9s} idx_equal={ie} w_bitwise={we} max|dw|={d:.2g} dtype={i1.dtype}")
print("ALL BITWISE" if ok_all else "MISMATCH")

# timing (indicative, M4 Pro)
D, E = 4608, 384
W = (mx.random.normal((E, D)) / D ** 0.5).astype(mx.bfloat16)
Wf = W.astype(mx.float32)
bias = mx.random.normal((E,)) * 0.01
for R in (1, 16):
    x = mx.random.normal((R, 1, D)).astype(mx.bfloat16)
    for name, fn in (("stock _route (bf16 W)", lambda: lang._route(x, W, bias, 8, 8.0)),
                     ("stock _route (f32 W, glue)", lambda: lang._route(x, Wf, bias, 8, 8.0)),
                     ("fused route", lambda: fk.route(x, Wf, bias, 8, 8.0))):
        for _ in range(5): mx.eval(fn())
        n = 200; t = time.perf_counter()
        for _ in range(n): mx.eval(fn())
        print(f"R={R:2d} {name:28s} {(time.perf_counter() - t) / n * 1e6:7.1f} us/call (incl. eval sync)")
    # chained (no sync per call): 48 calls like a decode step
    for name, fn in (("stock glue x48 chained", lambda xx: lang._route(xx, Wf, bias, 8, 8.0)),
                     ("fused x48 chained", lambda xx: fk.route(xx, Wf, bias, 8, 8.0))):
        def chain():
            xx = x; outs = []
            for _ in range(48):
                i, w = fn(xx); outs.append(w)
                xx = xx + w.sum().astype(xx.dtype) * 0  # dependency
            return outs
        for _ in range(3): mx.eval(chain())
        n = 20; t = time.perf_counter()
        for _ in range(n): mx.eval(chain())
        print(f"R={R:2d} {name:28s} {(time.perf_counter() - t) / n * 1e3:7.3f} ms per 48")
assert ok_all
