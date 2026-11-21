"""LM head GEMV vs FP32 reference. Run: python -m mlx_vlm.models.inkling2.tests_fast.test_head [--bench]"""
import sys, time
import mlx.core as mx
import os, sys; sys.path.insert(0, os.path.dirname(__file__)); from _load import load
lm_head_gemv = load("mlx_vlm.models.inkling2.fast_kernels").lm_head_gemv

V, VOUT, D, CAP = 201024, 200064, 2304, 30.0


def stock(h, wf32):
    logits = h.astype(mx.float32) @ wf32.T
    logits = mx.tanh(logits / CAP) * CAP
    return logits[..., :VOUT]


def main(bench):
    mx.random.seed(0)
    w = (mx.random.normal((V, D)) * 0.02).astype(mx.bfloat16)
    wf = w.astype(mx.float32)
    mx.eval(w, wf)
    worst = 0.0
    for M in (1, 2, 3, 4, 8, 9, 15):
        h = mx.random.normal((1, M, D)) * 3
        h = h.astype(mx.bfloat16).astype(mx.float32)  # stock promotes via embed dtype
        ref_raw = h @ wf.T
        got_raw = lm_head_gemv(h, w, V, 0.0)
        ref = stock(h, wf)
        got = lm_head_gemv(h, w, VOUT, CAP)
        rel = (mx.abs(got_raw - ref_raw).max() / mx.abs(ref_raw).max()).item()
        cap_diff = mx.abs(got - ref).max().item()
        amax_ok = mx.array_equal(mx.argmax(got, -1), mx.argmax(ref, -1)).item()
        worst = max(worst, rel)
        print(f"M={M:2d} raw max|d|/max|ref|={rel:.2e}  softcapped max|d|={cap_diff:.2e}  argmax_equal={amax_ok} shape={got.shape}")
    assert worst < 1e-5, worst
    print("PASS head")
    if bench:
        for M in (1, 2, 4, 8, 9, 15):
            h = mx.random.normal((1, M, D))
            for name, fn in (("fp32 matmul+cap", lambda: stock(h, wf)), ("bf16 gemv kernel", lambda: lm_head_gemv(h, w, VOUT, CAP))):
                for _ in range(3): mx.eval(fn())
                n = 20; t = time.perf_counter()
                for _ in range(n): mx.eval(fn())
                dt = (time.perf_counter() - t) / n * 1e3
                print(f"M={M:2d} {name:18s} {dt:7.3f} ms")


if __name__ == "__main__":
    main("--bench" in sys.argv)
