"""Bitwise check: fast_kernels.lm_head_gemv_exact vs stock M=1 FP32 head.

Real shapes (V=201024 padded rows, D=2304, unpadded vout, softcap 30), random
BF16 weights and BF16-rounded activations of several scales. Run on the GPU.
Usage: python test_head_exact.py [V] [D] [VOUT]
"""
import sys
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk

V = int(sys.argv[1]) if len(sys.argv) > 1 else 201088
D = int(sys.argv[2]) if len(sys.argv) > 2 else 2304
VOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 201024
CAP = 30.0
mx.random.seed(0)
w = (mx.random.normal((V, D)) * 0.02).astype(mx.bfloat16)
wf = w.astype(mx.float32)
mx.eval(w, wf)


def stock(h):
    h = h.astype(mx.bfloat16).astype(mx.float32)
    lg = h @ wf.T
    lg = mx.tanh(lg / CAP) * CAP
    return lg[..., :VOUT]


stock_c = mx.compile(stock)
res = {}
for fma in (True, False):
    bad_e = bad_c = 0
    maxd = 0.0
    for seed in range(12):
        mx.random.seed(100 + seed)
        scale = [0.1, 1.0, 10.0, 50.0][seed % 4]
        h = (mx.random.normal((1, 1, D)) * scale).astype(mx.bfloat16).astype(mx.float32)
        a = fk.lm_head_gemv_exact(h, w, VOUT, CAP, use_fma=fma)
        b = stock(h)
        c = stock_c(h)
        mx.eval(a, b, c)
        ne = int((a != b).sum().item())
        nc = int((a != c).sum().item())
        bad_e += ne
        bad_c += nc
        maxd = max(maxd, float(mx.abs(a - b).max().item()))
    res[fma] = (bad_e, bad_c, maxd)
    print(f"fma={fma}: mismatches vs eager stock {bad_e}, vs compiled stock {bad_c}, max|d| {maxd:.3e}  (12 vectors x {VOUT})")
# stock eager vs compiled consistency
print("OK" if min(r[0] + r[1] for r in res.values()) == 0 else "FAIL")

# Sensitivity control: the older head_approx kernel (different FP32 order)
# must NOT be bitwise here, or the comparison above proves nothing.
mx.random.seed(100)
h = (mx.random.normal((1, 1, D)) * 1.0).astype(mx.bfloat16).astype(mx.float32)
a = fk.lm_head_gemv(h, w, VOUT, CAP)
b = stock(h)
mx.eval(a, b)
n = int((a != b).sum().item())
print(f"control head_approx vs stock: {n} mismatches (expect > 0), max|d| {float(mx.abs(a - b).max().item()):.3e}")
