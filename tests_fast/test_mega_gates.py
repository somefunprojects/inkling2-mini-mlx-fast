"""megafix: gate and layout checks for mega_ctrl / mega_norm (REVIEW_MEGA L1, L2, L4).
- L1: ctrl_supported refuses non-BF16 or mis-shaped forget/gain weights; accepts the real layouts
  (3 stacked controllers [3,R,2D]/[3,D,R], one mlp controller [R,2D]/[D,R]).
- L2: rms_norm_cast reads a strided weight view correctly (bitwise vs RMSNorm with that view);
  norm_supported refuses a weight whose length is not D.
- L4: controllers() fed non-contiguous mixed / messages views is bitwise vs the stock chain.
Real dims D=2304, R=64."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: F401
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.inkling2 import mega_kernels as mk
from mlx_vlm.models.inkling2.fast_decode import _controllers3

D, R, EPS = 2304, 64, 1e-6
fails = []


def check(name, cond):
    print(f"{'ok  ' if cond else 'FAIL'} {name}", flush=True)
    if not cond:
        fails.append(name)


bf = mx.bfloat16
x = mx.zeros((1, 1, D))
dw3, fw3, gw3 = mx.zeros((3, R, 2 * D), bf), mx.zeros((3, D, R), bf), mx.zeros((3, D, R), bf)
dw1, fw1, gw1 = mx.zeros((R, 2 * D), bf), mx.zeros((D, R), bf), mx.zeros((D, R), bf)
check("L1 ctrl3 real layout accepted", mk.ctrl_supported(x, dw3, fw3, gw3, 3))
check("L1 mlp ctrl real layout accepted", mk.ctrl_supported(x, dw1, fw1, gw1, 1))
check("L1 fw f32 refused", not mk.ctrl_supported(x, dw3, fw3.astype(mx.float32), gw3, 3))
check("L1 gw f16 refused", not mk.ctrl_supported(x, dw3, fw3, gw3.astype(mx.float16), 3))
check("L1 gw uint32 (packed) refused", not mk.ctrl_supported(x, dw3, fw3, mx.zeros((3, D, R // 8), mx.uint32), 3))
check("L1 fw transposed shape refused", not mk.ctrl_supported(x, dw3, mx.zeros((3, R, D), bf), gw3, 3))
check("L1 fewer stacked than n_ctrl refused", not mk.ctrl_supported(x, dw3[:2], fw3[:2], gw3[:2], 3))
check("L1 missing fw/gw refused", not mk.ctrl_supported(x, dw3))
check("L1 2-D fw with 3-D dw refused", not mk.ctrl_supported(x, dw3, fw1, gw3, 3))

# L2: strided weight views (stride 2, and a column of a 2-D array: stride D... use [D,2][:,1])
nm = nn.RMSNorm(D, eps=EPS)
bad = 0
for seed in range(10):
    mx.random.seed(seed)
    big = (1 + 0.3 * mx.random.normal((2 * D,))).astype(bf)
    col = (1 + 0.3 * mx.random.normal((D, 3))).astype(bf)
    xs = [mx.random.normal((1, 1, D)) * (0.5 + seed) for _ in range(3)]
    ws = [big[::2], col[:, 1], big[D:]]
    mx.eval(big, col, xs)
    got = mk.rms_norm_cast(xs, ws, EPS)
    for i in range(3):
        nm.weight = ws[i]
        ref = nm(xs[i]).astype(bf).reshape(D)
        nm.weight = mx.array(ws[i])  # contiguous copy, same values
        ref2 = nm(xs[i]).astype(bf).reshape(D)
        ok = mx.array_equal(ref, got[i]).item() and mx.array_equal(ref2, got[i]).item()
        bad += not ok
check("L2 strided weight views bitwise (30 rows)", bad == 0)
check("L2 norm gate accepts [D] bf16 view", mk.norm_supported(xs[0], big[::2]))
check("L2 norm gate refuses weight of length 2D", not mk.norm_supported(xs[0], big))
check("L2 norm gate refuses f32 weight", not mk.norm_supported(xs[0], big[::2].astype(mx.float32)))
check("L2 norm gate refuses 2 rows", not mk.norm_supported(mx.zeros((2, 1, D)), big[::2]))

# L4: non-contiguous mixed / messages into controllers
bad = 0
for seed in range(10):
    mx.random.seed(50 + seed)
    dw = (mx.random.normal((3, R, 2 * D)) * 0.02).astype(bf)
    fw = (mx.random.normal((3, D, R)) * 0.2).astype(bf)
    gw = (mx.random.normal((3, D, R)) * 0.2).astype(bf)
    base = mx.random.normal((1, 1, 6, D)) * 2
    mixed = base[..., ::2, :]                      # [1,1,3,D] stride 2 rows
    msgs = mx.random.normal((1, 1, D, 4)).swapaxes(-1, -2) * 0.7  # [1,1,4,D] transposed view
    mx.eval(dw, fw, gw, base, msgs)
    parts = [mixed[..., i, :] for i in range(3)]
    ref = _controllers3(mx.stack(parts), mx.moveaxis(msgs[..., :3, :], -2, 0), dw, fw, gw, EPS)
    got = mk.controllers(mixed.reshape(3, D), msgs.reshape(4, D), 0, dw, fw, gw, EPS, 3, 3).reshape(3, 1, 1, D)
    bad += not mx.array_equal(ref, got).item()
check("L4 ctrl3 on non-contiguous mixed/messages bitwise (10 seeds)", bad == 0)
print("GATES", "PASS" if not fails else f"FAIL {fails}")
