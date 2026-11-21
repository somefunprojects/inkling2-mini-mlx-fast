"""Reviewer checks for megaperf (rg2, dpar at B*L>1 and inf/NaN). Flags toggled via fk.FLAGS."""
import sys, os
import mlx.core as mx
mx.set_memory_limit(20 << 30)
from mlx_vlm.models.inkling2 import fast_kernels as fk

res = {}
def cmp(name, a, b, nan=False):
    ok = mx.array_equal(a, b, equal_nan=nan).item()
    t = res.setdefault(name, [0, 0]); t[1] += 1; t[0] += ok
    if not ok:
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
        print(f"  DIFF {name}: {int((d > 0).sum().item())} elems differ", flush=True)

def flag(n, on):
    fk.FLAGS = (fk.FLAGS | {n}) if on else (fk.FLAGS - {n})

# ---- rg2 vs rg1: wrap edges, B=2, strided fallback, misaligned view ----
def rg(kb, vb, qp, W, R, on):
    flag("mega_rg2", on)
    o = fk.ring_gather(kb, vb, qp, W, R); mx.eval(o); return o

D, NKV = 128, 8
for (W, R) in ((1024, 1024), (1024, 1536), (8, 12)):
    kb = mx.random.normal((2, NKV, R + 1, D)).astype(mx.bfloat16)
    vb = mx.random.normal((2, NKV, R + 1, D)).astype(mx.bfloat16)
    mx.eval(kb, vb)
    ps = sorted({0, 1, W - 2, W - 1, W, W + 1, R - 1, R, R + 1, 2 * R - 1, 2 * R + 3, 5000, 9000})
    for p in ps:
        qp = mx.array([p, max(0, p - 3)], dtype=mx.int32)
        a, b = rg(kb, vb, qp, W, R, False), rg(kb, vb, qp, W, R, True)
        cmp(f"rg2 W{W} R{R} contiguous", a[0], b[0]); cmp(f"rg2 W{W} R{R} contiguous", a[1], b[1])
    # innermost-strided view (fallback branch)
    kt = mx.random.normal((2, NKV, D, R + 1)).astype(mx.bfloat16).swapaxes(2, 3)
    vt = mx.random.normal((2, NKV, D, R + 1)).astype(mx.bfloat16).swapaxes(2, 3)
    mx.eval(kt, vt)
    qp = mx.array([W + 5, 3], dtype=mx.int32)
    a, b = rg(kt, vt, qp, W, R, False), rg(kt, vt, qp, W, R, True)
    cmp(f"rg2 W{W} R{R} strided", a[0], b[0]); cmp(f"rg2 W{W} R{R} strided", a[1], b[1])
    # innermost-contiguous but row stride 129 / offset 1 element: not 8-byte aligned
    big = mx.random.normal((2, NKV, R + 1, D + 1)).astype(mx.bfloat16); mx.eval(big)
    km = big[..., 1:]
    vm = big[..., :D]
    qp = mx.array([W + 5, 3], dtype=mx.int32)
    a, b = rg(km, vm, qp, W, R, False), rg(km, vm, qp, W, R, True)
    cmp(f"rg2 W{W} R{R} MISALIGNED view (latent)", a[0], b[0]); cmp(f"rg2 W{W} R{R} MISALIGNED view (latent)", a[1], b[1])

# ---- dpar attn_res at B*L > 1, and with inf/NaN keys ----
def ar(keys, part, q, on):
    flag("mega_dpar", on); fk._attnres_kernels.clear()
    o = fk.attn_res(keys, part, q, raw_partial=True); mx.eval(o); return o

D = 2304
for (B, L) in ((1, 1), (1, 5), (2, 3), (4, 4), (16, 1)):
    for nk in (1, 3, 6, 8, 12):
        keys = [fk._block_key(mx.random.normal((B, L, D)) * (1 + i)) for i in range(nk)]
        part = (mx.random.normal((B, L, D)) * 2).astype(mx.bfloat16)
        q = mx.random.normal((D,)).astype(mx.bfloat16)
        mx.eval(keys, part, q)
        cmp(f"dpar attnres B*L>1", ar(keys, part, q, False), ar(keys, part, q, True))
for bad in (float("inf"), float("-inf"), float("nan")):
    keys = [fk._block_key(mx.random.normal((1, 1, D))) for i in range(4)]
    k2 = mx.array(keys[1]); k2 = mx.where(mx.arange(D) == 7, mx.array(bad, mx.float32), k2)
    keys[1] = k2
    part = mx.random.normal((1, 1, D)).astype(mx.bfloat16); q = mx.random.normal((D,)).astype(mx.bfloat16)
    mx.eval(keys, part, q)
    cmp(f"dpar attnres key has {bad} (nan-equal)", ar(keys, part, q, False), ar(keys, part, q, True), nan=True)

bad = 0
for k, (ok, n) in sorted(res.items()):
    print(f"{k:50s} {ok}/{n}"); bad += (n - ok) * ("latent" not in k)
print("RV UNIT PASS" if bad == 0 else f"RV UNIT FAIL {bad}")
