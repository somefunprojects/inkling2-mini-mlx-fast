# Review F1 repro. Usage: PYTHONPATH=<repo> python <this> <repo>. Expect 30/30 bitwise for every variant.
"""Tiny model, window 1024: FD exact vs true stock around p=1000..1030; variants that
grow the global cap beyond p+1 (as spec_step does) before plain L=1 steps."""
import sys, os
sys.path.insert(0, os.path.join(sys.argv[1], "tests_fast"))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
mx.set_memory_limit(8 * 1024**3)
lm = tiny.build(window=1024, rel_extent=1024, n_floor=8)
rng = np.random.default_rng(0)
T0, N = int(os.environ.get("T0", 1000)), int(os.environ.get("N", 30))
toks = mx.array(rng.integers(1, 500, size=(1, T0 + N)).astype(np.int32))
def stock():
    c = lm.make_cache(); lg = lm(inputs=toks[:, :T0], cache=c).logits[:, -1]; out = []
    for t in range(N):
        mx.eval(lg); out.append(np.array(lg.astype(mx.float32)))
        lg = lm(inputs=toks[:, T0 + t:T0 + t + 1], cache=c).logits[:, -1]
    return np.stack(out)
with tiny.true_stock():
    ref = stock()
def fd(grow=0, ring_extra=0):
    c = lm.make_cache(); lg = lm(inputs=toks[:, :T0], cache=c).logits[:, -1]; mx.eval(lg)
    d = FastDecoder(lm, ring_extra=ring_extra).from_cache(c)
    if grow: d._grow(d.pos + grow)
    out = [np.array(lg.astype(mx.float32))]; caps = []
    for t in range(N - 1):
        l, _ = d.step(toks[:, T0 + t:T0 + t + 1]); mx.eval(l); out.append(np.array(l)); caps.append(d.cap)
    return np.stack(out), sorted(set(caps))
for name, kw in (("plain", {}), ("ring_extra7", dict(ring_extra=7)), ("grow+30", dict(grow=30)), ("grow+23", dict(grow=23)), ("grow+8,ring7", dict(grow=8, ring_extra=7))):
    o, caps = fd(**kw)
    dd = np.abs(o - ref).reshape(N, -1).max(-1); bad = np.nonzero(dd)[0]
    print(f"{name:14s} caps={caps} positions {T0}..{T0+N-1}: bitwise steps {N-len(bad)}/{N}; diff positions {[T0+int(i) for i in bad][:12]} max {dd.max():.2e}", flush=True)
