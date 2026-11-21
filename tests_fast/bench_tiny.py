"""Step timing + graph node counts: stock vs FastDecoder (tiny model)."""
import sys, os, time, io, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
mx.set_memory_limit(4 * 1024**3)
lm = tiny.build()
P, N = 40, 300
BATCH = int(os.environ.get('B', '1'))
prompt = mx.array(np.random.default_rng(0).integers(0, 500, size=(BATCH, P)).astype(np.int32))

def count_nodes(outs):
    path = "/tmp/_g.dot" if False else os.path.join(os.environ.get("TMPDIR", "/tmp"), "_g.dot")
    mx.export_to_dot(path, *outs)
    txt = open(path).read()
    prims = re.findall(r'label\s*=\s*"([A-Za-z_]+)', txt)
    from collections import Counter
    c = Counter(p for p in prims)
    return sum(v for k, v in c.items()), c

# stock
cache = lm.make_cache(); lg = lm(inputs=prompt, cache=cache).logits[:, -1]; mx.eval(lg)
tok = mx.argmax(lg, -1)[:, None]
g = lm(inputs=tok, cache=cache).logits[:, -1]
n_stock, c_stock = count_nodes([g]); mx.eval(g)
def stock_loop(n, pipelined):
    global tok
    t0 = time.perf_counter(); build = 0.0
    y = mx.argmax(g, -1)[:, None]
    for i in range(n):
        tb = time.perf_counter()
        l = lm(inputs=y, cache=cache).logits[:, -1]
        y = mx.argmax(l, -1)[:, None]
        build += time.perf_counter() - tb
        if pipelined: mx.async_eval(y)
        else: mx.eval(y)
    mx.eval(y)
    return (time.perf_counter() - t0) / n * 1e3, build / n * 1e3

# fast
cache2 = lm.make_cache(); lg2 = lm(inputs=prompt, cache=cache2).logits[:, -1]
dec = FastDecoder(lm, buckets=(4096,)).from_cache(cache2)
y2 = mx.argmax(lg2, -1).astype(mx.int32)[:, None]
out = dec._step_flat(y2, *dec._flatten(dec.state))
n_fast_unc, c_fast = count_nodes([out[0]])
def fast_loop(n, pipelined):
    y = y2
    t0 = time.perf_counter(); build = 0.0
    for i in range(n):
        tb = time.perf_counter()
        _, nxt = dec.step(y)
        y = nxt[:, None]
        build += time.perf_counter() - tb
        if pipelined: mx.async_eval(y)
        else: mx.eval(y)
    mx.eval(y)
    return (time.perf_counter() - t0) / n * 1e3, build / n * 1e3

fast_loop(5, False)  # trace + warm
_, nxt = dec.step(y2); mx.eval(nxt)
stock_loop(5, False)
for name, fn in (("stock", stock_loop), ("fast", fast_loop)):
    for pip in (False, True):
        ms, b = fn(N, pip)
        print(f"B={BATCH} {name:5s} {'pipelined' if pip else 'synced   '} {ms:6.2f} ms/step  (python graph build {b:5.2f} ms)")
print(f"B={BATCH} flags={os.environ.get('INKLING_FAST','')!r} graph nodes per step: stock {n_stock}, fast uncompiled {n_fast_unc}")
print("peak mem GB", mx.get_peak_memory() / 1e9)
