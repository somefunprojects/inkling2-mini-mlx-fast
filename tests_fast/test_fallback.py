"""Fallbacks and the to_cache round trip (FP32 so 'exact' means ~1e-6).
- ring_extra=7 (verify-ready ring) matches stock
- decoding past the largest bucket hands off to stock via to_cache, matches stock
- from_cache -> to_cache -> stock decode matches stock (B=1 and left-padded B=4)
- sharded model / too-long prompt raise Unsupported"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models import cache as C
from mlx_vlm.models.inkling2.fast_decode import FastDecoder, Unsupported
lm = tiny.build(dtype=mx.float32)

def caches(B, lens):
    T = max(lens); rng = np.random.default_rng(B)
    ids = np.zeros((B, T), np.int32)
    for b, n in enumerate(lens): ids[b, T - n:] = rng.integers(1, 500, size=n)
    if len(set(lens)) == 1:
        c = lm.make_cache()
    else:
        c = []
        lp = [T - n for n in lens]
        for cl in lm.make_cache():
            kv, conv = cl.caches
            kvb = C.BatchRotatingKVCache(kv.max_size, lp) if isinstance(kv, C.RotatingKVCache) else C.BatchKVCache(lp)
            conv.left_padding = mx.array(lp); c.append(C.CacheList(kvb, conv))
    lg = lm(inputs=mx.array(ids), cache=c).logits[:, -1]
    return c, lg

def stock_steps(c, lg, n):
    outs = []; y = mx.argmax(lg, -1)
    toks = []
    for _ in range(n):
        toks.append(y); lg = lm(inputs=y[:, None], cache=c).logits[:, -1]; mx.eval(lg); outs.append(np.array(lg)); y = mx.argmax(lg, -1)
    return np.stack(outs), toks

def fast_steps(step, toks):
    outs = []
    for y in toks:
        lg, _ = step(y[:, None]); mx.eval(lg); outs.append(np.array(lg))
    return np.stack(outs)

N = 60
for B, lens in ((1, [12]), (4, [12, 7, 12, 3])):
    c, lg = caches(B, lens); ref, toks = stock_steps(c, lg, N)
    for name, kw in (("ring_extra=7", dict(ring_extra=7, buckets=(64, 128))),
                     ("bucket overflow->stock", dict(buckets=(32,))),
                     ("ring_extra=7+overflow", dict(ring_extra=7, buckets=(24, 40)))):
        c2, _ = caches(B, lens); dec = FastDecoder(lm, **kw).from_cache(c2)
        got = fast_steps(dec.step, toks)
        print(f"B={B} {name:24s} max|d|={np.abs(got - ref).max():.1e} fell_back={dec.fallback is not None}")
    c3, _ = caches(B, lens); dec = FastDecoder(lm, buckets=(64,), ring_extra=3).from_cache(c3)
    for y in toks[:20]: dec.step(y[:, None])
    back = dec.to_cache()
    got = fast_steps(lambda t: (lm(inputs=t, cache=back).logits[:, -1], None), toks[20:])
    print(f"B={B} {'20 fast steps, to_cache, stock':24s} max|d|={np.abs(got - ref[20:]).max():.1e}")
try:
    lm.sharding_group = object(); FastDecoder(lm); print("sharded: NOT refused")
except Unsupported as e: print("sharded refused:", e)
finally: lm.sharding_group = None
c, _ = caches(1, [40])
try: FastDecoder(lm, buckets=(32,)).from_cache(c); print("too long: NOT refused")
except Unsupported as e: print("too long refused:", e)
