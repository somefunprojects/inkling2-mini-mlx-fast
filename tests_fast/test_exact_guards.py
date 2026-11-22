"""Review F2/F4 guards (tiny model; run with FORK on PYTHONPATH, any INKLING_FAST).
1. check_exact_runtime refuses an unverified MLX version unless INKLING_FAST_UNVERIFIED_OK=<ver>.
2. Exact FastDecoder raises Unsupported on left-padded (unequal) batches, and on use_kernels=False.
3. Equal-length B=4 batch: checks exact FD against the normal mlx-vlm path in the same process, not stock when INKLING_FAST is set (this case only; other batches can differ, see DETAILS.md)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import numpy as np
import mlx.core as mx
from mlx_vlm.models import cache as C
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2.fast_decode import FastDecoder, Unsupported

ok = True
# 1. version guard
real = mx.__version__
try:
    fk.check_exact_runtime("test"); print("1a verified runtime accepted: True")
except RuntimeError as e:
    print("1a verified runtime accepted: False", e); ok = False
mx.__version__ = "9.9.9"
try:
    fk.check_exact_runtime("test"); print("1b unverified refused: False"); ok = False
except RuntimeError:
    print("1b unverified refused: True")
os.environ["INKLING_FAST_UNVERIFIED_OK"] = "9.9.9"
try:
    fk.check_exact_runtime("test"); print("1c override accepted: True")
except RuntimeError:
    print("1c override accepted: False"); ok = False
del os.environ["INKLING_FAST_UNVERIFIED_OK"]; mx.__version__ = real

mx.set_memory_limit(4 * 1024**3)
lm = tiny.build(dtype=mx.bfloat16)


def batch_cache(lp):
    out = []
    for c in lm.make_cache():
        kv, conv = c.caches
        kvb = (C.BatchRotatingKVCache(kv.max_size, lp) if isinstance(kv, C.RotatingKVCache)
               else C.BatchKVCache(lp))
        conv.left_padding = mx.array(lp)
        out.append(C.CacheList(kvb, conv))
    return out


rng = np.random.default_rng(0)
# 2. padded batch refused
lens = [20, 12, 9, 20]; T = 20
ids = np.zeros((4, T), np.int32)
for b, n in enumerate(lens):
    ids[b, T - n:] = rng.integers(1, 500, size=n)
cache = batch_cache([T - n for n in lens]); mx.eval(lm(inputs=mx.array(ids), cache=cache).logits)
try:
    FastDecoder(lm, buckets=(32, 64, 128)).from_cache(cache); print("2a padded refused: False"); ok = False
except Unsupported as e:
    print("2a padded refused: True |", e)
os.environ["INKLING_FAST_DECODE_PADDED"] = "1"
try:
    FastDecoder(lm, buckets=(32, 64, 128)).from_cache(cache); print("2b padded opt-in accepted: True")
except Unsupported:
    print("2b padded opt-in accepted: False"); ok = False
del os.environ["INKLING_FAST_DECODE_PADDED"]
try:
    FastDecoder(lm, buckets=(32, 64, 128), use_kernels=False); print("2c exact without kernels refused: False"); ok = False
except Unsupported:
    print("2c exact without kernels refused: True")
FastDecoder(lm, buckets=(32, 64, 128), use_kernels=False, exact=False); print("2d approx without kernels allowed: True")

# 3. equal-length batch, this case only (other batches can differ: see DETAILS.md)
ids = mx.array(rng.integers(1, 500, size=(4, 12)).astype(np.int32))
toks = rng.integers(1, 500, size=(30, 4)).astype(np.int32)
c1 = batch_cache([0, 0, 0, 0]); c2 = batch_cache([0, 0, 0, 0])
l1 = lm(inputs=ids, cache=c1).logits[:, -1]; l2 = lm(inputs=ids, cache=c2).logits[:, -1]; mx.eval(l1, l2)
dec = FastDecoder(lm, buckets=(32, 64, 128)).from_cache(c2)
eq = True
for t in range(30):
    a = lm(inputs=mx.array(toks[t])[:, None], cache=c1).logits[:, -1]
    b, _ = dec.step(mx.array(toks[t])[:, None]); mx.eval(a, b)
    eq &= bool(mx.array_equal(a, b).item())
print("3 equal-length B=4 FD bitwise vs stock batch:", eq); ok &= eq
print("ALL OK" if ok else "FAILED"); assert ok
