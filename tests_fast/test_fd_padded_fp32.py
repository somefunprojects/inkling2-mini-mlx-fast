"""F2: is FastDecoder's left-padded batch difference ordering-only?  (tiny model)

Truth = stock B=1 per row, unpadded, teacher-forced with the same tokens. In the
given dtype (float32 by default) compare against truth:
  stock-padded (mlx-vlm batch caches with left_padding, as generate/ar.py)
  FD-padded    (FastDecoder.from_cache on those batch caches, exact default)
If FD's distance to truth is of the same order as stock-padded's (both
rounding-level), FD's batch logic is exact and differs only in reduction order.
A logic error (wrong positions, attending padding, missing keys) shows up as an
FD distance orders of magnitude above stock's.
Usage: python tests_fast/test_fd_padded_fp32.py [--dtype float32] [--steps 40]"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.environ.get("FORK", ""), "tests_fast"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import numpy as np
import mlx.core as mx
from mlx_vlm.models import cache as C
from mlx_vlm.models.inkling2.fast_decode import FastDecoder

ap = argparse.ArgumentParser(); ap.add_argument("--dtype", default="float32")
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--batches", default="4,8")
a = ap.parse_args()
mx.set_memory_limit(4 * 1024**3)
lm = tiny.build(dtype=getattr(mx, a.dtype))


def batch_cache(lp):
    out = []
    for c in lm.make_cache():
        kv, conv = c.caches
        kvb = (C.BatchRotatingKVCache(kv.max_size, lp) if isinstance(kv, C.RotatingKVCache)
               else C.BatchKVCache(lp))
        conv.left_padding = mx.array(lp)
        out.append(C.CacheList(kvb, conv))
    return out


def stats(x, ref):
    d = np.abs(x.astype(np.float64) - ref)
    am = int((x.argmax(-1) != ref.argmax(-1)).sum())
    return d.max(), np.median(d.reshape(d.shape[0], -1).max(-1)), am


worst = []
for B in [int(x) for x in a.batches.split(",")]:
    rng = np.random.default_rng(B)
    lens = [int(x) for x in rng.integers(5, 21, size=B)]; lens[0] = 20
    T = max(lens)
    ids = np.zeros((B, T), np.int32)
    for b, n in enumerate(lens):
        ids[b, T - n:] = rng.integers(1, 500, size=n)
    toks = rng.integers(1, 500, size=(a.steps, B)).astype(np.int32)  # teacher-forced inputs
    # truth: per row, unpadded, B=1
    rows = []
    for b, n in enumerate(lens):
        c = lm.make_cache()
        lg = lm(inputs=mx.array(ids[b:b + 1, T - n:]), cache=c).logits[:, -1]
        out = [np.array(lg.astype(mx.float32))[0]]
        for t in range(a.steps):
            lg = lm(inputs=mx.array(toks[t, b:b + 1])[:, None], cache=c).logits[:, -1]
            out.append(np.array(lg.astype(mx.float32))[0])
        rows.append(np.stack(out))
    truth = np.stack(rows, 1).astype(np.float64)  # [steps+1, B, V]
    # stock padded
    cache = batch_cache([T - n for n in lens])
    lg0 = lm(inputs=mx.array(ids), cache=cache).logits[:, -1]
    sp = [np.array(lg0.astype(mx.float32))]
    for t in range(a.steps):
        lg = lm(inputs=mx.array(toks[t])[:, None], cache=cache).logits[:, -1]
        sp.append(np.array(lg.astype(mx.float32)))
    sp = np.stack(sp)
    # FD padded (fresh padded prefill, then FastDecoder)
    cache = batch_cache([T - n for n in lens])
    lg0 = lm(inputs=mx.array(ids), cache=cache).logits[:, -1]; mx.eval(lg0)
    dec = FastDecoder(lm, buckets=(32, 64, 128, 256)).from_cache(cache)
    fd = [np.array(lg0.astype(mx.float32))]
    for t in range(a.steps):
        lg, _ = dec.step(mx.array(toks[t])[:, None]); mx.eval(lg)
        fd.append(np.array(lg.astype(mx.float32)))
    fd = np.stack(fd)
    scale = np.abs(truth).max()
    for name, x in (("stock-padded", sp), ("FD-padded", fd)):
        mx_, med, am = stats(x[1:], truth[1:])
        print(f"{a.dtype} B={B} lens={lens} {name:12s} vs truth(B=1 unpadded): max|d|={mx_:.3e} "
              f"median-step max|d|={med:.3e} argmax diffs={am} (|logit| max {scale:.2f})", flush=True)
    m1, _, _ = stats(fd[1:], sp[1:].astype(np.float64))
    print(f"{a.dtype} B={B} FD-padded vs stock-padded: max|d|={m1:.3e}", flush=True)
    worst.append((stats(sp[1:], truth[1:])[0], stats(fd[1:], truth[1:])[0]))
r = max(f / max(s, 1e-30) for s, f in worst)
print(f"ratio max(FD-truth)/max(stock-padded-truth) over batches: {r:.2f}  "
      f"({'ordering-level' if r < 10 else 'FD has an extra error source'})")
