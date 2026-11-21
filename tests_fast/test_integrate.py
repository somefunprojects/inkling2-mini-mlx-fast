"""Cross-process gate on the tiny model.
  ref  : (run WITHOUT INKLING_FAST) stock teacher-forced logits -> npz
  check: (run with INKLING_FAST=all or any) FastDecoder vs saved stock refs
Cases: B=1 (unpadded), B=4/8 with left-padded prompts of different lengths
(stock path = mlx-vlm batch caches: BatchRotatingKVCache/BatchKVCache/ArraysCache
with left_padding, exactly as generate/ar.py to_batch_cache builds them)."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import numpy as np
import mlx.core as mx
from mlx_vlm.models import cache as C

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["ref", "check"])
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--batches", default="1,4,8")
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--same-len", action="store_true")
ap.add_argument("--out", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "fd_refs"))
a = ap.parse_args()
mx.set_memory_limit(4 * 1024**3)
os.makedirs(a.out, exist_ok=True)
dt = getattr(mx, a.dtype)
lm = tiny.build(dtype=dt)


def make_cache(lp):
    if lp is None:
        return lm.make_cache()
    out = []
    for c in lm.make_cache():
        kv, conv = c.caches
        kvb = (C.BatchRotatingKVCache(kv.max_size, lp) if isinstance(kv, C.RotatingKVCache)
               else C.BatchKVCache(lp))
        conv.left_padding = mx.array(lp)
        out.append(C.CacheList(kvb, conv))
    return out


def prompts(B):
    rng = np.random.default_rng(B)
    if B == 1 or a.same_len:
        lens = [12] * B
    else:
        lens = [int(x) for x in rng.integers(5, 21, size=B)]
        lens[0] = 20
    T = max(lens)
    ids = np.zeros((B, T), np.int32)
    for b, n in enumerate(lens):
        ids[b, T - n:] = rng.integers(1, 500, size=n)
    lp = None if B == 1 else [T - n for n in lens]
    return mx.array(ids), lp, lens


def tag(B):
    return os.path.join(a.out, f"{a.dtype}_B{B}{'_same' if a.same_len else ''}.npz")


def lsm(x):
    x = x.astype(np.float64); m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


for B in [int(x) for x in a.batches.split(",")]:
    ids, lp, lens = prompts(B)
    cache = make_cache(lp)
    logits = lm(inputs=ids, cache=cache).logits[:, -1]
    if a.mode == "ref":
        outs, toks = [], []
        for t in range(a.steps):
            mx.eval(logits); outs.append(np.array(logits)); toks.append(outs[-1].argmax(-1))
            logits = lm(inputs=mx.array(toks[-1].astype(np.int32))[:, None], cache=cache).logits[:, -1]
        np.savez(tag(B), logits=np.stack(outs), toks=np.stack(toks), lens=np.array(lens))
        print(f"ref B={B} lens={lens} saved")
        continue
    from mlx_vlm.models.inkling2.fast_decode import FastDecoder
    ref = np.load(tag(B)); toks = ref["toks"]
    dec = FastDecoder(lm, buckets=(32, 64, 128, 256)).from_cache(cache)
    outs = [np.array(logits)]
    for t in range(a.steps - 1):
        lg, _ = dec.step(mx.array(toks[t].astype(np.int32))[:, None]); mx.eval(lg); outs.append(np.array(lg))
    new = np.stack(outs); r = ref["logits"]
    d = np.abs(new - r).max()
    am = (new.argmax(-1) == r.argmax(-1))
    lr, ln = lsm(r), lsm(new); kl = (np.exp(lr) * (lr - ln)).sum(-1)
    print(f"{a.dtype:8s} B={B} lens={lens} max|dlogit|={d:.2e} argmax {am.sum()}/{am.size} "
          f"mean KL={kl.mean():.2e} max KL={kl.max():.2e} flags={os.environ.get('INKLING_FAST','')}")
