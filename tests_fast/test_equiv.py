"""Stock decode vs FastDecoder on the tiny random model.
Teacher-forces the stock greedy tokens through both; reports max |dlogit|, argmax match, KL."""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import numpy as np
import mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--prompt", type=int, default=10)
ap.add_argument("--sinks", type=int, default=1)
ap.add_argument("--compiled", type=int, default=1)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--exact", type=int, default=1, help="FastDecoder exact decode (0 = old approx path)")
ap.add_argument("--flagged-ref", type=int, default=0,
                help="1 = reference runs with the env INKLING_FAST flags (old behaviour); default true stock")
a = ap.parse_args()
mx.set_memory_limit(4 * 1024**3)

lm = tiny.build(window=int(os.environ.get("WIN", "16")), dtype=getattr(mx, os.environ.get("DT", "bfloat16")))
rng = np.random.default_rng(0)
prompt = mx.array(rng.integers(0, 500, size=(a.batch, a.prompt)))

def stock_run(tokens_forced=None, chunk=None):
    cache = lm.make_cache()
    if chunk is None:
        logits = lm(inputs=prompt, cache=cache).logits[:, -1]
    else:
        for s in range(0, a.prompt, chunk):
            logits = lm(inputs=prompt[:, s:s + chunk], cache=cache).logits[:, -1]
    outs, toks = [], []
    for t in range(a.steps):
        mx.eval(logits); outs.append(np.array(logits.astype(mx.float32)))
        nxt = outs[-1].argmax(-1) if tokens_forced is None else tokens_forced[t]
        toks.append(nxt)
        logits = lm(inputs=mx.array(nxt)[:, None], cache=cache).logits[:, -1]
    return np.stack(outs), np.stack(toks), cache

def lsm(x):
    x = x.astype(np.float64); m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))

def cmp(name, ref, new):
    d = np.abs(ref - new).max(axis=(1, 2)) if ref.ndim == 3 else np.abs(ref - new).max(-1)
    am = (ref.argmax(-1) == new.argmax(-1)).all(axis=-1)
    lr, ln = lsm(ref), lsm(new)
    kl = (np.exp(lr) * (lr - ln)).sum(-1).mean(-1)
    print(f"{name:28s} max|dlogit|={d.max():.3e} (median step {np.median(d):.2e})  argmax match {am.sum()}/{len(am)}  "
          f"mean KL={kl.mean():.2e} max KL={kl.max():.2e}  first mismatch={int(np.argmin(am)) if not am.all() else -1}")
    return d

import contextlib
ref_ctx = contextlib.nullcontext if a.flagged_ref else tiny.true_stock
with ref_ctx():
    ref, toks, _ = stock_run()
    # noise floor: same stock code, prompt prefilled token-by-token instead of in one chunk
    alt, _, _ = stock_run(tokens_forced=toks, chunk=1)
print("reference:", "flagged (env INKLING_FAST)" if a.flagged_ref else "true stock (all INKLING_FAST flags off)")
cmp("stock(chunk=1) vs stock", ref, alt)

# fast: prefill with stock, convert, teacher-force
cache = lm.make_cache()
logits0 = lm(inputs=prompt, cache=cache).logits[:, -1]
dec = FastDecoder(lm, buckets=tuple(int(x) for x in os.environ.get("BK","32,64,128,256").split(",")), use_sinks=bool(a.sinks), exact=bool(a.exact)).from_cache(cache)
outs = [np.array(logits0)]
caps = []
for t in range(a.steps - 1):
    lg, _ = dec.step(mx.array(toks[t])[:, None], compiled=bool(a.compiled))
    mx.eval(lg); outs.append(np.array(lg)); caps.append(dec.cap)
fast = np.stack(outs)
print(f"positions {a.prompt}..{a.prompt + a.steps - 1}; window {lm.config.sliding_window_size}; rel_extent {lm.config.rel_extent}; buckets used {sorted(set(caps))}")
d = cmp(f"fast(exact={a.exact},sinks={a.sinks},compiled={a.compiled}) vs stock", ref, fast)
print("per-step max|dlogit| (every 8th):", " ".join(f"{x:.1e}" for x in d[:100]))
