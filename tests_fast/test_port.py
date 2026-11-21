"""env-flagged language path and FastDecoder vs TRUE stock on the
tiny random model, teacher-forced. Bitwise check of logits per step.
Usage: INKLING_FAST=<flags> python test_port.py [--steps 40] [--prompt 20] [--batch 1]
       DT=float32 for the FP32-exact check. WIN sets the sliding window."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import numpy as np
import mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--prompt", type=int, default=20)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--fd", type=int, default=1)
a = ap.parse_args()
mx.set_memory_limit(4 * 1024**3)
if os.environ.get("HD"):  # real head_dim (qkvclip needs 128)
    _cfg = tiny.tiny_config
    def _hd_config(**kw):
        c = _cfg(**kw)
        c.head_dim = c.swa_head_dim = int(os.environ["HD"])
        return c
    tiny.tiny_config = _hd_config
lm = tiny.build(window=int(os.environ.get("WIN", "16")),
                dtype=getattr(mx, os.environ.get("DT", "bfloat16")))
rng = np.random.default_rng(0)
prompt = mx.array(rng.integers(0, 500, size=(a.batch, a.prompt)))


def run(forced=None):
    cache = lm.make_cache()
    logits = lm(inputs=prompt, cache=cache).logits[:, -1]
    outs, toks = [], []
    for t in range(a.steps):
        mx.eval(logits); outs.append(np.array(logits.astype(mx.float32)))
        nxt = outs[-1].argmax(-1) if forced is None else forced[t]
        toks.append(nxt)
        logits = lm(inputs=mx.array(nxt)[:, None], cache=cache).logits[:, -1]
    return np.stack(outs), np.stack(toks)


def report(name, ref, new):
    d = np.abs(ref - new).max(axis=tuple(range(1, ref.ndim)))
    same = int((d == 0).sum())
    am = (ref.argmax(-1) == new.argmax(-1)).all(-1).sum()
    print(f"{name:34s} bitwise steps {same}/{len(d)}  max|dlogit| {d.max():.3e}  argmax {am}/{len(d)}")
    return same == len(d)


with tiny.true_stock():
    ref, toks = run()
new, _ = run(forced=toks)
ok = report(f"lang[{os.environ.get('INKLING_FAST','')}]", ref, new)
if a.fd:
    cache = lm.make_cache()
    l0 = lm(inputs=prompt, cache=cache).logits[:, -1]
    dec = FastDecoder(lm, buckets=(32, 64, 128, 256)).from_cache(cache)
    outs = [np.array(l0.astype(mx.float32))]
    for t in range(a.steps - 1):
        lg, _ = dec.step(mx.array(toks[t])[:, None])
        mx.eval(lg); outs.append(np.array(lg))
    ok &= report("fast_decode(compiled)", ref, np.stack(outs))
print("PASS bitwise" if ok else "NOT bitwise")
