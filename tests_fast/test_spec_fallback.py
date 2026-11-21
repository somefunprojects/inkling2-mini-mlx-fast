"""Spec paths past the largest bucket: spec_step, step_captures and
speculative_loop hand the state to the stock cache like step() and keep decoding, instead of raising.
FP32 tiny model, B=1 and B=4 (equal lengths: exact FastDecoder refuses left padding), buckets=(32,) so the hand-off happens ~20 tokens in.
Gate: greedy tokens identical to stock greedy; fallback engaged; step_captures returns caps=None after it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models import cache as C
from mlx_vlm.models.inkling2.fast_decode import FastDecoder, speculative_loop

mx.set_memory_limit(4 * 1024**3)
lm = tiny.build(dtype=mx.float32)
N, OK = 48, True

def setup(B):
    rng = np.random.default_rng(20 + B)
    lens = [12] * B
    T = max(lens); ids = np.zeros((B, T), np.int32)
    for b, n in enumerate(lens): ids[b, T - n:] = rng.integers(1, 500, size=n)
    lp = None if B == 1 else [T - n for n in lens]
    def cache():
        if lp is None: return lm.make_cache()
        out = []
        for c in lm.make_cache():
            kv, conv = c.caches
            kvb = C.BatchRotatingKVCache(kv.max_size, lp) if isinstance(kv, C.RotatingKVCache) else C.BatchKVCache(lp)
            conv.left_padding = mx.array(lp); out.append(C.CacheList(kvb, conv))
        return out
    return mx.array(ids), cache, lens

for B in (1, 4):
    ids, cache, lens = setup(B)
    c = cache(); lg = lm(inputs=ids, cache=c).logits[:, -1]
    first = mx.argmax(lg, -1).astype(mx.int32); ref, y = [], first
    for _ in range(N + 16):
        y = mx.argmax(lm(inputs=y[:, None], cache=c).logits[:, -1], -1).astype(mx.int32); ref.append(np.array(y))
    ref = np.stack(ref, 1)
    seq = np.zeros((B, 128), np.int32)
    for b, nb in enumerate(lens):
        seq[b, nb] = int(np.array(first)[b]); seq[b, nb + 1: nb + 1 + ref.shape[1]] = ref[b]
    seq_d = mx.array(seq)
    for n in (3, 5):
        def drafter(rs):  # noisy oracle: every accept length occurs
            p = rs["q_pos"][:, None] + 1 + mx.arange(n, dtype=mx.int32)[None]
            d = mx.take_along_axis(seq_d, mx.clip(p, 0, 127), axis=1)
            return mx.where(((p * 40503 + 7) % 7) < 2, (d + 1) % 500, d)
        c2 = cache(); lm(inputs=ids, cache=c2)
        dec = FastDecoder(lm, buckets=(32,), ring_extra=7).from_cache(c2)
        fb = []
        toks, rounds = speculative_loop(dec, first, drafter, n, N,
                                        on_round=lambda k, r: fb.append(bool(r.get("fallback") or r.get("plain"))))
        same = np.array_equal(np.array(toks), ref[:, :N])
        ok = same and dec.fallback is not None and any(fb)
        OK &= ok
        print(f"B={B} n={n} speculative_loop: identical={same} fell_back={dec.fallback is not None} "
              f"rounds={rounds} stock_rounds={sum(fb)} -> {'PASS' if ok else 'FAIL'}")
    # direct calls: spec_step crossing the cap, then step_captures, both on the stock path
    c3 = cache(); lm(inputs=ids, cache=c3)
    dec = FastDecoder(lm, buckets=(32,), ring_extra=7).from_cache(c3)
    pos, y, errs = max(lens), first, []
    while dec.fallback is None:
        p = pos - max(lens)  # index into ref of the token after the anchor
        tok = mx.concatenate([y[:, None], mx.array(ref[:, p:p + 3])], axis=1)  # exact drafts
        r = dec.spec_step(tok, captures=(1,)); mx.eval(r["emitted"])
        acc = np.array(r["accepted"]); dec.observe(acc.tolist())
        if r.get("fallback"):
            errs.append(int((np.array(r["bonus"]) != ref[:, p]).sum()))
            y = r["bonus"]; pos += 1
        else:
            k = int(acc.min()); assert (acc == k).all(), acc  # exact drafts: equal accepts for all rows
            errs.append(int((np.array(r["bonus"]) != ref[:, p + k]).sum())); y = r["bonus"]; pos += k + 1
    for _ in range(4):
        p = pos - max(lens)
        lg2, nx, caps = dec.step_captures(y[:, None], (1,)); mx.eval(nx)
        errs.append(int((np.array(nx) != ref[:, p]).sum()) + (caps is not None)); y = nx; pos += 1
    ok = sum(errs) == 0
    OK &= ok
    print(f"B={B} direct spec_step->hand-off->step_captures: mismatches={sum(errs)} cap={dec.cap} -> {'PASS' if ok else 'FAIL'}")
print("SPEC FALLBACK", "PASS" if OK else "FAIL")
sys.exit(0 if OK else 1)
