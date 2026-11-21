"""Compiled speculative verify (INKLING_FAST_VERIFY) on the tiny model.
Gate: greedy output identical to non-speculative stock greedy decode (FP32),
with a device-side noisy-oracle drafter so every acceptance length occurs.
Also: verify-row logits vs stock decode logits, ms/round vs ms/decode-step."""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models import cache as C
from mlx_vlm.models.inkling2.fast_decode import FastDecoder, speculative_loop

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="float32")
ap.add_argument("--n", default="1,3,7")
ap.add_argument("--batches", default="1,4")
ap.add_argument("--tokens", type=int, default=96)
ap.add_argument("--hidden", type=int, default=512)  # 512: NVFP4 dedup kernel path (K % 256)
a = ap.parse_args()
mx.set_memory_limit(4 * 1024**3)
lm = tiny.build(dtype=getattr(mx, a.dtype))

def prompts(B):
    rng = np.random.default_rng(10 + B)
    lens = [12] if B == 1 else [int(x) for x in rng.integers(5, 21, size=B)]
    T = max(lens); ids = np.zeros((B, T), np.int32)
    for b, n in enumerate(lens): ids[b, T - n:] = rng.integers(1, 500, size=n)
    return mx.array(ids), (None if B == 1 else [T - n for n in lens]), lens

def make_cache(lp):
    if lp is None: return lm.make_cache()
    out = []
    for c in lm.make_cache():
        kv, conv = c.caches
        kvb = C.BatchRotatingKVCache(kv.max_size, lp) if isinstance(kv, C.RotatingKVCache) else C.BatchKVCache(lp)
        conv.left_padding = mx.array(lp); out.append(C.CacheList(kvb, conv))
    return out

ALL_OK = True
for B in [int(x) for x in a.batches.split(",")]:
    ids, lp, lens = prompts(B)
    # reference: stock greedy, non-speculative
    c = make_cache(lp); lg = lm(inputs=ids, cache=c).logits[:, -1]
    first = mx.argmax(lg, -1).astype(mx.int32)
    ref, ref_logits = [], []
    y = first
    for t in range(a.tokens + 24):
        lg = lm(inputs=y[:, None], cache=c).logits[:, -1]; y = mx.argmax(lg, -1).astype(mx.int32)
        mx.eval(y); ref.append(np.array(y)); ref_logits.append(np.array(lg))
    ref = np.stack(ref, 1)  # [B, T] tokens after `first`
    ref_logits = np.stack(ref_logits, 1)
    # absolute-position table for the oracle: seq[b, p] = token at position p
    Tmax = max(lens) + a.tokens + 64
    seq = np.zeros((B, Tmax), np.int32)
    for b, nb in enumerate(lens):
        seq[b, nb] = int(np.array(first)[b]); seq[b, nb + 1: nb + 1 + ref.shape[1]] = ref[b]
    seq_d = mx.array(seq)
    for n in [int(x) for x in a.n.split(",")]:
        def drafter(rs):
            p = rs["q_pos"][:, None] + 1 + mx.arange(n, dtype=mx.int32)[None]  # positions to predict
            d = mx.take_along_axis(seq_d, mx.clip(p, 0, Tmax - 1), axis=1)
            # corrupt ~1/3 of drafts (deterministic in position) -> all accept lengths
            bad = ((p * 40503 + 7) % int(os.environ.get("PERIOD", "7"))) < int(os.environ.get("NBAD", "2"))
            return mx.where(bad, (d + 1) % 500, d)
        c2 = make_cache(lp); lm(inputs=ids, cache=c2)
        dec = FastDecoder(lm, buckets=(64, 128, 256), ring_extra=7).from_cache(c2)
        hist, logs = [], []
        def on_round(k, res):
            hist.append(res["accepted"]); logs.append(res["logits"])
        toks, rounds = speculative_loop(dec, first, drafter, n, a.tokens, on_round=on_round)
        got = np.array(toks)
        same = np.array_equal(got, ref[:, :a.tokens])
        ALL_OK &= bool(same)
        acc = np.array([np.array(h) for h in hist])
        # verify rows fed only correct tokens (rows 0..accepted) vs stock decode logits (review D1)
        vdiff, nrows, P = 0.0, 0, [int(x) for x in lens]
        for k in range(len(hist)):
            lg = np.array(logs[k].astype(mx.float32))
            for b in range(B):
                for j in range(int(acc[k][b]) + 1):
                    t = P[b] + j - lens[b]
                    if t < ref_logits.shape[1]:
                        vdiff = max(vdiff, float(np.abs(lg[b, j] - ref_logits[b, t]).max())); nrows += 1
                P[b] += int(acc[k][b]) + 1
        # D5: plain step() after speculative_loop (final round drained) continues bitwise-close to stock
        sdiff, steps_ok = 0.0, True
        if B == 1:
            pos = dec._known[0]; assert not dec._rounds, "speculative_loop left an unobserved round"
            assert pos == P[0], (pos, P[0])
            for i in range(8):
                t = pos - lens[0]
                if t + 1 >= ref_logits.shape[1]: break
                tok = mx.array([[int(seq[0, pos])]], dtype=mx.int32)
                l, nx = dec.step(tok); mx.eval(l)
                sdiff = max(sdiff, float(np.abs(np.array(l[0]) - ref_logits[0, t]).max()))
                steps_ok &= int(np.array(nx)[0]) == int(ref[0, t]); pos += 1
            ALL_OK &= steps_ok
        print(f"   verify rows (accepted prefix) vs stock decode: rows={nrows} max|d|={vdiff:.2e}; "
              f"step() after loop: argmax ok={steps_ok} max|d|={sdiff:.2e} cap={dec.cap}")
        print(f"{a.dtype} B={B} n={n}: tokens identical to stock greedy: {same}  rounds={rounds} "
              f"mean accepted/round={acc.mean():.2f} accept-hist={np.bincount(acc.reshape(-1), minlength=n+1).tolist()}")
        # timing: compiled spec rounds vs compiled decode steps (pipelined)
        c3 = make_cache(lp); lm(inputs=ids, cache=c3)
        dec = FastDecoder(lm, buckets=(256,), ring_extra=7).from_cache(c3)
        tok = mx.concatenate([first[:, None], drafter(dict(q_pos=dec.state["q_pos"]))], 1)
        for _ in range(3): r = dec.spec_step(tok); mx.eval(r["bonus"]); dec.observe(r["accepted"])
        t0 = time.perf_counter(); R = 20
        for _ in range(R):
            r = dec.spec_step(tok); mx.async_eval(r["bonus"])
        mx.eval(r["bonus"]); ms_round = (time.perf_counter() - t0) / R * 1e3
        c4 = make_cache(lp); lm(inputs=ids, cache=c4)
        d2 = FastDecoder(lm, buckets=(256,)).from_cache(c4); y = first
        for _ in range(3): _, y = d2.step(y[:, None])
        t0 = time.perf_counter()
        for _ in range(R): _, y = d2.step(y[:, None]); mx.async_eval(y)
        mx.eval(y); ms_dec = (time.perf_counter() - t0) / R * 1e3
        print(f"   ms/round (L={n+1}) {ms_round:.2f} vs ms/decode-step {ms_dec:.2f} -> breakeven accepted/round {ms_round/ms_dec - 1:.2f}")
print("TEST_SPEC", "PASS" if ALL_OK else "FAIL")
assert ALL_OK
