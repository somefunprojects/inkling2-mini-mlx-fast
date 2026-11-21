"""Padded-row, row-isolation and ablation probe for INKLING_BATCHFAST; run by test_batch_fd.py.
GPU, tiny random model only (window 16, rel_extent 32). Needs the default INKLING_FAST stack,
INKLING_FAST_DECODE=1 and MLX_ENABLE_TF32=0.

T0  refusal: unequal row lengths must raise Unsupported unless INKLING_FAST_DECODE_PADDED=1.
T1  FP32, left-padded rows of 8 different lengths (5..61), 72 forced steps: rows cross the sliding window (16),
    rel_extent (32) and buckets 64 -> 128 -> 256 at DIFFERENT steps. BATCHFAST=all vs off, and every batch row vs
    the same row decoded alone at B=1 (FD exact). Repeated with bk.BSDPA_BLOCK=8 so the NB>1 split + combine
    kernels run (the default block 256 gives NB=1 on the tiny model, so the stock test never runs them).
T2  row isolation, equal lengths, FP32 + BF16, off and all: (a) permuting the batch rows permutes the logits
    bitwise; (b) changing ONE row's forced tokens from step 6 leaves every other row bitwise unchanged.
T3  BF16 per-token ablation at B=4/8/16, 3 seeds: max|dlogit| and argmax flips of BATCHFAST=<tok> vs off, plus
    each side's distance to the FP32 model with the same seed (is "all" further from FP32 than "off"?).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.inkling2 import batch_kernels as bk, spec_moe  # noqa: E402
from mlx_vlm.models.inkling2.fast_decode import FastDecoder, Unsupported  # noqa: E402
from mlx_vlm.generate.ar import _left_pad_prompts, _make_cache  # noqa: E402

mx.set_memory_limit(8 * 1024**3)
FAIL = []
HITS = {}


def _wrap(mod, name, key):
    f = getattr(mod, name)

    def w(*a, **k):
        HITS[key] = HITS.get(key, 0) + 1
        return f(*a, **k)
    setattr(mod, name, w)


_wrap(bk, "sdpa_brows", "sdpa")
_wrap(bk, "tail_brows", "tail")
_wrap(bk, "head_rows", "head")
_wrap(spec_moe, "moe_verify", "moe")


def prefill(lm, rows):
    lens = [len(r) for r in rows]
    if len(set(lens)) == 1:
        x, cache = mx.array(rows), lm.make_cache()
    else:
        x, cache = _left_pad_prompts(rows), _make_cache(lm, [max(lens) - n for n in lens])
    mx.eval(lm(inputs=x, cache=cache).logits)
    return cache


def run_fd(lm, rows, forced, bf, block=256, padded=False):
    os.environ["INKLING_BATCHFAST"] = bf
    os.environ["INKLING_FAST_DECODE_PADDED"] = "1" if padded else "0"
    bk.BSDPA_BLOCK = block
    HITS.clear()
    dec = FastDecoder(lm, buckets=(64, 128, 256)).from_cache(prefill(lm, rows))
    out = []
    for s in range(forced.shape[0]):
        lg, _ = dec.step(forced[s][:, None])
        out.append(lg.astype(mx.float32))
    out = mx.stack(out)
    mx.eval(out)
    os.environ["INKLING_BATCHFAST"] = ""
    return out, dict(HITS)


def mad(a, b):
    return float(mx.abs(a - b).max().item())


def flips(a, b):
    return int((a.argmax(-1) != b.argmax(-1)).sum().item())


def note(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


# ---------------------------------------------------------------- T0 refusal
lm = tiny.build(dtype=mx.float32)
rows = [list(range(1, 1 + n)) for n in (20, 13)]
for padded_env in (None, "0"):
    if padded_env is None:
        os.environ.pop("INKLING_FAST_DECODE_PADDED", None)
    else:
        os.environ["INKLING_FAST_DECODE_PADDED"] = padded_env
    try:
        FastDecoder(lm, buckets=(64, 128)).from_cache(prefill(lm, rows))
        note(False, f"T0 unequal lengths accepted with PADDED={padded_env}")
    except Unsupported as e:
        note(True, f"T0 unequal lengths refused (PADDED={padded_env}): {e}")
os.environ["INKLING_FAST_DECODE_APPROX"] = "1"
try:
    FastDecoder(lm, buckets=(64, 128), exact=False).from_cache(prefill(lm, rows))
    print("info T0 exact=False (INKLING_FAST_DECODE_APPROX=1): unequal lengths ACCEPTED without PADDED", flush=True)
except Unsupported as e:
    print(f"info T0 exact=False: refused too: {e}", flush=True)
os.environ.pop("INKLING_FAST_DECODE_APPROX", None)

# ---------------------------------------------------------------- T1 padded, crossings, NB>1
mx.random.seed(1)
lens = [40, 23, 57, 9, 33, 15, 61, 5]
B, S = len(lens), 72
rows = [mx.random.randint(1, 200, (n,)).tolist() for n in lens]
forced = mx.random.randint(1, 200, (S, B)).astype(mx.int32)
alone = []
for b in range(B):
    os.environ["INKLING_BATCHFAST"] = ""
    dec = FastDecoder(lm, buckets=(64, 128, 256)).from_cache(prefill(lm, [rows[b]]))
    alone.append(mx.stack([dec.step(forced[s][b:b + 1, None])[0][0].astype(mx.float32) for s in range(S)]))
alone = mx.stack(alone, axis=1)  # [S, B, V]
mx.eval(alone)
for block in (256, 8):
    off, h0 = run_fd(lm, rows, forced, "", block, padded=True)
    al, h1 = run_fd(lm, rows, forced, "all", block, padded=True)
    d = mad(off, al)
    note(d <= 1e-4 and h1.get("sdpa", 0) > 0,
         f"T1 FP32 padded lens={lens} {S} steps block={block}: all vs off max|d| {d:.3e} flips {flips(off, al)} hits {h1}")
    per = [mad(al[:, b], alone[:, b]) for b in range(B)]
    per0 = [mad(off[:, b], alone[:, b]) for b in range(B)]
    note(max(per) <= 1e-4, f"T1 block={block} batch(all) row vs B=1 alone max|d| per row {[f'{x:.1e}' for x in per]}")
    print(f"info T1 block={block} batch(off) row vs B=1 alone {[f'{x:.1e}' for x in per0]}", flush=True)
    # error over steps for the worst row: does it grow?
    wb = max(range(B), key=lambda b: per[b])
    seg = [mad(al[s0:s0 + 12, wb], alone[s0:s0 + 12, wb]) for s0 in range(0, S, 12)]
    print(f"info T1 block={block} worst row {wb} (len {lens[wb]}) max|d| per 12-step segment {[f'{x:.1e}' for x in seg]}",
          flush=True)
# equal-length rows, NB>1 (FP32), long enough to cross 64 and 128
rows_eq = mx.random.randint(1, 200, (B, 40)).tolist()
off, _ = run_fd(lm, rows_eq, forced, "", 8)
al, h = run_fd(lm, rows_eq, forced, "all", 8)
d = mad(off, al)
note(d <= 1e-4, f"T1 FP32 equal len 40, {S} steps (buckets 64->128), block=8 (NB>1): all vs off max|d| {d:.3e} hits {h}")
del lm

# ---------------------------------------------------------------- T2 row isolation
for dtype in (mx.float32, mx.bfloat16):
    lm = tiny.build(dtype=dtype)
    mx.random.seed(2)
    B, S = 8, 16
    rows_eq = mx.random.randint(1, 200, (B, 40)).tolist()
    forced = mx.random.randint(1, 200, (S, B)).astype(mx.int32)
    perm = [3, 7, 0, 5, 1, 6, 2, 4]
    f2 = mx.array(forced)
    f2[6:, 5] = mx.random.randint(1, 200, (S - 6,)).astype(mx.int32)
    for bf in ("", "all"):
        base, _ = run_fd(lm, rows_eq, forced, bf)
        pr, _ = run_fd(lm, [rows_eq[i] for i in perm], forced[:, perm], bf)
        eqp = bool(mx.array_equal(pr, base[:, perm]).item())
        ch, _ = run_fd(lm, rows_eq, f2, bf)
        others = [i for i in range(B) if i != 5]
        eqo = bool(mx.array_equal(ch[:, others], base[:, others]).item())
        tag = f"T2 {str(dtype)[-8:]} BATCHFAST={bf or 'off'}"
        print(f"{'ok  ' if eqp else 'NOTE'} {tag} permute rows -> logits permuted bitwise: {eqp} "
              f"(max|d| {mad(pr, base[:, perm]):.2e})", flush=True)
        print(f"{'ok  ' if eqo else 'NOTE'} {tag} change row 5 only -> other rows bitwise unchanged: {eqo} "
              f"(max|d| {mad(ch[:, others], base[:, others]):.2e})", flush=True)
        if bf == "all" and not (eqp and eqo):
            FAIL.append(tag + " row isolation")
    del lm

# ---------------------------------------------------------------- T3 bf16 per-token ablation
toks = ["moe", "head", "fuse", "sdpa", "post", "pre", "dattn", "all"]
for seed in (0, 1, 2):
    lm16 = tiny.build(seed=seed, dtype=mx.bfloat16)
    lm32 = tiny.build(seed=seed, dtype=mx.float32)
    for B in (4, 8, 16):
        mx.random.seed(100 + seed)
        rows_eq = mx.random.randint(0, 200, (B, 40)).tolist()
        forced = mx.random.randint(0, 200, (12, B)).astype(mx.int32)
        ref32, _ = run_fd(lm32, rows_eq, forced, "")
        off, _ = run_fd(lm16, rows_eq, forced, "")
        msg = [f"off->fp32 {mad(off, ref32):.2e}"]
        for t in toks:
            got, _ = run_fd(lm16, rows_eq, forced, t)
            msg.append(f"{t} {mad(off, got):.2e}/{flips(off, got)}")
            if t == "all":
                msg.append(f"all->fp32 {mad(got, ref32):.2e}")
        print(f"T3 bf16 seed={seed} B={B:2d} max|d| vs off / flips: " + "  ".join(msg), flush=True)
    del lm16, lm32

print("BATCH_FD_PROBE", "PASS" if not FAIL else f"FAIL {len(FAIL)}: " + " | ".join(FAIL), flush=True)
