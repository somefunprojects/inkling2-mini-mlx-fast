"""INKLING_BATCHFAST kernels at B>1, L=1 (FastDecoder decode step).

  (1) spec_moe.moe_verify at (B, 1), B = 2,4,8,16: bitwise per row vs the L=1 decode-path MoE run one row at a time
      (tiny bf16 + fp32; real dims bf16 + fp32, fused + unfused scales; duplicate rows).
  (2) head: fk.lm_head_gemv at M = 2..16 rows vs the same kernel at M = 1 per row (bitwise expected) and vs the FP32
      GEMM (ordering-only; max rel diff printed).
  (3) e2e tiny FP32 (MLX_ENABLE_TF32=0 recommended): FastDecoder B=4/8/16 equal-length rows, 12 forced steps,
      BATCHFAST=all vs BATCHFAST unset: max|dlogit| (FP32-exact means ordering-level, <= 1e-4) and argmax equality.
Prints BATCH_FD PASS iff (1) and (2 vs M=1) are bitwise and (3) is within tolerance."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.inkling2 import spec_moe, fast_kernels as fk  # noqa: E402
from mlx_vlm.models.inkling2.fast_decode import FastDecoder  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--skip-real", action="store_true"); a = ap.parse_args()
mx.set_memory_limit(60 * 1024**3)
OK = True


def ref_rows(moe, x):
    B, L, _ = x.shape
    return mx.concatenate([mx.concatenate([moe(x[b:b + 1, i:i + 1]) for i in range(L)], axis=1)
                           for b in range(B)], axis=0)


def check(tag, ref, got, tol=None):
    global OK
    mx.eval(ref, got)
    eq = bool(mx.array_equal(ref, got).item())
    d = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max().item())
    ok = eq if tol is None else d <= tol
    OK &= ok
    print(f"{tag}: bitwise={eq} max|d|={d:.3e} {'ok' if ok else 'FAIL'}", flush=True)


# (1) MoE
for dtype in (mx.bfloat16, mx.float32):
    lm = tiny.build(dtype=dtype)
    D = lm.model.layers[0].mlp.gate_weight.shape[1]
    for li, layer in enumerate(lm.model.layers):
        moe = layer.mlp
        if not spec_moe.supported(moe, mx.zeros((2, 1, D), dtype), moe.fast_prepare()):
            continue
        for B in (2, 4, 8, 16):
            x = (mx.random.normal((B, 1, D)) * 0.5).astype(dtype)
            x[1] = x[0]  # duplicate row
            check(f"moe tiny {str(dtype)[-8:]} layer {li} B={B}", ref_rows(moe, x),
                  spec_moe.moe_verify(moe, x, moe.fast_prepare()))
    del lm
if not a.skip_real:
    from spec_moe_util import build_real
    for dtype in (mx.bfloat16, mx.float32):
        for fs in (True, False):
            moe = build_real(1, dtype, fs, (2304, 1152, 128, 8)); moe.fast_prepare()
            for B in (2, 4, 8, 16):
                x = mx.random.normal((B, 1, 2304)).astype(dtype)
                x[2] = x[0]
                check(f"moe real {str(dtype)[-8:]} fused={fs} B={B}", ref_rows(moe, x),
                      spec_moe.moe_verify(moe, x, moe.fast_prepare()))
            del moe

# (2) head
w = (mx.random.normal((4099, 2304)) * 0.02).astype(mx.bfloat16)
for M in (2, 4, 8, 15, 16):
    h = mx.random.normal((M, 2304)).astype(mx.float32)
    got = fk.lm_head_gemv(h, w, 4097, 30.0)
    one = mx.concatenate([fk.lm_head_gemv(h[i:i + 1], w, 4097, 30.0) for i in range(M)], axis=0)
    check(f"head M={M} vs M=1 rows", one, got)
    ref = h @ w[:4097].astype(mx.float32).T
    ref = 30.0 * mx.tanh(ref / 30.0)
    mx.eval(ref)
    rel = float((mx.abs(ref - got).max() / mx.abs(ref).max()).item())
    print(f"head M={M} vs FP32 GEMM+softcap: max|d|/max|ref| {rel:.2e}", flush=True)
    OK &= rel < 1e-5

# (2b) batch_kernels.head_rows: bitwise vs fk.lm_head_gemv (same per-lane order), plus eager timing at real V
from mlx_vlm.models.inkling2 import batch_kernels as bk
import time
for M in (2, 4, 8, 16):
    h = mx.random.normal((M, 2304)).astype(mx.float32)
    check(f"head_rows M={M} vs lm_head_gemv", fk.lm_head_gemv(h, w, 4097, 30.0), bk.head_rows(h, w, 4097, 30.0))
wr = (mx.random.normal((201024, 2304)) * 0.02).astype(mx.bfloat16); mx.eval(wr)
def tm(fn, n=20):
    for _ in range(3): mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1e3
for M in (1, 2, 4, 8, 16):
    h = mx.random.normal((M, 2304)).astype(mx.float32); mx.eval(h)
    t_old = tm(lambda: fk.lm_head_gemv(h, wr, 201024, 30.0))
    msg = f"HEADPERF M={M} lm_head_gemv {t_old:.3f} ms"
    if M >= 2:
        for rows in (1, 2, 4, 8):
            msg += f" | head_rows R{rows} {tm(lambda: bk.head_rows(h, wr, 201024, 30.0, rows=rows)):.3f}"
    print(msg + "  (byte floor 0.84 ms)", flush=True)
del wr

# (3) e2e tiny, FP32 (exactness) and BF16 (engagement of the bf16-only head path)
HITS = {"moe": 0, "head": 0, "sdpa": 0, "tail": 0}
_mv, _hg, _hr = spec_moe.moe_verify, fk.lm_head_gemv, bk.head_rows
def _cmv(*a, **k):
    HITS["moe"] += 1; return _mv(*a, **k)
def _chg(*a, **k):
    HITS["head"] += 1; return _hg(*a, **k)
def _chr(*a, **k):
    HITS["head"] += 1; return _hr(*a, **k)
spec_moe.moe_verify, fk.lm_head_gemv, bk.head_rows = _cmv, _chg, _chr
_sb, _tb = bk.sdpa_brows, bk.tail_brows
def _csb(*a, **k):
    HITS["sdpa"] += 1; return _sb(*a, **k)
def _ctb(*a, **k):
    HITS["tail"] += 1; return _tb(*a, **k)
bk.sdpa_brows, bk.tail_brows = _csb, _ctb
for dtype, tol in ((mx.float32, 1e-4), (mx.bfloat16, None)):
    lm = tiny.build(dtype=dtype)
    for B in (4, 8, 16):
        rows = mx.random.randint(0, 200, (B, 40)).astype(mx.int32)
        forced = mx.random.randint(0, 200, (12, B)).astype(mx.int32)
        outs = {}
        for bf in ("", "all"):
            os.environ["INKLING_BATCHFAST"] = bf
            HITS.update(moe=0, head=0, sdpa=0, tail=0)
            cache = lm.make_cache()
            lm(inputs=rows, cache=cache)
            dec = FastDecoder(lm, buckets=(64, 128)).from_cache(cache)
            lg = [dec.step(forced[s][:, None])[0] for s in range(12)]
            outs[bf] = mx.stack(lg).astype(mx.float32)
            mx.eval(outs[bf])
            print(f"e2e tiny {str(dtype)[-8:]} B={B} BATCHFAST={bf or 'off'} trace hits {dict(HITS)}", flush=True)
        os.environ["INKLING_BATCHFAST"] = ""
        if tol is not None:
            check(f"e2e tiny {str(dtype)[-8:]} B={B} BATCHFAST=all vs off", outs[""], outs["all"], tol=tol)
        else:
            d = float(mx.abs(outs[""] - outs["all"]).max().item())
            print(f"e2e tiny {str(dtype)[-8:]} B={B} BATCHFAST=all vs off: max|d| {d:.3e} (bf16, info only)", flush=True)
        same = bool(mx.array_equal(outs[""].argmax(-1), outs["all"].argmax(-1)).item())
        print(f"e2e tiny {str(dtype)[-8:]} B={B} argmax equal: {same}", flush=True)
    del lm
print("BATCH_FD", "PASS" if OK else "FAIL")
