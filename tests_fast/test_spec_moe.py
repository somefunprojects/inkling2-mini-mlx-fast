"""spec_moe.moe_verify (INKLING_SPECFAST=moe) vs the L=1 decode-path MoE run per row.

Run with the default INKLING_FAST stack (route+fusemoe+glue on). Cases:
  tiny  : tests_fast/tiny.py MoE layers (K=256, NH=128, E=8, k=2), bf16 + fp32, unfused and
          fused (gate/up scales forced equal), B=1 L=2..8 (+B=2 L=4), also via FastDecoder._moe.
  real  : D=2304, NH=1152, E=128, k=8 random NVFP4 experts (per-expert scale_2), fused + unfused,
          bf16 + fp32, L=2..8, router ties (duplicate router rows) and duplicate tokens.
  perf  : (--perf) real dims bf16 fused, L=5, ms per layer: moe_verify vs FastDecoder._moe stock
          verify path (dedup grouped_qmv) over 8 distinct layers.
Prints max|d| per case; TEST_SPEC_MOE PASS iff every case is bitwise.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("INKLING_SPECFAST", "moe")
import tiny  # noqa: E402  (sets up the package path)
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.inkling2 import spec_moe  # noqa: E402
from mlx_vlm.models.inkling2.fast_decode import FastDecoder  # noqa: E402
from mlx_vlm.models.inkling2.language import SparseMoE  # noqa: E402
from mlx_vlm.quantization.nvfp4 import replace_scaled_quantized_linears  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--perf", action="store_true")
ap.add_argument("--skip-real", action="store_true")
a = ap.parse_args()
mx.set_memory_limit(60 * 1024**3)

ALL_OK = True


def ref_rows(moe, x):
    # the decode path, one (b, i) row at a time (B>1 decode batches rows into one matmul)
    B, L, _ = x.shape
    return mx.concatenate([mx.concatenate([moe(x[b:b + 1, i:i + 1]) for i in range(L)], axis=1)
                           for b in range(B)], axis=0)


def check(tag, moe, x, fn=None):
    global ALL_OK
    ref = ref_rows(moe, x)
    got = spec_moe.moe_verify(moe, x, moe.fast_prepare()) if fn is None else fn(moe, x)
    if got is None:
        print(f"{tag}: UNSUPPORTED (None)")
        ALL_OK = False
        return
    mx.eval(ref, got)
    eq = bool(mx.array_equal(ref, got).item())
    d = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max().item())
    ALL_OK &= eq
    print(f"{tag}: bitwise={eq} max|d|={d:.3e}")


# ---------------------------------------------------------------- tiny
for dtype in (mx.bfloat16, mx.float32):
    for equal_scales in (False, True):
        lm = tiny.build(dtype=dtype)
        if equal_scales:
            for layer in lm.model.layers:
                sm = layer.mlp.switch_mlp
                if hasattr(sm.gate_proj, "global_scale"):
                    sm.up_proj[sm.up_proj._scale_name] = sm.gate_proj.global_scale
        dec = FastDecoder(lm, buckets=(64,))
        assert "moe" in dec.sf
        D = lm.model.layers[0].mlp.gate_weight.shape[1]
        for li, layer in enumerate(lm.model.layers):
            moe = layer.mlp
            fused = moe.fast_prepare()
            sup = spec_moe.supported(moe, mx.zeros((1, 2, D), dtype), fused)
            if not sup:
                print(f"tiny {dtype} eq={equal_scales} layer {li}: not supported (expected for mxfp8)")
                continue
            for B, L in [(1, L) for L in range(2, 9)] + [(2, 4)]:
                x = (mx.random.normal((B, L, D)) * 0.5).astype(dtype)
                check(f"tiny {str(dtype)[-8:]} fused={fused is not None} layer {li} B={B} L={L}", moe, x)
            x = (mx.random.normal((1, 5, D)) * 0.5).astype(dtype)
            check(f"tiny {str(dtype)[-8:]} fused={fused is not None} layer {li} via dec._moe L=5", moe, x,
                  fn=dec._moe)
        del lm, dec


# ---------------------------------------------------------------- real dims
from spec_moe_util import build_real  # noqa: E402


if not a.skip_real:
    for dims in ((2304, 1152, 128, 8), (1024, 512, 16, 4)):  # 2nd: K%512==0 (qmv_fast)
        for dtype in (mx.bfloat16, mx.float32):
            for fused_scales in (True, False):
                moe = build_real(1, dtype, fused_scales, dims)
                fused = moe.fast_prepare()
                assert (fused is not None) == fused_scales, "fusion expectation"
                for L in range(2, 9):
                    x = (mx.random.normal((1, L, dims[0]))).astype(dtype)
                    if L >= 4:
                        x[:, 3] = x[:, 1]  # duplicate token rows (same experts)
                    check(f"real{dims} {str(dtype)[-8:]} fused={fused is not None} L={L}", moe, x)
                del moe

# ---------------------------------------------------------------- perf
if a.perf:
    lm = tiny.build()
    dec = FastDecoder(lm, buckets=(64,))
    moes = [build_real(10 + i, mx.bfloat16, True) for i in range(8)]
    for m in moes:
        m.fast_prepare()
    xs = [mx.random.normal((1, 5, 2304)).astype(mx.bfloat16) for _ in range(8)]
    mx.eval(xs)

    def run(fn, reps=30):
        for _ in range(3):
            mx.eval([fn(m, x) for m, x in zip(moes, xs)])
        t0 = time.perf_counter()
        for _ in range(reps):
            mx.eval([fn(m, x) for m, x in zip(moes, xs)])
        return (time.perf_counter() - t0) / reps / len(moes) * 1e3

    def old(m, x):
        sf = dec.sf
        dec.sf = sf - {"moe"}
        try:
            return dec._moe(m, x)
        finally:
            dec.sf = sf

    print(f"PERF L=5 ms/layer (eager, 8 distinct layers): spec_moe {run(lambda m, x: spec_moe.moe_verify(m, x, m.fast_prepare())):.3f}"
          f"  old verify {run(old):.3f}  decode-per-row x5 {run(ref_rows):.3f}")

print("TEST_SPEC_MOE", "PASS" if ALL_OK else "FAIL")
