# Usage: MLX_ENABLE_TF32=0 PYTHONPATH=<repo> python <this> <repo> fp32|bf16 [noquant]  (TF32 is ON by default in MLX 0.32 on M5: FP32 GEMMs with M>=2 lose ~7e-4 rel.)
# FP32 variant of the tiny verify check: FD verify (spec_step) vs stock L-forward, tiny model.
import sys, os
sys.path.insert(0, os.path.join(sys.argv[1], "tests_fast"))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
mx.set_memory_limit(8 * 1024**3)
dt = {"fp32": mx.float32, "bf16": mx.bfloat16}[sys.argv[2]]
if len(sys.argv) > 3 and sys.argv[3] == "noquant":
    tiny.nn.quantize = lambda *a, **k: None
    tiny.replace_scaled_quantized_linears = lambda *a, **k: None
lm = tiny.build(window=1024, rel_extent=1024, n_floor=8, dtype=dt)
rng = np.random.default_rng(0)
for L in (2, 5, 8):
  for T0 in (20, 600, 1500):
    toks = mx.array(rng.integers(1, 500, size=(1, T0 + L)).astype(np.int32))
    with tiny.true_stock():
        c = lm.make_cache(); lm(inputs=toks[:, :T0], cache=c)
        ref = lm(inputs=toks[:, T0:], cache=c).logits.astype(mx.float32); mx.eval(ref); ref = np.array(ref[0])
        c = lm.make_cache(); lm(inputs=toks[:, :T0], cache=c)
        ref1 = []
        for t in range(L):
            l = lm(inputs=toks[:, T0+t:T0+t+1], cache=c).logits[:, -1].astype(mx.float32); mx.eval(l); ref1.append(np.array(l[0]))
        ref1 = np.stack(ref1)
    c = lm.make_cache(); lm(inputs=toks[:, :T0], cache=c)
    d = FastDecoder(lm, ring_extra=L - 1).from_cache(c)
    res = d.spec_step(toks[:, T0:]); lg = res["logits"].astype(mx.float32); mx.eval(lg); lg = np.array(lg[0])
    sc = np.abs(ref).max()
    print(f"{sys.argv[2]} L={L} T0={T0}: |logit|max={sc:.2f} FDverify-vs-stockL max|d|={np.abs(lg-ref).max():.2e} rows_bitwise={(np.abs(lg-ref).max(-1)==0).sum()}/{L}; "
          f"FDverify-vs-stock{L}xL1 {np.abs(lg-ref1).max():.2e}; stockL-vs-stock{L}xL1 {np.abs(ref-ref1).max():.2e}", flush=True)
