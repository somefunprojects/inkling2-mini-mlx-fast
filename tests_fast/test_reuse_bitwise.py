"""reuse_keys / batch_ctrl are meant to be bitwise identical to the plain FastDecoder (bf16).

With MLX_ENABLE_TF32=0 and the INKLING_FAST flags it prints a nonzero difference at B=16 (known limit, see DETAILS.md; it does not cover the failing default-settings case); it has no pass/fail verdict and exits 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
lm = tiny.build()
rng = np.random.default_rng(3)
for B in (1, 4, 8, 16):
    ids = mx.array(rng.integers(1, 500, size=(B, 12)).astype(np.int32))
    res = {}
    for name, kw in (("plain", dict(reuse_keys=False, batch_ctrl=False)),
                     ("reuse", dict(reuse_keys=True, batch_ctrl=False)),
                     ("ctrl3", dict(reuse_keys=False, batch_ctrl=True)),
                     ("both", dict(reuse_keys=True, batch_ctrl=True))):
        c = lm.make_cache(); lg = lm(inputs=ids, cache=c).logits[:, -1]
        dec = FastDecoder(lm, buckets=(64, 128), **kw).from_cache(c)
        y = mx.argmax(lg, -1).astype(mx.int32)
        outs = []
        for t in range(60):
            lg, y = dec.step(y[:, None]); mx.eval(lg); outs.append(np.array(lg))
        res[name] = np.stack(outs)
    print(f"B={B}", " ".join(f"{k}: max|d|={np.abs(v - res['plain']).max():.1e}" for k, v in res.items() if k != "plain"))
