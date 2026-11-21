"""FastDecoder with xpass depth-logit reuse vs without: compiled, bitwise?

Model: test_e2e.build() (hidden 2304, 3 loop passes). Usage: python test_fd_xpass.py
"""
import json, os, sys, time
os.environ.setdefault("E2E_CFG", json.dumps({"intermediate_size": 1152, "loop_config": {
    "core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 64}}))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import test_e2e
from _load import load
import mlx.core as mx

fd = load("mlx_vlm.models.inkling2.fast_decode")
lm, cfg = test_e2e.build()
rng = np.random.default_rng(3)
ok = True
for B in (1, 4):
    ids = mx.array(rng.integers(1, 8000, size=(B, 40)).astype(np.int32))
    res, times = {}, {}
    for xp in (False, True):
        c = lm.make_cache()
        lg = lm(inputs=ids, cache=c).logits[:, -1]
        dec = fd.FastDecoder(lm, buckets=(128, 256)).from_cache(c)
        dec.xpass = xp
        y = mx.argmax(lg, -1).astype(mx.int32)
        outs = []
        for t in range(40):
            if t == 5:
                mx.eval(y); t0 = time.perf_counter()
            lg, y = dec.step(y[:, None]); mx.eval(lg); outs.append(np.array(lg))
        times[xp] = (time.perf_counter() - t0) / 35 * 1e3
        res[xp] = np.stack(outs)
    eq = np.array_equal(res[False], res[True])
    ok &= eq
    print(f"B={B} FastDecoder xpass vs keys-only: bitwise={eq} max|d|={np.abs(res[False] - res[True]).max():.2g} "
          f"step ms keys-only {times[False]:.2f} / xpass {times[True]:.2f}")
assert ok
print("PASS fd xpass")
