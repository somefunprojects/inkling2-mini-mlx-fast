"""generate_fast greedy tokens vs stock greedy loop on the tiny model (bf16)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, numpy as np, mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import generate_fast
lm = tiny.build(dtype=getattr(mx, os.environ.get("DT", "bfloat16")))
class M: language_model = lm
ids = [int(x) for x in np.random.default_rng(1).integers(0, 500, size=12)]
N = 80
cache = lm.make_cache(); lg = lm(inputs=mx.array(ids)[None], cache=cache).logits[:, -1]
ref = []
for i in range(N):
    t = int(mx.argmax(lg, -1).item()); ref.append(t)
    lg = lm(inputs=mx.array([[t]]), cache=cache).logits[:, -1]
seen = []
got = generate_fast(M, None, ids, max_tokens=N, prefill_step=5,
                    logits_hook=lambda i, l: seen.append(i), decoder_kwargs=dict(buckets=(16, 32, 64, 128)))
print("tokens identical:", got == ref, "n", len(got), "first diff", next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), -1), "hook steps", len(seen))
