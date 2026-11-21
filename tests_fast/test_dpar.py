"""mega_dpar (parallel depth-attention scores/softmax) vs the serial kernels, real D, bitwise.
attnres / mega_post / mega_pre for every key count 1..31, boundary/norm variants; attnres also vs
the stock op chain. Toggles fk.FLAGS and clears the kernel caches between variants."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from types import SimpleNamespace
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2 import mega_kernels as mk
from mlx_vlm.models.inkling2.fast_decode import _attn_res_keys_c

D = int(os.environ.get("D", "2304"))
SEEDS = int(os.environ.get("SEEDS", "3"))
res = {}


def cmp(name, ref, got):
    ok = mx.array_equal(ref, got).item()
    t = res.setdefault(name, [0, 0]); t[1] += 1; t[0] += ok
    if not ok and t[1] - t[0] <= 3:
        d = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))
        print(f"  {name}: {int((d > 0).sum().item())} differ, max {d.max().item():.3e}", flush=True)


def variant(on):
    fk.FLAGS = (fk.FLAGS | {"mega_dpar"}) if on else (fk.FLAGS - {"mega_dpar"})
    fk._attnres_kernels.clear()
    for k in [k for k in mk._kern if k.startswith(("post", "pre"))]:
        del mk._kern[k]


W = 4
mod = SimpleNamespace(kernel_size=W, conv=SimpleNamespace(weight=None))
for seed in range(SEEDS):
    mx.random.seed(seed)
    mod.conv.weight = (mx.random.normal((D, 1, W)) * 0.3).astype(mx.bfloat16)
    norm = SimpleNamespace(weight=(1 + 0.1 * mx.random.normal((D,))).astype(mx.bfloat16), eps=1e-6)
    for nk in range(1, 13):  # real model: <= 6 keys; > ~12 overflows Metal's 31 buffer slots
        keys = [fk._block_key(mx.random.normal((1, 1, D)) * (1 + (i % 5))) for i in range(nk)]
        q = mx.random.normal((D,)).astype(mx.bfloat16) * (0.5 + seed)
        part = (mx.random.normal((1, 1, D)) * 2).astype(mx.bfloat16)
        state = mx.random.normal((1, 3, D))
        attn = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
        x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
        K = 8
        routes = mx.random.normal((1, 1, K, D)).astype(mx.bfloat16)
        rw = mx.random.uniform(shape=(1, 1, K))
        resid = mx.random.normal((1, 1, D))
        mx.eval(keys, q, part, state, attn, x, routes, rw, resid)
        outs = {}
        for on in (False, True):
            variant(on)
            o = [fk.attn_res(keys, part, q, raw_partial=True) if nk <= 8 else mx.zeros((1,))]
            for bnd in (False, True):
                if nk + (1 if bnd else 0) + 1 > 32:
                    continue
                for nm in (None, norm):
                    r = mk.post_block(mod, state, attn, x, bnd, keys, q, nm)
                    o += [r[k] for k in sorted(r)]
            for nm in (None, norm):
                r = mk.pre_block(mod, state, routes, resid, rw, keys, q, nm)
                o += [r[k] for k in sorted(r)]
            mx.eval(o)
            outs[on] = o
        if nk <= 8:
            cmp("attnres dpar vs stock chain", _attn_res_keys_c(keys, part, q, True), outs[True][0])
        for i, (a, b) in enumerate(zip(outs[False], outs[True])):
            cmp(f"dpar vs serial out{i}", a, b)
        if nk <= 8:
            cmp("serial attnres vs stock chain (sanity)", _attn_res_keys_c(keys, part, q, True), outs[False][0])

# ---- mega_oproj2 vs mega_oproj vs stock chain (gate kernel + gemv), real dims ----
N_, K_ = 2304, 3072
for seed in range(SEEDS * 3):
    mx.random.seed(100 + seed)
    lin = SimpleNamespace(weight=(mx.random.normal((N_, K_)) * 0.02).astype(mx.bfloat16))
    att = mx.random.normal((1, 1, K_)).astype(mx.bfloat16)
    gate = (mx.random.normal((1, 1, K_)) * 3).astype(mx.bfloat16)
    mx.eval(lin.weight, att, gate)
    outs = {}
    for on in (False, True):
        fk.FLAGS = (fk.FLAGS | {"mega_oproj2"}) if on else (fk.FLAGS - {"mega_oproj2"})
        outs[on] = mk.gate_oproj(lin, att, gate)
        mx.eval(outs[on])
    cmp("oproj2 vs oproj", outs[False], outs[True])
bad = 0
for k, (ok, n) in sorted(res.items()):
    print(f"{k:45s} {ok}/{n}")
    bad += n - ok
print("DPAR BITWISE PASS" if bad == 0 else f"DPAR FAIL {bad}")
sys.exit(1 if bad else 0)
