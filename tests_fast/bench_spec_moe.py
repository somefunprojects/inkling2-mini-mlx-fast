"""GPU-time microbench of spec_moe stages at real dims (bf16, fused), L rows, 8 layers.
Each measurement evaluates REPS x 8 layers in one mx.eval (launch overhead amortized)."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("INKLING_SPECFAST", "moe")
import tiny  # noqa
import mlx.core as mx
from mlx_vlm.models.inkling2 import spec_moe as S
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
import spec_moe_util as TB

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=5)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--variants", default="")  # e.g. "GUA_TV=1,GUA_TV=2;DOWN_TV=1"
a = ap.parse_args()
mx.set_memory_limit(60 * 1024**3)

moes = [TB.build_real(10 + i, mx.bfloat16, True) for i in range(a.layers)]
for m in moes:
    m.fast_prepare()
xs = [mx.random.normal((1, a.L, 2304)).astype(mx.bfloat16) for _ in moes]
mx.eval(xs)
lm = tiny.build()
dec = FastDecoder(lm, buckets=(64,))


def timeit(fn, label):
    for _ in range(2):
        mx.eval([fn(m, x) for m, x in zip(moes, xs)])
    t0 = time.perf_counter()
    outs = []
    for _ in range(a.reps):
        outs += [fn(m, x) for m, x in zip(moes, xs)]
    mx.eval(outs)
    ms = (time.perf_counter() - t0) / a.reps / len(moes) * 1e3
    print(f"  {label:40s} {ms*1000:8.1f} us/layer")
    return ms


def old(m, x):
    sf = dec.sf
    dec.sf = sf - {"moe"}
    try:
        return dec._moe(m, x)
    finally:
        dec.sf = sf


def decode_rows(m, x):
    return mx.concatenate([m(x[:, i:i + 1]) for i in range(x.shape[1])], axis=1)


# precomputed stage inputs
st = [S.moe_verify(m, x, m.fast_prepare(), stages=True) for m, x in zip(moes, xs)]
mx.eval([s[:5] for s in st])
stage_in = {id(m): s for m, s in zip(moes, st)}


def only_router(m, x):
    lg = S.router_rows(x.reshape(-1, 2304), m._fast_router_weight())
    return fk.route_topk(lg, m.e_score_correction_bias, m.top_k, m.route_scale)[0]


def only_down(m, x):
    s = stage_in[id(m)]
    return S.moe_verify(m, x, m.fast_prepare(), stages="act_in", stages_act=s[3])


print(f"L={a.L} layers={a.layers} reps={a.reps}")
from mlx_vlm.models.inkling2 import mega_kernels as mk


def stock_gu(m, x):
    s = stage_in[id(m)]
    f = m.fast_prepare()
    idx = s[1].reshape(1, -1)
    xe = mx.repeat(x.reshape(-1, 1, 1, 2304), m.top_k, axis=0).reshape(1, -1, 1, 1, 2304)[0]
    return mx.gather_qmm(xe, f["weight"], f["scales"], rhs_indices=idx.reshape(-1, 1), transpose=True,
                         group_size=16, bits=4, mode="nvfp4", global_scale=f["factor"], sorted_indices=False)


def stock_down(m, x):
    s = stage_in[id(m)]
    dn = m.switch_mlp.down_proj
    idx = s[1].reshape(-1, 1)
    fac = mk._down_factor(dn)
    if fac is None:
        fac = mx.broadcast_to((2688.0 * dn.global_scale.astype(mx.float32)).reshape(-1), (128,))
    return mx.gather_qmm(s[3].reshape(-1, 1, 1, 1152), dn.weight, dn.scales, rhs_indices=idx, transpose=True,
                         group_size=16, bits=4, mode="nvfp4", global_scale=fac, sorted_indices=False)


timeit(stock_gu, "stock gather_qmm gate+up, 40 routes")
timeit(stock_down, "stock gather_qmm down, 40 routes")
timeit(old, "old verify (dedup grouped)")
timeit(decode_rows, "decode path per row")
for var in [""] + [v for v in a.variants.split(";") if v]:
    S._perf_hdr = ""
    for kv in [t for t in var.split(",") if t]:
        k, v = kv.split("=")
        if k.startswith("SPEC_"):
            S._perf_hdr += "#define %s %d\n" % (k, int(v))
        else:
            setattr(S, k, float(v) if k == "ZFRAC" else int(v))
    S._kern.clear()
    print(f" variant [{var}] GUA rows/sgs/tv={S.GUA_ROWS}/{S.GUA_SGS}/{S.GUA_TV} DOWN={S.DOWN_ROWS}/{S.DOWN_SGS}/{S.DOWN_TV}")
    full = timeit(lambda m, x: S.moe_verify(m, x, m.fast_prepare()), "spec_moe full")
    r = timeit(only_router, "router+topk")
    d = timeit(only_down, "router+topk+down (act given)")
    print(f"  => gua ~{(full - d)*1000:.1f} us, down ~{(d - r)*1000:.1f} us")
