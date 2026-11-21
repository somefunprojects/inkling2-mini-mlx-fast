# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""Batched-MoE microbench: NVFP4 MoE layer at L=1, B rows, real dims (2304/1152/128/8), bf16, fused gate/up scales.
Routing realism: rows x_i = sqrt(rho)*c + sqrt(1-rho)*n_i with rho calibrated per B so the mean number of distinct
experts per layer U matches the in-situ measurement (24.1 / 40.1 / 61.9 at B = 4 / 8 / 16).
Each timing evaluates REPS x LAYERS layer calls in one mx.eval (GPU-bound; launch overhead amortized). us/layer."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa
import mlx.core as mx
from mlx_vlm.models.inkling2 import spec_moe as S
from mlx_vlm.models.inkling2 import fast_kernels as fk
import spec_moe_util as TB

ap = argparse.ArgumentParser()
ap.add_argument("--batches", default="4,8,16")
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--stock", type=int, default=1)
ap.add_argument("--stages", type=int, default=1)
ap.add_argument("--chain", type=int, default=0)  # 1: layer i+1 input = x_{i+1} + 0*out_i (serial, like in situ)
ap.add_argument("--variants", default="")  # "A=1,B=2;C=3": module attrs (SPEC_* -> #define)
a = ap.parse_args()
mx.set_memory_limit(60 * 1024**3)
TARGET_U = {1: 8.0, 2: 15.0, 4: 24.1, 8: 40.1, 16: 61.9}
D = 2304
moes = [TB.build_real(10 + i, mx.bfloat16, True) for i in range(a.layers)]
for m in moes:
    m.fast_prepare()


def mean_u(xs):
    us = []
    for m, x in zip(moes, xs):
        lg = S.router_rows(x.reshape(-1, D), m._fast_router_weight())
        idx = fk.route_topk(lg, m.e_score_correction_bias, m.top_k, m.route_scale)[0]
        us.append(len(set(idx.reshape(-1).tolist())))
    return sum(us) / len(us), us


def make_x(B, rho, seed):
    mx.random.seed(seed)
    out = []
    for _ in moes:
        c = mx.random.normal((1, 1, D))
        n = mx.random.normal((B, 1, D))
        out.append((rho**0.5 * c + (1 - rho) ** 0.5 * n).astype(mx.bfloat16))
    mx.eval(out)
    return out


def calib(B):
    lo, hi = 0.0, 0.999
    best = None
    for _ in range(14):
        rho = (lo + hi) / 2
        u, us = mean_u(make_x(B, rho, 1000 + B))
        best = (rho, u, us)
        if abs(u - TARGET_U[B]) < 0.6:
            break
        if u > TARGET_U[B]:
            lo = rho
        else:
            hi = rho
    return best


_prev = [None]


def run_all(fn, xs):
    if not a.chain:
        return [fn(m, x) for m, x in zip(moes, xs)]
    outs, prev = [], _prev[0]
    for m, x in zip(moes, xs):
        xi = x if prev is None else x + (0 * prev.reshape(-1)[:1].astype(x.dtype))
        prev = fn(m, xi)
        prev = prev[0] if isinstance(prev, tuple) else prev
        outs.append(prev)
    _prev[0] = prev
    return outs


def timeit(fn, xs, label):
    for _ in range(2):
        mx.eval(run_all(fn, xs))
    _prev[0] = None
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        outs = []
        for _ in range(a.reps):
            outs += run_all(fn, xs)
        mx.eval(outs)
        best = min(best, (time.perf_counter() - t0) / a.reps / len(moes) * 1e6)
    print(f"    {label:44s} {best:8.1f} us/layer", flush=True)
    return best


def full(m, x):
    return S.moe_verify(m, x, m.fast_prepare())


_DEF = {k: getattr(S, k) for k in dir(S) if k.isupper() and isinstance(getattr(S, k), (int, float))}


def apply_variant(var):
    for k, v in _DEF.items():
        setattr(S, k, v)
    S._perf_hdr = ""
    for kv in [t for t in var.split(",") if t]:
        k, v = kv.split("=")
        if k.startswith("SPEC_"):
            S._perf_hdr += "#define %s %d\n" % (k, int(v))
        else:
            setattr(S, k, float(v) if "." in v else int(v))
    S._kern.clear()


RES = {}
for B in [int(b) for b in a.batches.split(",")]:
    rho, u, us = calib(B)
    xs = make_x(B, rho, 1000 + B)
    print(f"B={B} rho={rho:.3f} U mean={u:.1f} (target {TARGET_U[B]}) per-layer={us}", flush=True)
    if a.stock:
        RES[(B, "stock")] = timeit(lambda m, x: m(x), xs, "stock SparseMoE (moesort+gather_qmm)")
    for var in [""] + [v for v in a.variants.replace("/", ";").split(";") if v]:
        apply_variant(var)
        print(f"  variant [{var}]", flush=True)
        f = timeit(full, xs, "full layer (router..combine)")
        RES[(B, var)] = f
        if a.stages:
            lg = timeit(lambda m, x: S.moe_verify(m, x, m.fast_prepare(), stages="upto_logits"), xs, "router")
            print(f"    => router {lg:.1f}", flush=True)
            r = timeit(lambda m, x: S.moe_verify(m, x, m.fast_prepare(), stages="upto_route"), xs, "router+topk")
            me = timeit(lambda m, x: S.moe_verify(m, x, m.fast_prepare(), stages="upto_meta"), xs, "..+meta")
            g = timeit(lambda m, x: S.moe_verify(m, x, m.fast_prepare(), stages="upto_gua"), xs, "..+gua")
            dn = timeit(lambda m, x: S.moe_verify(m, x, m.fast_prepare(), stages="upto_down"), xs, "..+down")
            print(f"    => route {r:.1f} meta {me - r:.1f} gua {g - me:.1f} down {dn - g:.1f} combine {f - dn:.1f}", flush=True)
print("SUMMARY us/layer:", {f"B{k[0]}:{k[1] or 'base'}": round(v, 1) for k, v in RES.items()})
