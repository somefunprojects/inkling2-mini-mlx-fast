"""mega_sdpa (mega_kernels.sdpa_exact) vs mx.fast.scaled_dot_product_attention on
the FastDecoder exact operands, bitwise. Real dims (HQ 24, HK 8, D 128, bf16).
Sliced regime: SDPA on kb[:, :, :N] (global N = whole buffer, sliding N = cap + 1);
gather regime: SDPA on the temporal ring gather (N = W + 1). Rows after the sink
are filled with garbage (finite) as well as zeros: skipping them must not matter."""
import os, sys, time
import mlx.core as mx
from mlx_vlm.models.inkling2 import mega_kernels as mk

HQ, HK, D = 24, 8, 128
W = 1024
NEG = -1e30
dt = mx.bfloat16
scale = D ** -0.5
res = {}
fails = 0


def mk_mask(n_rows, real, sink_at, lead_masked=0):
    m = (mx.random.normal((1, HQ, 1, n_rows)) * 3).astype(dt)
    col = mx.arange(n_rows)[None, None, None, :]
    sink = (mx.random.normal((1, HQ, 1, 1)) * 2).astype(dt)
    m = mx.where(col == sink_at, sink, m)
    m = mx.where((col > sink_at) | (col < lead_masked), mx.array(NEG, dt), m)
    return m


def check(name, ref, got):
    global fails
    ok = mx.array_equal(ref, got).item()
    t = res.setdefault(name, [0, 0]); t[1] += 1; t[0] += ok
    if not ok:
        fails += 1
        d = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))
        if fails <= 10:
            print(f"  FAIL {name}: {int((d > 0).sum().item())} differ, max {d.max().item():.3e}", flush=True)


def ref_sdpa(q, k, v, m):
    o = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k, v, scale=scale, mask=m)
    return o.transpose(0, 2, 1, 3)


for seed in range(int(os.environ.get("SEEDS", "3"))):
    mx.random.seed(seed)
    # ---- sliced: global buffers (KS = N) and sliding sliced (KS = R + 1, N = cap + 1)
    for KS, N, ps in (
        (1023, 1023, (0, 5, 30, 31, 32, 100, 511, 700, 1020, 1021)),
        (1025, 1023, (0, 40, 600, 1021)),             # sliding ring rows, N = cap + 1
        (2049, 2049, (1022, 1023, 1500, 2047)),
        (4097, 4097, (2048, 3000, 4095)),
        (8193, 8193, (4096, 4300, 8191)),
        (1200, 1100, (126, 127, 128, 900, 1098)),     # 2-pass, few keys/blocks edge
    ):
        for p in ps:
            if p + 1 >= N:
                continue
            for garbage in (False, True):
                q = (mx.random.normal((1, 1, HQ, D)) * (1 + seed)).astype(dt)
                kb = mx.random.normal((1, HK, KS, D)).astype(dt)
                vb = mx.random.normal((1, HK, KS, D)).astype(dt)
                row = mx.arange(KS)[None, None, :, None]
                keep = ((row <= p) | (row > p + 1)) if garbage else (row <= p)
                kb = mx.where(keep, kb, mx.zeros_like(kb))
                vb = mx.where(keep, vb, mx.zeros_like(vb))
                m = mk_mask(N, p + 1, p + 1)
                qp = mx.array([p], mx.int32)
                mx.eval(q, kb, vb, m)
                ref = ref_sdpa(q, kb[:, :, :N], vb[:, :, :N], m)
                got = mk.sdpa_exact(q, kb, vb, m, qp, N, False, W, 1024, scale)
                two, bl = mk.sdpa_config(HQ, HK, N)
                check(f"sliced N={N} {'2pass/' + str(bl) if two else '1pass'}", ref, got)
    # ---- gather regime: ring of R slots + zero row R, N = W + 1
    for R in (W, W + 7):
        for p in (1022, 1023, 1024, 1500, 2047, 4000, 4351):
            q = (mx.random.normal((1, 1, HQ, D)) * (1 + seed)).astype(dt)
            kb = mx.random.normal((1, HK, R + 1, D)).astype(dt)
            vb = mx.random.normal((1, HK, R + 1, D)).astype(dt)
            row = mx.arange(R + 1)[None, None, :, None]
            kb = mx.where(row == R, mx.zeros_like(kb), kb)
            vb = mx.where(row == R, mx.zeros_like(vb), vb)
            S = min(p + 1, W)
            base = max(p - W + 1, 0)
            t = mx.arange(W + 1)
            idx = mx.where(t < S, (base + t) % R, R)
            ke, ve = mx.take(kb, idx, axis=2), mx.take(vb, idx, axis=2)
            m = mk_mask(W + 1, S, S, lead_masked=int(os.environ.get("LEAD", "3")))
            qp = mx.array([p], mx.int32)
            mx.eval(q, kb, vb, m, ke, ve)
            ref = ref_sdpa(q, ke, ve, m)
            got = mk.sdpa_exact(q, kb, vb, m, qp, W + 1, True, W, R, scale)
            check(f"gather R={R}", ref, got)

for k, (ok, n) in sorted(res.items()):
    print(f"{k:28s} {ok}/{n} bitwise")

# timing: 40 dependent calls (output -> next query) inside one mx.compile'd
# function, i.e. the GPU timeline incl. launch + barrier, as in the FD step.
def chain(fn, q0, reps=40):
    def f(q):
        for _ in range(reps):
            q = fn(q)
        return q
    g = mx.compile(f)
    for _ in range(3):
        mx.eval(g(q0))
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); mx.eval(g(q0)); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[3] / reps * 1e6

from mlx_vlm.models.inkling2 import fast_kernels as fk
mx.random.seed(0)
for KS, N, p, gather in ((1023, 1023, 600, False), (1025, 1023, 600, False), (1025, 1025, 4300, True),
                         (8193, 8193, 4300, False)):
    kb = mx.random.normal((1, HK, KS, D)).astype(dt)
    vb = mx.random.normal((1, HK, KS, D)).astype(dt)
    q0 = mx.random.normal((1, 1, HQ, D)).astype(dt)
    m = mk_mask(N, p + 1, min(p + 1, W) if gather else p + 1)
    qp = mx.array([p], mx.int32)
    mx.eval(q0, kb, vb, m)
    if gather:
        t_ref = chain(lambda q: ref_sdpa(q, *fk.ring_gather(kb, vb, qp, W, 1024), m), q0)
    else:
        t_ref = chain(lambda q: ref_sdpa(q, kb[:, :, :N], vb[:, :, :N], m), q0)
    out = [f"stock(+gather) {t_ref:.1f}"]
    for tk in (1, 2, 4, 8):
        mk.SDPA_TK = tk
        out.append(f"TK{tk} {chain(lambda q: mk.sdpa_exact(q, kb, vb, m, qp, N, gather, W, 1024, scale), q0):.1f}")
    print(f"timing KS={KS} N={N} p={p} gather={gather} us/call: " + "  ".join(out), flush=True)

print("SDPA UNIT", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
