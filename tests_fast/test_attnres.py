"""attn_res kernel vs stock _attn_res_keys / attn_res_prefix (bitwise).

megafix: every (B, L) with B*L <= 16 the gate can admit (L in {1, 2, 8},
B*L = 16), N = nk + 1 up to 13 (Metal buffer-slot bound; 14, 20 must be gated), f32 / bf16 partial (raw_partial too) and bf16 / f32 query.
Shapes the gate refuses are counted, not run, except in the "gate needed" section, which runs the
kernel ungated at shapes past the gemv split-K bound (D < 16 * rows * N) to show the gate is
load-bearing there.
Env: SEEDS (default 6), D (default 2304)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: F401
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2.fast_decode import _attn_res_keys, _attn_res_keys_c

D = int(os.environ.get("D", "2304"))
SEEDS = int(os.environ.get("SEEDS", "6"))
SHAPES = [(1, 1), (4, 1), (16, 1), (1, 2), (4, 2), (8, 2), (1, 8), (2, 8), (1, 16), (4, 4)]
NKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19]
res, gated = {}, {}


def cmp(name, ref, got):
    ok = mx.array_equal(ref, got).item()
    t = res.setdefault(name, [0, 0]); t[1] += 1; t[0] += ok
    if not ok and t[1] - t[0] <= 4:
        d = mx.abs(ref - got)
        print(f"  FAIL {name}: {int((d > 0).sum().item())} differ, max {d.max().item():.3e}", flush=True)


def case(B, L, nk, pdt, qdt, D=D):
    keys = [fk._block_key(mx.random.normal((B, L, D)) * (1 + i)) for i in range(nk)]
    partial = (mx.random.normal((B, L, D)) * 3).astype(pdt)
    q = (mx.random.normal((D,)) * 0.5).astype(qdt)
    mx.eval(keys, partial, q)
    return keys, partial, q


for seed in range(SEEDS):
    mx.random.seed(seed)
    for (B, L) in SHAPES:
        for nk in NKS:
            for pdt, qdt in ((mx.float32, mx.bfloat16), (mx.bfloat16, mx.bfloat16), (mx.float32, mx.float32)):
                keys, partial, q = case(B, L, nk, pdt, qdt)
                lab = f"B{B}L{L}"
                if not fk.attn_res_supported(partial, q, nk):
                    gated[lab] = gated.get(lab, 0) + 1
                    continue
                tag = f"{lab} p={'f32' if pdt == mx.float32 else 'bf16'} q={'f32' if qdt == mx.float32 else 'bf16'}"
                got = fk.attn_res(keys, partial, q)
                cmp(f"{tag} vs compiled", _attn_res_keys_c(keys, partial, q, True), got)
                cmp(f"{tag} vs eager", _attn_res_keys(keys, partial, q, True), got)
                if pdt == mx.bfloat16:  # INKLING_FAST=mega_attnres
                    cmp(f"{tag} raw_partial", got, fk.attn_res(keys, partial, q, raw_partial=True))
                if nk >= 2:  # xpass prefix path
                    P = nk // 2
                    sc = fk.depth_scores(keys[:P], q)
                    cmp(f"{tag} vs prefix", fk.attn_res_prefix(sc, keys[:P], keys[P:], partial, q), got)

print("gated off (not run):", gated)
for k in sorted(res):
    ok, n = res[k]
    print(f"{k}: {ok}/{n} bitwise")
bad = sum(n - ok for ok, n in res.values())
tot = sum(n for ok, n in res.values())
print(f"attn_res kernel vs stock (gated shapes): {tot - bad}/{tot} bitwise")

# Gate needed: past D >= 16 * rows * N stock switches gemv variant. Run the kernel ungated there.
print("-- ungated past the split-K bound (expect some non-bitwise; the gate must refuse these):")
for (Dx, B, L, nk) in ((1024, 8, 1, 8), (1024, 1, 8, 8), (2304, 16, 1, 9), (2304, 1, 16, 12), (2304, 4, 4, 12)):
    nb = n = 0
    for seed in range(SEEDS):
        mx.random.seed(1000 + seed)
        keys, partial, q = case(B, L, nk, mx.float32, mx.bfloat16, D=Dx)
        assert not fk.attn_res_supported(partial, q, nk), (Dx, B, L, nk)
        n += 1
        nb += mx.array_equal(_attn_res_keys_c(keys, partial, q, True), fk.attn_res(keys, partial, q)).item()
    print(f"  D={Dx} B={B} L={L} N={nk + 1}: gate refuses; kernel would be {nb}/{n} bitwise")
# Metal's 31 buffer slots: 2 * nk + 7 <= 31 -> nk <= 12 at any shape.
for nk in (13, 20, 31):
    keys, partial, q = case(1, 1, nk, mx.float32, mx.bfloat16)
    ok = not fk.attn_res_supported(partial, q, nk)
    print(f"  nk={nk} (N={nk + 1}) B*L=1: gate refuses (buffer slots) = {ok}")
    bad += not ok
print("ATTNRES", "PASS" if bad == 0 else "FAIL")
