"""mega kernels vs the stock op chains they replace, real dims (bitwise)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa: F401
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2 import mega_kernels as mk
from mlx_vlm.models.inkling2.fast_decode import _controllers3, _controllers3_c

D, R, EPS = 2304, 64, 1e-6
SEEDS = int(os.environ.get("SEEDS", "20"))
res = {}

def cmp(name, ref, got):
    ok = mx.array_equal(ref, got).item()
    t = res.setdefault(name, [0, 0]); t[1] += 1; t[0] += ok
    if not ok and t[1] - t[0] <= 3:
        d = mx.abs(ref - got)
        print(f"  {name}: {int((d > 0).sum().item())} differ, max {d.max().item():.3e}")

outer3 = mx.compile(lambda x3, m3, dw, fw, gw: _controllers3(x3, m3, dw, fw, gw, EPS) * 1.0)
for seed in range(SEEDS):
    mx.random.seed(seed)
    dw = (mx.random.normal((3, R, 2 * D)) * 0.02).astype(mx.bfloat16)
    fw = (mx.random.normal((3, D, R)) * 0.2).astype(mx.bfloat16)
    gw = (mx.random.normal((3, D, R)) * 0.2).astype(mx.bfloat16)
    for nq in (3, 1):
        mixed = mx.random.normal((1, 1, nq, D)) * (1 + seed % 3)
        msgs = mx.random.normal((1, 1, 4, D)) * 0.7
        mx.eval(dw, fw, gw, mixed, msgs)
        parts = [mixed[..., min(i, nq - 1), :] for i in range(3)]
        x3 = mx.stack(parts)
        m3 = mx.moveaxis(msgs[..., :3, :], -2, 0)
        got = mk.controllers(mixed.reshape(nq, D), msgs.reshape(4, D), 0, dw, fw, gw, EPS, 3, nq)
        got = got.reshape(3, 1, 1, D)
        cmp("ctrl3 vs eager", _controllers3(x3, m3, dw, fw, gw, EPS), got)
        cmp("ctrl3 vs compiled", _controllers3_c(x3, m3, dw, fw, gw, EPS), got)
        cmp("ctrl3 vs nested-compiled", outer3(x3, m3, dw, fw, gw), got)
    # mlp controller (single, weights as views of a stack like FastDecoder)
    h = mx.random.normal((1, 1, D)) * 2
    mx.eval(h)
    ref = fk.message_controller(h, msgs[..., 3, :], dw[0], fw[0], gw[0], EPS)
    ref_e = fk._message_controller(h, msgs[..., 3, :], dw[0], fw[0], gw[0], EPS)
    got = mk.controllers(h.reshape(1, D), msgs.reshape(4, D), 3, dw[:1], fw[:1], gw[:1], EPS, 1, 1).reshape(1, 1, D)
    cmp("mlp vs compiled", ref, got)
    cmp("mlp vs eager", ref_e, got)
import mlx.nn as nn
norms = [nn.RMSNorm(D, eps=EPS) for _ in range(3)]
for seed in range(SEEDS):
    mx.random.seed(100 + seed)
    for nm in norms:
        nm.weight = (1 + 0.3 * mx.random.normal((D,))).astype(mx.bfloat16)
    mixed = mx.random.normal((1, 1, 3, D)) * (0.1 + seed)
    mx.eval(mixed, [nm.weight for nm in norms])
    parts = [mixed[..., i, :] for i in range(3)]
    got = mk.rms_norm_cast(parts, [nm.weight for nm in norms], EPS)
    for i in range(3):
        cmp("norm3 vs RMSNorm.astype", norms[i](parts[i]).astype(mx.bfloat16).reshape(D), got[i])
    got1 = mk.rms_norm_cast([mixed[..., 1, :]], [norms[0].weight], EPS)
    cmp("norm1", norms[0](mixed[..., 1, :]).astype(mx.bfloat16).reshape(D), got1[0])
from mlx_vlm.models.inkling2.language import DynamicShortConvolution
from types import SimpleNamespace
cfg = SimpleNamespace(sconv_kernel_size=4, hidden_size=D)
C, H = 1024, 8
mods = [DynamicShortConvolution(cfg, C, H, i) for i in (0, 1)]
def dref(xk, xv, dk, dv, sk, sv):
    conv = [sk, sv, None, None]
    yk = mods[0](xk, dk, conv, None)
    yv = mods[1](xv, dv, conv, None)
    return yk, yv, conv[0], conv[1]
dref_c = mx.compile(dref)
for seed in range(SEEDS):
    mx.random.seed(300 + seed)
    for m in mods:
        m.static_weight = (mx.random.normal((4, C)) * 0.5).astype(mx.bfloat16)
        m.weight_proj.weight = (mx.random.normal((H * 4, D)) * 0.02).astype(mx.bfloat16)
    xk, xv = [(mx.random.normal((1, 1, C)) * 2).astype(mx.bfloat16) for _ in range(2)]
    dk, dv = [(mx.random.normal((1, 1, D))).astype(mx.bfloat16) for _ in range(2)]
    sk, sv = [mx.random.normal((1, 3, C)) for _ in range(2)]
    mx.eval(xk, xv, dk, dv, sk, sv, [m.parameters() for m in mods])
    got = mk.dsconv_pair(mods[0], mods[1], xk, xv, dk, dv, sk, sv)
    dref_c = mx.compile(dref)  # params are captured at trace: re-trace per seed
    for nm, f in (("dsconv eager", dref), ("dsconv compiled", dref_c)):
        ref = f(xk, xv, dk, dv, sk, sv)
        for i, lab in enumerate(("yk", "yv", "sk", "sv")):
            cmp(f"{nm} {lab}", ref[i], got[i])
for seed in range(SEEDS):
    mx.random.seed(500 + seed)
    w32 = (mx.random.normal((128, D)) * 0.05).astype(mx.bfloat16).astype(mx.float32)
    xb = (mx.random.normal((1, 1, D)) * 2).astype(mx.bfloat16)
    mx.eval(w32, xb)
    cmp("router gemv", (xb.astype(mx.float32) @ w32.T).reshape(1, 128), mk.router_logits(xb, w32))
from mlx_vlm.models.inkling2.fast_decode import _attn_res_keys_c
for seed in range(SEEDS):
    mx.random.seed(700 + seed)
    for nk in (1, 4, 9):
        keys = [fk._block_key(mx.random.normal((1, 1, D)) * (1 + i)) for i in range(nk)]
        part = (mx.random.normal((1, 1, D)) * 3).astype(mx.bfloat16)
        q = (mx.random.normal((D,)) * 0.5).astype(mx.bfloat16)
        mx.eval(keys, part, q)
        cmp("attnres raw bf16 partial", _attn_res_keys_c(keys, part, q, True), fk.attn_res(keys, part, q, raw_partial=True))
for k, (ok, n) in res.items():
    print(f"{k}: {ok}/{n} bitwise")
