"""Stage-by-stage diff of spec_moe vs the decode path (one real-dims layer, L=5)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("INKLING_SPECFAST", "moe")
import tiny  # noqa
import mlx.core as mx
from mlx_vlm.models.inkling2 import spec_moe as S
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2 import mega_kernels as mk
from mlx_vlm.models.inkling2.language import _combine_routes
import spec_moe_util as TB

mx.set_memory_limit(60 * 1024**3)
dtype = mx.bfloat16
for fused_scales in (True, False):
    moe = TB.build_real(1, dtype, fused_scales)
    fused = moe.fast_prepare()
    L, D = 5, 2304
    x = mx.random.normal((1, L, D)).astype(dtype)
    w32 = moe._fast_router_weight()
    R = L
    lg = S.router_rows(x.reshape(R, D), w32)
    ref_lg = mx.concatenate([x[:, i].astype(mx.float32) @ w32.T for i in range(L)], 0)
    ref_lg2 = mx.concatenate([mk.router_logits(x[:, i], w32) for i in range(L)], 0)
    print("fused", fused is not None)
    print(" logits eq gemv", mx.array_equal(lg, ref_lg).item(), "eq mega", mx.array_equal(lg, ref_lg2).item())
    idx, wts = fk.route_topk(lg, moe.e_score_correction_bias, moe.top_k, moe.route_scale)
    ri = []; rw = []
    for i in range(L):
        a, b = fk.route(x[:, i:i+1], w32, moe.e_score_correction_bias, moe.top_k, moe.route_scale)
        ri.append(a.reshape(1, -1)); rw.append(b.reshape(1, -1))
    ri = mx.concatenate(ri, 0); rw = mx.concatenate(rw, 0)
    print(" idx eq", mx.array_equal(idx, ri).item(), " wts eq", mx.array_equal(wts, rw).item())
    # per route stock expert outputs (decode path) vs kernels
    sm = moe.switch_mlp
    outs = []
    for i in range(L):
        xi = x[:, i:i+1]
        if fused is not None:
            o = moe._fast_decode_experts(xi, ri[i].reshape(1, 1, -1), fused)
        else:
            o = sm(xi, ri[i].reshape(1, 1, -1))
        outs.append(o.reshape(-1, D))
    ref_y = mx.concatenate(outs, 0)  # [P, D]
    # my kernel pipeline pieces
    out = S.moe_verify(moe, x, fused)
    _, sidx, _, act, y, probe = S.moe_verify(moe, x, fused, stages=True)
    print(" idx", sidx.tolist())
    print(" probe", probe.tolist())
    print(" act[0,:8]", act[0, :8].tolist(), "ref", "(below)")
    # stock act per route
    racts = []
    for i in range(L):
        xe = mx.expand_dims(x[:, i:i+1], (-2, -3))
        if fused is not None:
            up, gate = fk.fused_gate_up(fused, xe, ri[i].reshape(1, 1, -1))
        else:
            up = sm.up_proj(xe, ri[i].reshape(1, 1, -1)); gate = sm.gate_proj(xe, ri[i].reshape(1, 1, -1))
        racts.append(sm.activation(up, gate).reshape(-1, act.shape[-1]))
    ract = mx.concatenate(racts, 0)
    da = mx.abs(ract.astype(mx.float32) - act.astype(mx.float32))
    print(" ract[0,:8]", ract[0, :8].tolist())
    print(" caps", S._caps(sm.activation.gate_cap, sm.activation.up_cap).tolist(), sm.activation.gate_cap, sm.activation.up_cap)
    y2 = S.moe_verify(moe, x, fused, stages="act_in", stages_act=ract)
    print(" down(stock act) eq", mx.array_equal(y2, ref_y).item(), mx.abs(y2.astype(mx.float32) - ref_y.astype(mx.float32)).max().item())
    print(" act eq", mx.array_equal(ract, act).item(), "max|d|", da.max().item(), "max|ref|", mx.abs(ract).max().item(), "ndiff", (da > 0).sum().item(), "of", da.size)
    mx.eval(ref_y, y)
    d = mx.abs(ref_y.astype(mx.float32) - y.astype(mx.float32))
    print(" y eq", mx.array_equal(ref_y, y).item(), "max|d|", d.max().item(), "max|ref|", mx.abs(ref_y).max().item(),
          "n diff elems", (d > 0).sum().item(), "of", d.size)
    ref = mx.concatenate([moe(x[:, i:i+1]) for i in range(L)], 1)
    comb = _combine_routes(ref_y.reshape(1, L, -1, D), rw.reshape(1, L, -1))
    comb_rows = mx.concatenate([_combine_routes(ref_y.reshape(1, L, -1, D)[:, i:i+1], rw.reshape(1, L, -1)[:, i:i+1]) for i in range(L)], 1)
    print(" combine(L rows) eq ref", mx.array_equal(comb, ref).item(), " combine per row eq ref", mx.array_equal(comb_rows, ref).item(),
          " out eq ref", mx.array_equal(out, ref).item(), "max|ref|", mx.abs(ref).max().item())
