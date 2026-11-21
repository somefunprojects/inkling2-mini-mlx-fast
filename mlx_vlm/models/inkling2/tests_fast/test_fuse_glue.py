"""fuse / fusemoe / glue pieces vs stock, realistic shapes, random tensors."""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _load import load
import mlx.core as mx
import mlx.nn as nn

fk = load("mlx_vlm.models.inkling2.fast_kernels")
lang = load("mlx_vlm.models.inkling2.language")
ink = load("mlx_vlm.models.inkling.language")
kernels = load("mlx_vlm.models.inkling2.kernels")
sw = load("mlx_vlm.models.switch_layers")
nv = load("mlx_vlm.quantization.nvfp4")
D, H, KV, HD, DREL = 2304, 24, 8, 128, 16
bf = mx.bfloat16
mx.random.seed(3)
results = []


def check(name, a, b):
    a, b = mx.array(a), mx.array(b)
    eq = a.shape == b.shape and mx.array_equal(a, b).item()
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() if a.shape == b.shape else float("nan")
    results.append((name, eq, d))
    print(f"{name:52s} bitwise={eq} max|d|={d:.3g}")


def lin(o, i):
    l = nn.Linear(i, o, bias=False)
    l.weight = (mx.random.normal((o, i)) / math.sqrt(i)).astype(bf)
    return l


# ---- fuse: [q; r; gate; sink] and [k; v] at M=1
q, r, g, s = lin(H * HD, D), lin(H * DREL, D), lin(H * HD, D), lin(H, D)
k, v = lin(KV * HD, D), lin(KV * HD, D)
x = mx.random.normal((1, 1, D)).astype(bf)
ref = [m(x) for m in (q, r, g, s, k, v)]
mx.eval(ref)
w, splits = fk.fuse_linears([q, r, g, s])
w2, splits2 = fk.fuse_linears([k, v])
got = mx.split(x @ w.T, splits, axis=-1) + mx.split(x @ w2.T, splits2, axis=-1)
for n, a, b in zip(("q", "r", "gate", "sink", "k", "v"), got, ref):
    check(f"fuse M=1 {n} (fused GEMV vs separate)", a, b)
# views still give identical results through the original Linear (L>1 path)
x8 = mx.random.normal((1, 8, D)).astype(bf)
check("fuse: q_proj via row view, L=8", q(x8), x8 @ mx.array(mx.contiguous(q.weight)).T)

# ---- fusemoe: gate+up one gather_qmm (nvfp4 + global scale; mxfp8)
E, I, TOPK = 128, 2304, 8
for mode, gs_, bits in (("nvfp4", 16, 4), ("mxfp8", 32, 8)):
    mlp = sw.SwitchGLU(D, I, E, activation=lang.SoftcappedSwiGLU(lang.TextConfig()))
    scale = (0.5 + mx.random.uniform(shape=(E,))) / 2688.0
    for name in ("gate_proj", "up_proj", "down_proj"):
        qq = sw.QuantizedSwitchLinear(D if name != "down_proj" else I, I if name != "down_proj" else D, E,
                                      bias=False, group_size=gs_, bits=bits, mode=mode)
        if mode == "nvfp4":
            # gate/up share the w13 amax -> same global scale; down has its own
            qq = nv.ScaledQuantizedSwitchLinear(qq, scale if name != "down_proj" else scale * 1.3, "weight_scale_2")
        mlp[name] = qq
    mlp.eval()
    mx.eval(mlp.parameters())
    xe = mx.random.normal((1, 1, D)).astype(bf)
    idx = mx.array(mx.random.permutation(E)[:TOPK].reshape(1, 1, TOPK))
    ref = mlp(xe, idx)
    mx.eval(ref)
    fused = fk.fuse_switch_gate_up(mlp)
    assert fused is not None
    xx = mx.expand_dims(xe, (-2, -3))
    up, gate = fk.fused_gate_up(fused, xx, idx)
    got = mlp.down_proj(mlp.activation(up, gate), idx, sorted_indices=False).squeeze(-2)
    check(f"fusemoe {mode} E={E} I={I} top{TOPK} (expert outputs)", got, ref)
    up_ref = mlp.up_proj(xx, idx)  # through the (now view) weights
    check(f"fusemoe {mode} up via non-contiguous view", up, up_ref)
    del mlp, fused

# ---- glue: sconv with folded state vs stock kernel + _next_conv_state
C, W = KV * HD, 4
state = mx.random.normal((1, W - 1, C))
wt = mx.random.normal((C, W)).astype(bf)
xs = mx.random.normal((1, 1, C)).astype(bf)
ref_out = kernels.short_convolution(state, wt, xs, None, W)
ref_state = ink._next_conv_state(state, xs, None)
out, nstate = fk.short_convolution_with_state(state, wt, xs, W)
check("glue sconv out (L=1)", out, ref_out)
check("glue sconv next state (L=1)", nstate, ref_state)
xs4 = mx.random.normal((2, 3, C)).astype(bf)
st2 = mx.random.normal((2, W - 1, C))
out, nstate = fk.short_convolution_with_state(st2, wt, xs4, W)
check("glue sconv out (B=2,L=3)", out, kernels.short_convolution(st2, wt, xs4, None, W))
check("glue sconv next state (B=2,L=3)", nstate, ink._next_conv_state(st2, xs4, None))

# output gate
o = mx.random.normal((1, 1, H * HD)).astype(bf)
gg = mx.random.normal((1, 1, H * HD)).astype(bf)
check("glue compiled output gate", fk.output_gate(o, gg),
      (o.astype(mx.float32) * mx.sigmoid(gg.astype(mx.float32))).astype(bf))

# message controller
class Cfg:  # minimal
    class loop_config: gate_rank = 64
    hidden_size = D
    rms_norm_eps = 1e-6
mc = lang.MessageController(Cfg)
mc.down_weight = (mx.random.normal((64, 2 * D)) / math.sqrt(2 * D)).astype(bf)
mc.forget_weight = mx.random.normal((D, 64)) / 8
mc.gain_weight = mx.random.normal((D, 64)) / 8
xm, msg = mx.random.normal((1, 1, D)), mx.random.normal((1, 1, D))
pair = mx.concatenate([mx.fast.rms_norm(t, None, 1e-6) for t in (xm, msg)], axis=-1)
hidden = (pair.astype(bf) @ mc.down_weight.T).astype(mx.float32)
ref = (1 - mx.tanh(hidden @ mc.forget_weight.T)) * xm + mx.tanh(hidden @ mc.gain_weight.T) * msg
check("glue compiled MessageController", fk.message_controller(xm, msg, mc.down_weight, mc.forget_weight, mc.gain_weight, 1e-6), ref)

# router with cached FP32 weight
gw = mx.random.normal((E, D)).astype(bf)
bias = mx.random.normal((E,)) * 0.01
xr = mx.random.normal((1, 1, D)).astype(bf)
i1, w1 = lang._route(xr, gw, bias, 8, 8.0)
i2, w2 = lang._route(xr, gw.astype(mx.float32), bias, 8, 8.0)
check("glue router: indices with cached FP32 weight", i2, i1)
check("glue router: weights with cached FP32 weight", w2, w1)

# log-n tau memo
tau = fk.log_n_tau(5000, 1, 0.1, 128000)
positions = mx.arange(1) + mx.array(5000)[..., None] + 1
check("glue log-n tau memo", tau, 1 + 0.1 * mx.log(mx.maximum(positions / 128000, 1)))

bad = [n for n, eq, _ in results if not eq]
print("NOT BITWISE:", bad if bad else "none")
assert not bad
print("PASS fuse/glue (all bitwise)")
