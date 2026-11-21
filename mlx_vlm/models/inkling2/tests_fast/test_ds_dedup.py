"""verify_dedup: expert-grouped verify MoE vs the decode (L=1) path, bitwise. Random weights at Inkling 2 Mini
shapes (D=2304, I=1152, E=128, k=8), nvfp4 (ScaledQuantizedSwitchLinear) and mxfp8.
Reference = stock SwitchGLU at L=1 per position (what decode and FastDecoder tokenwise run)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from _load import load
import mlx.core as mx

fk = load("mlx_vlm.models.inkling2.fast_kernels")
lang = load("mlx_vlm.models.inkling2.language")
sw = load("mlx_vlm.models.switch_layers")
nv = load("mlx_vlm.quantization.nvfp4")
vd = load("mlx_vlm.models.inkling2.verify_dedup")
mx.random.seed(7)
D, I, E, K = 2304, 1152, 128, 8


def make(mode):
    mlp = sw.SwitchGLU(D, I, E, activation=lang.SoftcappedSwiGLU(lang.TextConfig()))
    gsz, bits = (16, 4) if mode == "nvfp4" else (32, 8)
    scale = (0.5 + mx.random.uniform(shape=(E,))) / 2688.0
    for name in ("gate_proj", "up_proj", "down_proj"):
        i, o = (D, I) if name != "down_proj" else (I, D)
        q = sw.QuantizedSwitchLinear(i, o, E, bias=False, group_size=gsz, bits=bits, mode=mode)
        if mode == "nvfp4":
            q = nv.ScaledQuantizedSwitchLinear(q, scale * (1.3 if name == "down_proj" else 1), "weight_scale_2")
        mlp[name] = q
    mlp.eval()
    mx.eval(mlp.parameters())
    return mlp


def tokenwise(mlp, x, idx):
    return mx.concatenate([mlp(x[:, i:i + 1], idx[:, i:i + 1]) for i in range(x.shape[1])], axis=1)


ok = True
for mode in ("nvfp4", "mxfp8"):
    mlp = make(mode)
    print(mode, "supported:", vd.supported(mlp), {n: vd.kind_of(getattr(mlp, n)) for n in ("gate_proj", "up_proj", "down_proj")})
    cases = []
    for L in (2, 5, 8):
        for trial in range(3):
            x = (mx.random.normal((1, L, D)) * (3.0 if trial == 1 else 1.0)).astype(mx.bfloat16)
            hi = E if trial < 2 else 12   # trial 2: heavy expert overlap
            idx = mx.stack([mx.random.permutation(hi)[:K] for _ in range(L)]).reshape(1, L, K).astype(mx.uint32)
            cases.append((L, trial, x, idx, tokenwise(mlp, x, idx)))
    mx.eval([c[4] for c in cases])
    for V in (2, 4, 8):
        for L, trial, x, idx, ref in cases:
            got = vd.dedup_routes(mlp, x, idx, V=V)
            eq = mx.array_equal(ref, got).item(); ok &= eq
            if not eq or V == 2:
                d = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max().item()
                print(f"  {mode} V={V} L={L} trial={trial} bitwise={eq} max|d|={d:.3g}")
    # combine over [1,L,k,D] vs per position
    for L, trial, x, idx, ref in cases[:3]:
        w = mx.random.uniform(shape=(1, L, K))
        a = lang._combine_routes(ref, w)
        b = mx.concatenate([lang._combine_routes(ref[:, i:i + 1], w[:, i:i + 1]) for i in range(L)], axis=1)
        eq = mx.array_equal(a, b).item(); ok &= eq
        print(f"  {mode} combine L={L} bitwise={eq}")
    # fusemoe: gate/up are views of the fused buffer afterwards; reference stays the unfused stock result
    fused = fk.fuse_switch_gate_up(mlp)
    for L, trial, x, idx, ref in cases:
        got = vd.dedup_routes(mlp, x, idx, V=2, fused=fused)
        eq = mx.array_equal(ref, got).item(); ok &= eq
        if not eq or L == 5:
            print(f"  {mode} fused L={L} trial={trial} bitwise={eq}")
    # timing: 24 chained layers, L=5 (n=4) -- tokenwise vs dedup (fused buffers)
    for L in ((3, 5, 8) if os.environ.get("DS_TIME") else ()):
        x0 = mx.random.normal((1, L, D)).astype(mx.bfloat16)
        idxs = [mx.stack([mx.random.permutation(E)[:K] for _ in range(L)]).reshape(1, L, K).astype(mx.uint32) for _ in range(24)]
        mx.eval(x0, idxs)

        def chain(f):
            x = x0
            for idx in idxs:
                x = (x + f(x, idx).sum(-2) * 1e-3).astype(mx.bfloat16)
            return x
        res = {}
        for name, f in (("tokenwise", lambda x, i: tokenwise(mlp, x, i)),
                        ("dedup", lambda x, i: vd.dedup_routes(mlp, x, i, fused=fused))) * 2:
            for _ in range(3): mx.eval(chain(f))
            n = 20; t = time.perf_counter()
            for _ in range(n): mx.eval(chain(f))
            res[name] = (time.perf_counter() - t) / n * 1e3
        print(f"  {mode} L={L} 24 MoE calls: tokenwise {res['tokenwise']:.2f} ms, dedup {res['dedup']:.2f} ms")
print("ALL BITWISE" if ok else "MISMATCH")
