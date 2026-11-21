"""INKLING_FAST=moesort: expert-sorted grouped decode MoE vs stock SwitchGLU."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from _load import load
import mlx.core as mx

fk = load("mlx_vlm.models.inkling2.fast_kernels")
lang = load("mlx_vlm.models.inkling2.language")
sw = load("mlx_vlm.models.switch_layers")
nv = load("mlx_vlm.quantization.nvfp4")
mx.random.seed(11)
D, I, E, K = 2304, 1152, 128, 8   # Inkling 2 Mini expert shapes


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


ok = True
for mode in ("nvfp4", "mxfp8"):
    mlp = make(mode)
    print(mode, "moesort_supported:", fk.moesort_supported(mlp),
          {n: fk._grouped_kind(getattr(mlp, n), 256) for n in ("gate_proj", "up_proj", "down_proj")})
    for B in (4, 8, 16):
        for trial in range(3):
            x = mx.random.normal((B, 1, D)).astype(mx.bfloat16)
            # realistic: random distinct experts per token; trial 2 forces heavy overlap
            hi = E if trial < 2 else 12
            idx = mx.stack([mx.random.permutation(hi)[:K] for _ in range(B)]).reshape(B, 1, K).astype(mx.uint32)
            ref = mlp(x, idx)
            got = fk.sorted_switch_glu(mlp, x, idx)
            eq = mx.array_equal(ref, got).item()
            d = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max().item()
            ok &= eq
            print(f"  {mode} B={B:2d} {'overlap' if trial == 2 else 'random '} bitwise={eq} max|d|={d:.3g}")
    # timing: 24 chained layers' worth (same weights), stock vs sorted
    for B in (4, 8, 16):
        x0 = mx.random.normal((B, 1, D)).astype(mx.bfloat16)
        idxs = [mx.stack([mx.random.permutation(E)[:K] for _ in range(B)]).reshape(B, 1, K).astype(mx.uint32) for _ in range(24)]
        mx.eval(x0, idxs)
        def chain(f):
            x = x0
            for idx in idxs:
                x = (x + f(x, idx).sum(-2) * 1e-3).astype(mx.bfloat16)
            return x
        res = {}
        for name, f in (("stock", lambda x, i: mlp(x, i)), ("moesort", lambda x, i: fk.sorted_switch_glu(mlp, x, i))) * 2:
            for _ in range(3): mx.eval(chain(f))
            n = 20; t = time.perf_counter()
            for _ in range(n): mx.eval(chain(f))
            res[name] = (time.perf_counter() - t) / n * 1e3
        print(f"  {mode} B={B:2d} 24 MoE calls: stock {res['stock']:.2f} ms, moesort {res['moesort']:.2f} ms")
    # fusemoe + moesort: gate/up become strided views of the fused buffer.
    # Stock on unfused weights is the reference; sorted_switch_glu must match
    # it bitwise both when handed the fused state and on the bare views.
    cases = []
    for B in (4, 8, 16):
        x = mx.random.normal((B, 1, D)).astype(mx.bfloat16)
        idx = mx.stack([mx.random.permutation(E)[:K] for _ in range(B)]).reshape(B, 1, K).astype(mx.uint32)
        cases.append((B, x, idx, mlp(x, idx)))
    mx.eval([c[3] for c in cases])
    fused = fk.fuse_switch_gate_up(mlp)
    assert fused is not None, "test weights should be fusable"
    for B, x, idx, ref in cases:
        for name, got in (("fused", fk.sorted_switch_glu(mlp, x, idx, fused)),
                          ("views", fk.sorted_switch_glu(mlp, x, idx)),
                          ("stock-views", mlp(x, idx))):
            eq = mx.array_equal(ref, got).item(); ok &= eq
            print(f"  {mode} B={B:2d} fusemoe {name:11s} bitwise={eq}")
    for B, x0, _, _ in cases:
        idxs = [mx.stack([mx.random.permutation(E)[:K] for _ in range(B)]).reshape(B, 1, K).astype(mx.uint32) for _ in range(24)]
        mx.eval(idxs)
        def chain(f):
            x = x0
            for idx in idxs:
                x = (x + f(x, idx).sum(-2) * 1e-3).astype(mx.bfloat16)
            return x
        res = {}
        for name, f in (("views", lambda x, i: fk.sorted_switch_glu(mlp, x, i)),
                        ("fused", lambda x, i: fk.sorted_switch_glu(mlp, x, i, fused)),
                        ("stock", lambda x, i: mlp(x, i))) * 2:
            for _ in range(3): mx.eval(chain(f))
            n = 20; t = time.perf_counter()
            for _ in range(n): mx.eval(chain(f))
            res[name] = (time.perf_counter() - t) / n * 1e3
        print(f"  {mode} B={B:2d} fusemoe, 24 MoE calls: moesort-on-views {res['views']:.2f} ms, "
              f"moesort-fused {res['fused']:.2f} ms, stock-on-views {res['stock']:.2f} ms")
    del mlp
print("ALL BITWISE" if ok else "MISMATCH")
assert ok
