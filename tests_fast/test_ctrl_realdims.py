"""Batched controllers / cached-key AttentionResidual at real Inkling-2-Mini dims (D=2304, r=64); bitwise up to B=8 in our runs, ctrl3 differs at B=16 (the decoder gates it to B*L <= 8)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # package stubs
import mlx.core as mx
from mlx_vlm.models.inkling2 import fast_kernels as fk
from mlx_vlm.models.inkling2.language import _attention_residual
from mlx_vlm.models.inkling2.fast_decode import _controllers3, _controllers3_c, _attn_res_keys_c
D, r = 2304, 64
mx.random.seed(0)
dw = (mx.random.normal((3, r, 2 * D)) * 0.02).astype(mx.bfloat16)
fw = (mx.random.normal((3, D, r)) * 0.1).astype(mx.bfloat16)
gw = (mx.random.normal((3, D, r)) * 0.1).astype(mx.bfloat16)
for B in (1, 4, 8, 16):
    x3 = mx.random.normal((3, B, 1, D)); m3 = mx.random.normal((3, B, 1, D))
    ref = mx.stack([fk._message_controller(x3[i], m3[i], dw[i], fw[i], gw[i], 1e-6) for i in range(3)])
    refc = mx.stack([fk.message_controller(x3[i], m3[i], dw[i], fw[i], gw[i], 1e-6) for i in range(3)])
    a = _controllers3(x3, m3, dw, fw, gw, 1e-6); b = _controllers3_c(x3, m3, dw, fw, gw, 1e-6)
    blocks = [mx.random.normal((B, 1, D)).astype(mx.bfloat16) for _ in range(4)]
    part = mx.random.normal((B, 1, D))
    for nq in (1, 3):
        q = mx.random.normal((D,) if nq == 1 else (nq, D)).astype(mx.bfloat16)
        s = _attention_residual(mx.stack([*blocks, part], axis=-2), q, True)
        bk = [mx.fast.rms_norm(x.astype(mx.float32)[..., None, :], None, 1e-6) for x in blocks]
        t = _attn_res_keys_c(bk, part, q, True)
        print(f"B={B} attn_res nq={nq} max|d|={mx.abs(s - t).max().item():.1e}", end="; ")
    print(f"ctrl3 vs per-ctrl {mx.abs(a - ref).max().item():.1e}, compiled {mx.abs(b - refc).max().item():.1e}")
