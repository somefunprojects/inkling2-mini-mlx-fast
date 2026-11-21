"""FastDecoder._moe (verify, L>1, expert-grouped) vs decode-path MoE per position: bitwise (bf16).
Run with INKLING_FAST=all to cover the fused gate+up buffer (scales forced equal)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
for equal_scales in (False, True):
    lm = tiny.build()
    if equal_scales:
        for layer in lm.model.layers:
            sm = layer.mlp.switch_mlp
            if hasattr(sm.gate_proj, "global_scale"):
                sm.up_proj[sm.up_proj._scale_name] = sm.gate_proj.global_scale
    dec = FastDecoder(lm, buckets=(64,))
    for li, layer in enumerate(lm.model.layers):
        moe = layer.mlp
        for B, L in ((1, 2), (1, 8), (4, 8)):
            x = (mx.random.normal((B, L, 256)) * 0.5).astype(mx.bfloat16)
            ref = mx.concatenate([moe(x[:, i:i + 1]) for i in range(L)], axis=1)
            got = dec._moe(moe, x)
            st = moe.__dict__.get("_fast_gate_up_state")
            print(f"equal_scales={equal_scales} layer {li} dedup={dec._moe_dedup_ok(moe)} fused={st is not None and st[1] is not None} "
                  f"B={B} L={L} bitwise={bool(mx.array_equal(ref, got).item())}")
