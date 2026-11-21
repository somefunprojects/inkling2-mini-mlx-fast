"""Random NVFP4 SparseMoE builder for spec_moe tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny  # noqa
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.inkling2.language import SparseMoE
from mlx_vlm.quantization.nvfp4 import replace_scaled_quantized_linears


def build_real(seed, dtype, fused_scales, dims=(2304, 1152, 128, 8)):
    mx.random.seed(seed)
    D, NH, E, KT = dims
    cfg = tiny.tiny_config()
    cfg.hidden_size, cfg.intermediate_size = D, NH
    cfg.n_routed_experts, cfg.num_experts_per_tok = E, KT
    moe = SparseMoE(cfg)
    sm = moe.switch_mlp
    gw = mx.random.normal((E, D)) * (1.0 / D**0.5)
    gw[7] = gw[3]  # router ties: experts 3 and 7 always score equal
    gw[E - 1] = gw[3]
    moe.gate_weight = gw.astype(dtype)
    bias = 0.01 * mx.random.normal((E,))
    bias[7] = bias[3]
    bias[E - 1] = bias[3]
    moe.e_score_correction_bias = bias.astype(mx.float32)
    for nm, (o, i) in (("gate_proj", (NH, D)), ("up_proj", (NH, D)), ("down_proj", (D, NH))):
        getattr(sm, nm).weight = (mx.random.normal((E, o, i)) * (0.6 / i**0.5)).astype(dtype)
    nn.quantize(moe, group_size=16, bits=4, mode="nvfp4",
                class_predicate=lambda p, m: "switch_mlp" in p and hasattr(m, "to_quantized"))
    scales = {}
    g = (0.5 + mx.random.uniform(shape=(E,))).astype(mx.float32) / 2688.0 * 8
    for nm in ("gate_proj", "up_proj", "down_proj"):
        s = g if (fused_scales and nm == "up_proj") else (
            0.5 + mx.random.uniform(shape=(E,))).astype(mx.float32) / 2688.0 * 8
        if nm == "gate_proj":
            s = g
        scales[f"switch_mlp.{nm}.weight_scale_2"] = s
    replace_scaled_quantized_linears(moe, scales)
    moe.eval()
    mx.eval(moe.parameters())
    return moe


