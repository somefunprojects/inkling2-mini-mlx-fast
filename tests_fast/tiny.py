"""Tiny random-weight Inkling2 loop model for fast_decode tests on any Apple-silicon Mac."""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.inkling", "mlx_vlm.models.inkling2", "mlx_vlm.speculative"):
    if name not in sys.modules:
        pkg = types.ModuleType(name); pkg.__path__ = [os.path.join(ROOT, *name.split("."))]
        sys.modules[name] = pkg
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_vlm.models.inkling2.config import TextConfig
from mlx_vlm.models.inkling2.language import LanguageModel
from mlx_vlm.quantization.nvfp4 import replace_scaled_quantized_linears


def tiny_config(window=16, rel_extent=32, n_floor=8):
    return TextConfig(
        model_type="inkling2_loop_text", hidden_size=256, num_hidden_layers=6,
        vocab_size=512, unpadded_vocab_size=500, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, swa_num_attention_heads=4,
        swa_num_key_value_heads=2, swa_head_dim=64, sliding_window_size=window,
        local_layer_ids=[0, 2, 4], d_rel=16, rel_extent=rel_extent,
        log_scaling_n_floor=n_floor, log_scaling_alpha=0.1, intermediate_size=128,
        n_routed_experts=8, num_experts_per_tok=2, n_layers_per_attn_res_block=2,
        loop_config=dict(core_start=2, core_end=4, num_passes=3, gate_rank=16),
    )


def build(seed=0, dtype=mx.bfloat16, **kw):
    mx.random.seed(seed)
    cfg = tiny_config(**kw)
    lm = LanguageModel(cfg)
    # random weights: small normals for matrices, ~1 for norms/scales
    def init(path, p):
        if p.ndim >= 2:
            return (mx.random.normal(p.shape) * (0.6 / p.shape[-1] ** 0.5)).astype(dtype)
        if "norm" in path or "scale" in path:
            return (1 + 0.1 * mx.random.normal(p.shape)).astype(dtype)
        return (0.1 * mx.random.normal(p.shape)).astype(dtype)
    params = tree_flatten(lm.parameters())
    lm.load_weights([(k, init(k, v)) for k, v in params], strict=False)
    # rel_proj / queries need to be non-trivial
    # quantize experts: nvfp4 (with per-expert global scale) except last layer mxfp8
    last = cfg.num_hidden_layers - 1
    def pred_nv(path, m):
        return ".switch_mlp." in path and hasattr(m, "to_quantized") and not path.startswith(f"model.layers.{last}.")
    def pred_mx(path, m):
        return ".switch_mlp." in path and hasattr(m, "to_quantized") and path.startswith(f"model.layers.{last}.")
    nn.quantize(lm, group_size=16, bits=4, mode="nvfp4", class_predicate=pred_nv)
    nn.quantize(lm, group_size=32, bits=8, mode="mxfp8", class_predicate=pred_mx)
    scales = {}
    for i in range(last):
        for nm in ("gate_proj", "up_proj", "down_proj"):
            scales[f"model.layers.{i}.mlp.switch_mlp.{nm}.weight_scale_2"] = (
                0.5 + mx.random.uniform(shape=(cfg.n_routed_experts,))).astype(mx.float32) / 2688.0 * 8
    replace_scaled_quantized_linears(lm, scales)
    lm.eval()
    mx.eval(lm.parameters())
    return lm


import contextlib


@contextlib.contextmanager
def true_stock():
    """Run the language path with every INKLING_FAST flag off (import-time
    module flags included), whatever the process env says. Use it for the
    reference side of an equivalence test: a flagged "stock" hides any
    non-bitwise fast path shared by both sides."""
    from mlx_vlm.models.inkling2 import fast_kernels as fk, language as lang
    saved_flags = fk.FLAGS
    saved = {k: v for k, v in vars(lang).items() if k.startswith("_FAST_")}
    fk.FLAGS = frozenset()
    for k in saved:
        setattr(lang, k, False)
    try:
        yield
    finally:
        fk.FLAGS = saved_flags
        for k, v in saved.items():
            setattr(lang, k, v)
