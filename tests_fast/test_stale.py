"""INKLING_FAST=fuse,fusemoe caches must follow weight reloads and module swaps (run with INKLING_FAST=all)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
def run(lm):
    c = lm.make_cache(); lm(inputs=mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]), cache=c)
    return lm(inputs=mx.array([[7]]), cache=c).logits
lm = tiny.build(window=16); run(lm)
a = tiny.build(window=16, seed=1)
lm.load_weights([(k, v) for k, v in tree_flatten(a.parameters())], strict=False)
print("after load_weights: max|d| =", mx.abs(run(lm) - run(a)).max().item())
class Wrap(nn.Module):  # LoRA-style wrapper with a zero delta
    def __init__(self, base):
        super().__init__(); self.base = base
    def __call__(self, x):
        return self.base(x) * 1.0
for layer in lm.model.layers:
    layer.self_attn.q_proj = Wrap(layer.self_attn.q_proj)
print("after module wrap:  max|d| =", mx.abs(run(lm) - run(a)).max().item())
