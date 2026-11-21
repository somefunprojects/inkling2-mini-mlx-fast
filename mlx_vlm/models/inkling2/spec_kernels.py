# Copyright (c) 2026 somefunprojects. MIT License (see LICENSE).
"""L>1 verify kernels for INKLING_SPECFAST. Verify rows only; the L=1
decode path never calls these. Class NONBITWISE-OK (each kernel is FP32-exact vs the stock math)."""
import mlx.core as mx

# --------------------------------------------------------------------------
# 1. Rel-bias slot mask for L query rows (+ sink column), one launch.
#    Per (b, h, l, j) the same serial FP32 dot over d_rel as fast_kernels._MASK_SRC (the stock
#    mask kernel's order), so every row equals the decode mask of that query position.
# --------------------------------------------------------------------------
_MASKL_SRC = r"""
    // rel: [B, L, H, D_REL] (strided), proj: [D_REL, R] (strided), slot_pos: [B, S] (strided),
    // qpos: [B, L] (strided), sinks: [B, L, H] (strided); out: [B, H, L, S + 1], column S = sink.
    const int L = rel_shape[1];
    const int H = rel_shape[2];
    const int S = slot_pos_shape[1];
    const int j = (int)thread_position_in_grid.x;
    const int l = (int)thread_position_in_grid.y;
    const int bh = (int)thread_position_in_grid.z;
    if (j > S || l >= L) return;
    const int b = bh / H, h = bh % H;
    const size_t o = ((size_t)bh * L + l) * (S + 1) + j;
    if (j == S) {
        out[o] = (T)sinks[(size_t)b * sinks_strides[0] + (size_t)l * sinks_strides[1]
                          + (size_t)h * sinks_strides[2]];
        return;
    }
    const int q = qpos[(size_t)b * qpos_strides[0] + (size_t)l * qpos_strides[1]];
    const int p = slot_pos[(size_t)b * slot_pos_strides[0] + (size_t)j * slot_pos_strides[1]];
    const int dist = q - p;
    T val;
    if (p < 0 || dist < 0 || (WINDOW > 0 && dist >= WINDOW)) {
        val = (T)(-1e30f);
    } else if (dist < REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0] + (size_t)l * rel_strides[1]
                           + (size_t)h * rel_strides[2];
        for (int d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0] + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[o] = val;
"""
_k = {}


def _kernel(name, src, inputs):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=["out"], source=src,
            ensure_row_contiguous=False)
    return k


def slot_mask_rows(rel, proj, slot_pos, qpos, window, sinks, dtype):
    """rel [B,L,H,dr], proj [dr,R], slot_pos [B,S] int32 (-1 empty), qpos [B,L] int32,
    sinks [B,L,H] -> [B,H,L,S+1] in dtype (mask of _rel_slot_mask + sink column)."""
    B, L, H, dr = rel.shape
    S = slot_pos.shape[1]
    if rel.dtype != dtype:
        rel = rel.astype(dtype)
    return _kernel("inkling2_spec_maskL", _MASKL_SRC,
                   ["rel", "proj", "slot_pos", "qpos", "sinks"])(
        inputs=[rel, proj, slot_pos.astype(mx.int32), qpos.astype(mx.int32), sinks],
        template=[("T", dtype), ("D_REL", dr), ("REL_EXTENT", proj.shape[1]),
                  ("WINDOW", int(window))],
        grid=(S + 1, L, B * H),
        threadgroup=(min(256, S + 1), 1, 1),
        output_shapes=[(B, H, L, S + 1)],
        output_dtypes=[dtype],
    )[0]
