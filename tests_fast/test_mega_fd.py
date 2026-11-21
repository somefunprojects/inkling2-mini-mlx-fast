"""FastDecoder with vs without each mega flag on the tiny model: logits bitwise.
Run with the same INKLING_FAST list as the target config plus the mega flags;
each mega attribute is toggled off for the reference decoder.

megafix (REVIEW_MEGA L4):
- MEGA_ATTRS defaults to every mega attribute plus ``attnres`` (was mega_ctrl
  only); with TINY_WIDE=1 (D 1024) the attnres kernel engages, so this is its
  end-to-end tiny test.  The reference decoder then runs the xpass/keys path.
- Kernel uses are counted per (dtype, T0) and printed; the FP32 rows use no
  mega kernel (every gate needs BF16), so they only check the fallback path.
- BATCH=B (default 1): equal-length batch of B rows (mega_* gate off at B>1;
  attnres stays on up to its gate).
- SPEC=n (default 0): also compare FastDecoder.spec_step(L=n+1) rows against
  step rows with all flags on (FAST_VERIFY relies on verify == decode)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder
from mlx_vlm.models.inkling2 import mega_kernels as mk

# count kernel-path uses (trace time) so a silent fallback cannot pass
USED = {}
for _n in ("controllers", "rms_norm_cast", "dsconv_pair", "router_logits", "post_block", "qkv_sink_mask", "sconv_pair", "gate_oproj", "pre_block", "sdpa_exact"):
    if hasattr(mk, _n):
        def _w(*a, _f=getattr(mk, _n), _n=_n, **k):
            USED[_n] = USED.get(_n, 0) + 1
            return _f(*a, **k)
        setattr(mk, _n, _w)
from mlx_vlm.models.inkling2 import fast_kernels as _fk
for _n, _kw in (("qkv_clip", "zrow"), ("attn_res", "raw_partial")):
    def _w(*a, _f=getattr(_fk, _n), _n=_n, _kw=_kw, **k):
        if k.get(_kw):
            USED[_n + "." + _kw] = USED.get(_n + "." + _kw, 0) + 1
        return _f(*a, **k)
    setattr(_fk, _n, _w)

if os.environ.get("TINY_WIDE", "0") == "1":
    # head_dim 128 and D 1024: the dims qkvclip / attnres kernels require.
    _orig_cfg = tiny.tiny_config
    def _wide(**kw):
        c = _orig_cfg(**kw)
        c.hidden_size = 1024
        c.head_dim = c.swa_head_dim = 128
        return c
    tiny.tiny_config = _wide
if os.environ.get("TINY_REAL", "0") == "1":
    # megafix (REVIEW_MEGA2 F4): the real model's attention/residual dims
    # (D 2304 -> every gemv D%1024 tail branch, HQ 24, HK 8, head_dim 128,
    # d_rel 16, gate rank 64) on the tiny depth/expert count.
    _orig_cfg2 = tiny.tiny_config
    def _real(**kw):
        c = _orig_cfg2(**kw)
        c.hidden_size = 2304
        c.num_attention_heads = c.swa_num_attention_heads = 24
        c.num_key_value_heads = c.swa_num_key_value_heads = 8
        c.head_dim = c.swa_head_dim = 128
        import dataclasses
        lc = c.loop_config
        if isinstance(lc, dict):
            lc["gate_rank"] = 64
        else:
            c.loop_config = dataclasses.replace(lc, gate_rank=64)
        return c
    tiny.tiny_config = _real

ALL_MEGA = ("mega_ctrl,mega_norm,mega_dsconv,mega_moe,mega_attnres,mega_kvz,mega_post,"
            "mega_qsm,mega_sconv2,mega_oproj,mega_kvip,mega_pre,attnres")
MEGA = [a for a in os.environ.get("MEGA_ATTRS", ALL_MEGA).split(",") if a]
BATCH = int(os.environ.get("BATCH", "1"))
SPEC = int(os.environ.get("SPEC", "0"))
FAILS = []
STEPS = int(os.environ.get("STEPS", "40"))  # T0=1000 + 40 crosses p = cap-1 = 1023 and the growth
# REVIEW_MEGAPERF P1: module-level INKLING_FAST flags (fk.enabled() at call
# time, e.g. mega_rg2, mega_dpar, mega_qsm2, mega_oproj2) are dropped for the
# reference decoder too, so the test compares them against their parents.
MEGA_FLAGS = [a for a in os.environ.get("MEGA_FLAGS", "").split(",") if a]
_FLAGS_ON = _fk.FLAGS
for _f in MEGA_FLAGS:
    assert _f in _FLAGS_ON, f"{_f} not in INKLING_FAST"


def _set_flags(on):
    _fk.FLAGS = _FLAGS_ON if on else (_FLAGS_ON - set(MEGA_FLAGS))
    if MEGA_FLAGS:  # compiled-kernel caches keyed without the flag
        _fk._attnres_kernels.clear()
        mk._kern.clear()
for dtype in (mx.bfloat16, mx.float32):
    lm = tiny.build(dtype=dtype, window=1024, rel_extent=1024, n_floor=8192)
    # Real checkpoints share gate/up global scales (fusemoe path); tiny's are
    # random, so equalise them to exercise the fused decode MoE.
    for layer in lm.model.layers:
        sm = layer.mlp.switch_mlp
        g, u = sm.gate_proj, sm.up_proj
        if hasattr(g, "_scale_name") and g._scale_name == u._scale_name:
            u[u._scale_name] = g[g._scale_name]
    for T0 in [int(t) for t in os.environ.get("T0S", "5,40,1000,1100").split(",")]:
        ids = mx.random.randint(0, 500, (BATCH, T0))
        outs = []
        before = dict(USED)
        for on in (False, True):
            _set_flags(on)
            cache = lm.make_cache()
            lg = lm(inputs=ids, cache=cache).logits[:, -1]
            y = mx.argmax(lg, axis=-1).astype(mx.int32)[:, None]
            y0 = y
            dec = FastDecoder(lm, ring_extra=SPEC).from_cache(cache)
            for a in MEGA:
                assert getattr(dec, a), f"{a} not enabled (INKLING_FAST?)"
                if not on:
                    setattr(dec, a, False)
            ls = []
            for _ in range(STEPS):
                lg, y2 = dec.step(y)
                ls.append(lg)
                y = y2[:, None]
            mx.eval(ls)
            outs.append(mx.stack(ls))
            if on and SPEC:
                # verify rows vs decode rows: greedy tokens of the decode run as drafts
                toks = [y0] + [mx.argmax(l, axis=-1).astype(mx.int32)[:, None] for l in ls]
                cache = lm.make_cache()
                lm(inputs=ids, cache=cache)
                sdec = FastDecoder(lm, ring_extra=SPEC).from_cache(cache)
                for a in MEGA:
                    assert getattr(sdec, a)
                L, i, nb, nr = SPEC + 1, 0, 0, 0
                while i + L <= STEPS:
                    r = sdec.spec_step(mx.concatenate(toks[i:i + L], axis=1))
                    mx.eval(r["logits"], r["accepted"])
                    acc = r["accepted"].tolist()
                    sdec.observe(acc)
                    for j in range(min(acc) + 1):
                        nr += 1
                        nb += mx.array_equal(r["logits"][:, j], ls[i + j]).item()
                    if len(set(acc)) > 1:  # rows diverged: verify != decode argmax somewhere
                        nr += 1
                        break
                    i += acc[0] + 1
                print(f"{dtype} T0={T0} B={BATCH}: spec L={L} verify rows bitwise vs decode rows {nb}/{nr}")
                if nb != nr:
                    FAILS.append(f"spec {dtype} T0={T0}")
        d = mx.abs(outs[0] - outs[1]).max().item()
        used = {k: v - before.get(k, 0) for k, v in USED.items() if v != before.get(k, 0)}
        eq = mx.array_equal(outs[0], outs[1]).item()
        if not eq:
            FAILS.append(f"{dtype} T0={T0}")
        print(f"  kernel uses this case (trace-time): {used or 'NONE (fallback only)'}")
        print(f"{dtype} T0={T0} B={BATCH}: bitwise={eq} max|d|={d:.3e}")
print("MEGA_FD", "PASS" if not FAILS else f"FAIL {FAILS}")
