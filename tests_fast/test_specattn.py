"""specfast-attn: FP32 unit + end-to-end tests for spec_attn.py (INKLING_SPECFAST sdpa/post/pre).

A1 sdpa_rows vs the stock verify composition (_rel_slot_mask + sink column + mx.fast SDPA over the
   buffer with its zero sink row + output gate): sliding ring (partly filled / wrapped), global
   (rel_extent < distance -> bias 0), NB = 1 and NB > 1 (combine launch), L = 2..8.
A2 tail_rows post/pre vs the stock module sequence (ShortConvolution module, _attn_res_keys,
   RMSNorm), boundary / no boundary, norm / hidden, on real tiny-model modules.
B  speculative_loop on the tiny model (FP32), drafts corrupted so every accept length occurs:
   tokens identical to stock greedy, verify rows (accepted prefix) vs stock decode logits, and
   the three kernels must have engaged (call counters). TINY_WIDE=1: D 1024, head_dim 128.
Run with MLX_ENABLE_TF32=0 and INKLING_SPECFAST=all (+ the INKLING_FAST stack for B)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
if os.environ.get("TINY_WIDE", "0") == "1":
    _orig_cfg = tiny.tiny_config

    def _wide(**kw):
        c = _orig_cfg(**kw)
        c.hidden_size = 1024
        c.head_dim = c.swa_head_dim = 128
        return c
    tiny.tiny_config = _wide
import numpy as np
import mlx.core as mx
from mlx_vlm.models.inkling2 import spec_attn as sa
from mlx_vlm.models.inkling2 import fast_decode as fd

mx.set_memory_limit(4 * 1024**3)
F32 = mx.float32
OK = True
TOL = float(os.environ.get("SA_TOL", "2e-5"))


def check(name, a, b, tol=TOL):
    global OK
    d = float(mx.max(mx.abs(a.astype(F32) - b.astype(F32))).item())
    good = d <= tol and not np.isnan(d)
    OK &= good
    print(f"  {'ok ' if good else 'BAD'} {name}: max|d|={d:.2e}")


USED = {}
for _n in ("sdpa_rows", "tail_rows", "kvsc_clip"):
    def _w(*a, _f=getattr(sa, _n), _n=_n, **k):
        key = _n if _n != "tail_rows" else f"tail{a[0]}"
        if _n == "tail_rows" and a[0] == 1 and getattr(a[6], "ndim", 1) == 2:
            key = "tail1_nq"
        USED[key] = USED.get(key, 0) + 1
        return _f(*a, **k)
    setattr(sa, _n, _w)


# ---------------------------------------------------------------- A1 sdpa
def a1(L, HQ, HK, DH, DR, R, window, ring, S, q0, block=None, dtype=F32, seed=0):
    key = mx.random.key(seed)
    ks = mx.random.split(key, 8)
    sliding = window > 0
    attn = types.SimpleNamespace(
        rel_proj=mx.random.normal((DR, R), key=ks[0]).astype(dtype) * 0.5,
        sliding=window, is_sliding=sliding, rel_extent=(window if sliding else R))
    q = mx.random.normal((1, L, HQ, DH), key=ks[1]).astype(dtype)
    r = mx.random.normal((1, L, HQ, DR), key=ks[2]).astype(dtype) * 0.3
    sinks = mx.random.normal((1, L, HQ), key=ks[3]).astype(dtype)
    gate = mx.random.normal((1, L, HQ * DH), key=ks[4]).astype(dtype)
    nslot = ring if sliding else S
    kb = mx.random.normal((1, HK, nslot + 1, DH), key=ks[5]).astype(dtype)
    vb = mx.random.normal((1, HK, nslot + 1, DH), key=ks[6]).astype(dtype)
    row = mx.arange(nslot + 1)[None, None, :, None]
    kb = mx.where(row == nslot, 0, kb).astype(dtype)  # zero sink row
    vb = mx.where(row == nslot, 0, vb).astype(dtype)
    qp = (q0 + mx.arange(L, dtype=mx.int32))[None]
    if sliding:
        sp = np.full((1, ring), -1, np.int32)
        for p in range(q0 + L):
            sp[0, p % ring] = p
        slot_pos = mx.array(sp)
    else:
        ar = mx.arange(S, dtype=mx.int32)[None]
        slot_pos = mx.where(ar <= qp[:, -1:], ar, -1)
    scale = DH ** -0.5
    proj = attn.rel_proj.astype(dtype)
    mask = fd._rel_slot_mask(r, proj, slot_pos, qp, attn.sliding, attn.rel_extent, dtype)
    mask = mx.concatenate([mask, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
    ref = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3), kb, vb, scale=scale, mask=mask)
    ref = ref.transpose(0, 2, 1, 3).reshape(1, L, -1)
    ref = (ref.astype(F32) * mx.sigmoid(gate.astype(F32))).astype(dtype)
    sa.prepare(types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)]))
    old = sa.SDPA_BLOCK
    if block:
        sa.SDPA_BLOCK = block
    try:
        got = sa.sdpa_rows(attn, q, kb, vb, r, slot_pos if sliding else None, qp, sinks, gate,
                           ring, scale, (S - 1) if not sliding else 1 << 20)
    finally:
        sa.SDPA_BLOCK = old
    tag = (f"sdpa L={L} H={HQ}/{HK} DH={DH} {'swa W=%d ring=%d' % (window, ring) if sliding else 'global S=%d R=%d' % (S, R)}"
           f" q0={q0}{' block=%d' % block if block else ''} {dtype}")
    check(tag, got, ref, TOL if dtype == F32 else 3e-2)


print("A1 sdpa_rows vs stock mask + SDPA + gate")
for L in (2, 5, 8):
    a1(L, 6, 2, 128, 16, 16, 16, 16 + L - 1, None, 3)          # sliding, not yet full
    a1(L, 6, 2, 128, 16, 16, 16, 16 + L - 1, None, 57)         # sliding, wrapped
    a1(L, 4, 2, 64, 16, 32, 0, 0, 300, 120)                    # global, dist >= R -> 0
a1(5, 24, 8, 128, 16, 1024, 1024, 1028, None, 1500)            # real dims, sliding full
a1(5, 24, 8, 128, 16, 1024, 0, 0, 4100, 3990)                  # real dims, global 4k: NB = 4
a1(5, 24, 8, 128, 16, 1024, 0, 0, 1023, 500, block=128)        # forced NB > 1 at 512 ctx
a1(5, 24, 8, 128, 16, 1024, 1024, 1028, None, 4000, block=300)  # sliding NB > 1
a1(5, 24, 8, 128, 16, 1024, 1024, 1028, None, 1500, dtype=mx.bfloat16)

# ---------------------------------------------------------------- A2 tails
print("A2 tail_rows (post / pre) vs stock module sequence")
lm = tiny.build(dtype=F32)
D = lm.model.layers[0].attn_sconv.conv.weight.shape[0]


def rn(x):
    return mx.fast.rms_norm(x.astype(F32), None, 1e-6)


for li, L, nk, boundary, use_norm in ((0, 5, 1, False, True), (1, 2, 2, True, True),
                                      (2, 8, 3, False, False), (3, 5, 2, True, False)):
    layer = lm.model.layers[li]
    key = mx.random.key(100 + li)
    ks = mx.random.split(key, 6)
    ac, mc = layer.attn_sconv, layer.mlp_sconv
    state = mx.random.normal((1, 3, D), key=ks[0])
    attn = mx.random.normal((1, L, D), key=ks[1])
    x = mx.random.normal((1, L, D), key=ks[2])
    bkeys = [rn(mx.random.normal((1, L, 1, D), key=k)) for k in mx.random.split(ks[3], nk)]
    q = layer.mlp_input.query
    pn = layer.post_attention_layernorm
    # stock (FastDecoder L>1 sequence)
    conv = [None] * 4
    conv[ac.conv_idx] = state
    s = ac(attn, conv, None)
    residual = s.astype(F32) if boundary else s.astype(F32) + x.astype(F32)
    keys = bkeys + ([rn(x[..., None, :])] if boundary else [])
    hid = fd._attn_res_keys(keys, residual, q, True)
    r = sa.tail_rows(0, ac, state, attn, x, bkeys, q, boundary=boundary,
                     norm=pn if use_norm else None, want_hid=not use_norm)
    tag = f"post layer={li} L={L} nk={nk} boundary={boundary} norm={use_norm}"
    check(tag + " ctx", r["ctx"], mx.concatenate([state, attn.astype(F32)], axis=1), 0.0)
    check(tag + " residual", r["res"], residual)
    if boundary:
        check(tag + " bkey", r["bkey"], keys[-1])
    if use_norm:
        check(tag + " hn", r["hn"], pn(hid[..., 0, :]).astype(x.dtype))
    else:
        check(tag + " hidden", r["hid"], hid)
    # pre: mlp sconv + residual, then the next attention_input depth attention
    nl = lm.model.layers[li + 1]
    ai = nl.attention_input
    if ai is None or ai.query.ndim != 1:  # multi-query inputs stay on the stock path
        ai = layer.mlp_input
    h = mx.random.normal((1, L, D), key=ks[4])
    conv[mc.conv_idx] = state
    s2 = mc(h, conv, None)
    xn = (residual + s2.astype(F32)).astype(x.dtype)
    mixed = fd._attn_res_keys(bkeys, xn, ai.query, True)
    nn_ = getattr(nl, "input_layernorm", None) or layer.post_attention_layernorm
    r2 = sa.tail_rows(1, mc, state, h, residual, bkeys, ai.query,
                      norm=nn_ if use_norm else None, want_hid=True, out_dtype=x.dtype)
    tag = f"pre  layer={li} L={L} nk={nk} norm={use_norm}"
    check(tag + " ctx", r2["ctx"], mx.concatenate([state, h.astype(F32)], axis=1), 0.0)
    check(tag + " x_next", r2["res"], xn)
    check(tag + " mixed", r2["hid"], mixed)
    if use_norm:
        check(tag + " hn", r2["hn"], nn_(mixed[..., 0, :]).astype(x.dtype))

print("A3 depth attention only (mode 2), 1 and 3 queries; pre with 3 queries")
for nq in (1, 3):
    for L, nk in ((5, 2), (8, 4), (2, 1)):
        ks = mx.random.split(mx.random.key(200 + nq * 10 + L), 4)
        qd = 0.3 * mx.random.normal((D,) if nq == 1 else (nq, D), key=ks[0])
        part = mx.random.normal((1, L, D), key=ks[1])
        bk = [rn(mx.random.normal((1, L, 1, D), key=k)) for k in mx.random.split(ks[2], nk)]
        ref = fd._attn_res_keys(bk, part, qd, True)
        got = sa.tail_rows(2, None, None, part, None, bk, qd)["hid"]
        check(f"dattn nq={nq} L={L} nk={nk}", got, ref)
        if nq == 3:
            mc = lm.model.layers[0].mlp_sconv
            st = mx.random.normal((1, 3, D), key=ks[3])
            h = mx.random.normal((1, L, D), key=ks[3])
            conv = [None] * 4
            conv[mc.conv_idx] = st
            xn = (part + mc(h, conv, None).astype(F32)).astype(F32)
            got = sa.tail_rows(1, mc, st, h, part, bk, qd, out_dtype=F32)
            check(f"pre nq=3 L={L} nk={nk} x_next", got["res"], xn)
            check(f"pre nq=3 L={L} nk={nk} mixed", got["hid"], fd._attn_res_keys(bk, xn, qd, True))

print("A4 kvsc (k/v sconv + qkv clip) vs ShortConvolution modules + fk.qkv_clip")
from mlx_vlm.models.inkling2 import fast_kernels as fk
for li, layer in enumerate(lm.model.layers):
    at = layer.self_attn
    if not at.is_sliding:
        continue
    for L in (2, 5, 8):
        if not sa.kvsc_supported(at, lm.config if hasattr(lm, "config") else lm.args, mx.zeros((1, L, D)),
                                 at.k_sconv, at.v_sconv, mx.zeros((1, 3, at.n_kv * at.head_dim)),
                                 mx.zeros((1, 3, at.n_kv * at.head_dim)), 1):
            print(f"  (kvsc unsupported at these dims: head_dim={at.head_dim})")
            break
        cfgx = lm.config if hasattr(lm, "config") else lm.args
        ks_ = mx.random.split(mx.random.key(300 + L), 6)
        C = at.n_kv * at.head_dim
        q = mx.random.normal((1, L, at.n_heads * at.head_dim), key=ks_[0])
        k = mx.random.normal((1, L, C), key=ks_[1])
        v = mx.random.normal((1, L, C), key=ks_[2])
        r = mx.random.normal((1, L, at.n_heads * cfgx.d_rel), key=ks_[3])
        kst = mx.random.normal((1, 3, C), key=ks_[4])
        vst = mx.random.normal((1, 3, C), key=ks_[5])
        conv = [None] * 4
        conv[at.k_sconv.conv_idx], conv[at.v_sconv.conv_idx] = kst, vst
        ks2, vs2 = at.k_sconv(k, conv, None), at.v_sconv(v, conv, None)
        ref = fk.qkv_clip(q, ks2, vs2, r, at, cfgx, None)
        got = sa.kvsc_clip(q, k, v, r, at, cfgx, at.k_sconv, at.v_sconv, kst, vst)
        for nm, g, rf in zip(("q", "k", "v", "r"), got[:4], ref):
            check(f"kvsc layer={li} L={L} {nm}", g, rf)
        check(f"kvsc layer={li} L={L} kctx", got[4], mx.concatenate([kst, k], axis=1), 0.0)
        check(f"kvsc layer={li} L={L} vctx", got[5], mx.concatenate([vst, v], axis=1), 0.0)
    break

# ---------------------------------------------------------------- B end to end
print("B speculative_loop FP32, verify rows vs stock decode",
      "(TINY_WIDE)" if os.environ.get("TINY_WIDE") == "1" else "")
from mlx_vlm.models.inkling2.fast_decode import FastDecoder, speculative_loop
USED.clear()
rng = np.random.default_rng(11)
ids = mx.array(rng.integers(1, 500, size=(1, 12)).astype(np.int32))
c = lm.make_cache()
lg = lm(inputs=ids, cache=c).logits[:, -1]
first = mx.argmax(lg, -1).astype(mx.int32)
ref, ref_logits, y = [], [], first
T = int(os.environ.get("SA_TOKENS", "64"))
for t in range(T + 24):
    lg = lm(inputs=y[:, None], cache=c).logits[:, -1]
    y = mx.argmax(lg, -1).astype(mx.int32)
    mx.eval(y)
    ref.append(int(y.item()))
    ref_logits.append(np.array(lg[0]))
seq = np.zeros((1, 12 + T + 64), np.int32)
seq[0, 12] = int(first.item())
seq[0, 13:13 + len(ref)] = ref
seq_d = mx.array(seq)
for n in (1, 4, 7):
    def drafter(rs):
        p = rs["q_pos"][:, None] + 1 + mx.arange(n, dtype=mx.int32)[None]
        d = mx.take_along_axis(seq_d, mx.clip(p, 0, seq.shape[1] - 1), axis=1)
        bad = ((p * 40503 + 7) % 7) < 2
        return mx.where(bad, (d + 1) % 500, d)
    c2 = lm.make_cache()
    lm(inputs=ids, cache=c2)
    dec = FastDecoder(lm, buckets=(64, 128, 256), ring_extra=7).from_cache(c2)
    hist, logs = [], []
    toks, rounds = speculative_loop(dec, first, drafter, n, T,
                                    on_round=lambda k, res: (hist.append(res["accepted"]),
                                                             logs.append(res["logits"])))
    got = np.array(toks)[0].tolist()
    same = got == ref[:T]
    vdiff, P = 0.0, 12
    for k in range(len(hist)):
        lgk = np.array(logs[k].astype(F32))
        a = int(np.array(hist[k]).reshape(-1)[0])
        for j in range(a + 1):
            t = P + j - 12
            if t < len(ref_logits):
                vdiff = max(vdiff, float(np.abs(lgk[0, j] - ref_logits[t]).max()))
        P += a + 1
    good = same and vdiff < 1e-4
    OK &= good
    print(f"  {'ok ' if good else 'BAD'} n={n}: tokens identical={same} rounds={rounds} "
          f"verify rows max|d|={vdiff:.2e}")
print("  kernel calls:", USED)
sf = {t for t in os.environ.get("INKLING_SPECFAST", "").split(",") if t}
need = []
if sf & {"all", "sdpa"}:
    need.append("sdpa_rows")
if sf & {"all", "post"}:
    need.append("tail0")
if sf & {"all", "pre"}:
    need.append("tail1")
if sf & {"all", "dattn"}:
    need.append("tail2")
if sf & {"all", "kvsc"} and lm.model.layers[0].self_attn.head_dim == 128:
    need.append("kvsc_clip")
for k in need:
    if not USED.get(k):
        OK = False
        print(f"  BAD kernel {k} never engaged")
print("TEST_SPECATTN", "PASS" if OK else "FAIL")
assert OK
