"""End-to-end decode A/B on a small random Inkling 2 (real module code).

Runs the same seeded random model in subprocesses with INKLING_FAST unset and
set, teacher-forced on the stock run's greedy tokens, and compares per-step
logits. Model: hidden 2304, 24 heads, 8 kv, hd 128, d_rel 16, 18 layers with a
2-pass loop over layers 6-11 (30 visits), window 64 (so the ring wraps during
decode), 16 experts top-8 (NVFP4 + scaled global scale; last layer MXFP8),
vocab 8192. Usage: python test_e2e.py [flags ...]   (default: all flag sets)
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("E2E_OUT", os.path.join(os.environ.get("TMPDIR", "/tmp"), "inkling2_fast_e2e"))
PREFILL, STEPS = 90, 60


def build():
    sys.path.insert(0, HERE)
    from _load import load
    import mlx.core as mx
    import mlx.nn as nn

    cfgm = load("mlx_vlm.models.inkling2.config")
    lang = load("mlx_vlm.models.inkling2.language")
    sw = load("mlx_vlm.models.switch_layers")
    nv = load("mlx_vlm.quantization.nvfp4")
    kw = dict(
        model_type="inkling2_loop_text", hidden_size=2304, num_hidden_layers=18,
        vocab_size=8192, unpadded_vocab_size=8000, num_attention_heads=24,
        num_key_value_heads=8, swa_num_attention_heads=24, swa_num_key_value_heads=8,
        sliding_window_size=64, rel_extent=48, intermediate_size=256,
        n_routed_experts=16, num_experts_per_tok=8, log_scaling_n_floor=40,
        loop_config={"core_start": 6, "core_end": 12, "num_passes": 2, "gate_rank": 64},
    )
    # E2E_CFG='{"intermediate_size": 1152, ...}' overrides (json)
    kw.update(json.loads(os.environ.get("E2E_CFG", "{}")))
    cfg = cfgm.TextConfig(**kw)
    mx.random.seed(1234)
    model = lang.LanguageModel(cfg)
    # random weights, deterministic order
    from mlx.utils import tree_flatten, tree_unflatten
    params = tree_flatten(model.parameters())
    new = []
    for i, (k, v) in enumerate(params):
        key = mx.random.key(i)
        if k.endswith("norm.weight") or "layernorm" in k:
            val = 1 + 0.1 * mx.random.normal(v.shape, key=key)
        elif "e_score_correction_bias" in k:
            val = 0.01 * mx.random.normal(v.shape, key=key)
        else:
            fan = v.shape[-1] if v.ndim > 1 else 1
            val = mx.random.normal(v.shape, key=key) * (1.0 / fan ** 0.5)
            if "rel_proj" in k or "query" in k or "message_queries" in k:
                val = val * 4
        dt = mx.float32 if ("query" in k or "correction" in k or "message_queries" in k) else mx.bfloat16
        new.append((k, val.astype(dt)))
    model.update(tree_unflatten(new))
    # quantize experts
    for li, layer in enumerate(model.model.layers):
        smlp = layer.mlp.switch_mlp
        mxfp8 = li == len(model.model.layers) - 1
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = smlp[name]
            E, O, I = lin.weight.shape
            q = sw.QuantizedSwitchLinear(I, O, E, bias=False, group_size=32 if mxfp8 else 16,
                                         bits=8 if mxfp8 else 4, mode="mxfp8" if mxfp8 else "nvfp4")
            w = lin.weight.astype(mx.float32) * 0.5
            q.weight, q.scales, *_ = mx.quantize(w, group_size=q.group_size, bits=q.bits, mode=q.mode)
            if not mxfp8:
                key = mx.random.key(1000 + li * 3 + (name == "down_proj"))
                gs = (0.5 + mx.random.uniform(shape=(E,), key=key)) / 2688.0
                q = nv.ScaledQuantizedSwitchLinear(q, gs.astype(mx.float32), "weight_scale_2")
            smlp[name] = q
    mx.eval(model.parameters())
    model.eval()  # mlx-vlm load() does this; stock kernel paths require it
    return model, cfg


def run(tag):
    import mlx.core as mx
    model, cfg = build()
    rng = np.random.default_rng(7)
    batch = int(os.environ.get("E2E_BATCH", "1"))  # equal-length rows, no padding
    prompt = mx.array(rng.integers(0, 8000, size=(batch, PREFILL)))
    cache = model.make_cache()
    tokens_path = os.path.join(OUT, "tokens.json")
    forced = json.load(open(tokens_path)) if tag != "stock" else None
    logits_all, toks = [], []
    out = model(prompt, cache=cache).logits[:, -1]
    for s in range(STEPS):
        mx.eval(out)
        lg = np.array(out.astype(mx.float32))  # [batch, V]
        logits_all.append(lg)
        t = [int(v) for v in lg.argmax(-1)] if forced is None else forced[s]
        toks.append([int(v) for v in lg.argmax(-1)])
        out = model(mx.array(t)[:, None], cache=cache).logits[:, -1]
    np.save(os.path.join(OUT, f"{tag}.npy"), np.stack(logits_all))
    if tag == "stock":
        json.dump(toks, open(tokens_path, "w"))
    print(f"{tag}: peak mem {mx.get_peak_memory() / 1e9:.2f} GB")


def compare(tag):
    a = np.load(os.path.join(OUT, "stock.npy")).astype(np.float64)
    b = np.load(os.path.join(OUT, f"{tag}.npy")).astype(np.float64)
    def lsm(x):
        m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
    pa, pb = lsm(a), lsm(b)
    kl = (np.exp(pa) * (pa - pb)).sum(-1)
    same = (a.argmax(-1) == b.argmax(-1)).mean()  # over steps x rows
    bit = np.array_equal(a, b)
    print(f"{tag:28s} bitwise={bit} max|dlogit|={np.abs(a - b).max():.3g} argmax_agree={same:.3f} "
          f"meanKL={kl.mean():.2e} maxKL={kl.max():.2e}")
    return kl.mean(), same


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run(sys.argv[2]); sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    sets = sys.argv[1:] or ["head", "mask_sink", "mask_ring", "mask", "glue", "fuse", "fusemoe", "head,mask,glue,fuse,fusemoe"]
    py = sys.executable
    def worker(tag, flags):
        env = dict(os.environ); env.pop("INKLING_FAST", None)
        if flags: env["INKLING_FAST"] = flags
        subprocess.run([py, __file__, "--worker", tag], env=env, check=True)
    worker("stock", "")
    worker("stock_rerun", "")
    compare("stock_rerun")
    for f in sets:
        tag = f.replace(",", "+")
        worker(tag, f)
        compare(tag)
