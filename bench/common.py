"""Shared helpers for the benchmark harness.

Environment: INKLING_MODEL (path to the released Inkling 2 Mini weights), INKLING_RUNS (output dir, default ./runs),
INKLING_PROMPTS (prompts JSON, default prompts.json next to this file; build one with build_prompts.py)."""
import json, os, subprocess, sys, time
import numpy as np
import mlx.core as mx

# count custom-kernel calls by name BEFORE any mlx_vlm import (see engaged.py). Additive, silent.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engaged
    engaged.install()
except Exception as _e:  # never let the probe break a run
    engaged = None
    print("engaged probe disabled:", _e, file=sys.stderr)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser(os.environ.get("INKLING_MODEL", "Inkling-2-Mini"))  # path to the released weights
FORK = os.environ.get("FORK", ".")  # git checkout recorded in result rows (sha + dirty flag)
RUNS = os.path.abspath(os.environ.get("INKLING_RUNS", "runs"))
REF_DIR = os.path.join(RUNS, "ref")
FAST_DECODE = os.environ.get("INKLING_FAST_DECODE", "0") == "1"

# Memory guard (hard): refuse any job whose estimated peak exceeds 55 GB.
# peak ~= 25 GB weights + B * (0.16 GB sliding state + 32 KB/token * ctx bf16 KV) + 3 GB headroom.
MEM_CAP_GB = 55.0
def mem_estimate_gb(B, ctx, chunk=2048):
    # + ONE prefill transient (assumes one prefill in flight at a time): fp32 attention scores chunk x ctx x 24 heads
    #   (longctx measured at 128K, B=1: 56 GB @chunk 2048, 43 GB @chunk 1024). Long-context prefill should use a smaller chunk.
    return 25.0 + B * (0.16 + 32e3 * ctx / 1e9) + chunk * ctx * 24 * 4 / 1e9 + 3.0
def mem_guard(B, ctx, what="", chunk=2048):
    """Raise SystemExit(4) if B sessions x ctx tokens would exceed MEM_CAP_GB. Call before batch/long-ctx work."""
    est = mem_estimate_gb(B, ctx, chunk)
    if est > MEM_CAP_GB:
        msg = f"MEM GUARD: refusing {what} B={B} ctx={ctx} chunk={chunk}: est peak {est:.1f} GB > {MEM_CAP_GB} GB"
        print(msg, flush=True); sys.stderr.write(msg + "\n"); raise SystemExit(4)
    return est
FD_COMPILED = os.environ.get("INKLING_FD_COMPILED", "1") != "0"  # 0: run FastDecoder.step uncompiled
_FAST = {}  # id(stock cache) -> FastDecoder (harness keeps the cache alive per prompt)
FULL_STEPS = (0, 63, 127, 255)
TOPK = 64
NEW_TOKENS = 256


def setup_memory():
    mx.set_memory_limit(60 * 1024**3)
    w = os.environ.get("INKLING_WIRED_GB")  # opt-in: mx.set_wired_limit
    if w:
        mx.set_wired_limit(int(float(w) * 1024**3))


def fork_info():
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", FORK, *a], text=True).strip()
        except Exception:
            return "unknown"
    return {"sha": g("rev-parse", "--short", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def env_flags():
    return {k: v for k, v in os.environ.items() if k.startswith(("INKLING", "MLX_"))}


def load_model():
    from mlx_vlm.utils import load
    return load(MODEL)


def tokenizer_of(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


PROMPTS = os.environ.get("INKLING_PROMPTS", os.path.join(HERE, "prompts.json"))


def load_prompts():
    """{"prompts": [{"name", "kind" (chat/code/doc_*), "text"}, ...]} from INKLING_PROMPTS."""
    with open(PROMPTS) as f:
        return json.load(f)["prompts"]


def chat_ids(processor, text):
    """Chat-template a single user turn -> mx.array [1, T]. add_special_tokens=False
    because the template is expected to emit BOS itself; token count is recorded in refs."""
    tok = tokenizer_of(processor)
    s = tok.apply_chat_template([{"role": "user", "content": text}],
                                add_generation_prompt=True, tokenize=False)
    ids = tok.encode(s, add_special_tokens=False)
    return mx.array(ids)[None]


def raw_ids(processor, n):
    """~n tokens of plain (un-templated) doc text, for perf runs."""
    tok = tokenizer_of(processor)
    docs = [p["text"] for p in load_prompts() if p["kind"].startswith("doc")]
    ids = []
    while len(ids) < n:
        for d in docs:
            ids += tok.encode(d, add_special_tokens=False)
    return mx.array(ids[:n])[None]


def prefill(model, ids, cache, step=2048):
    """Prefill all of ids; returns last-position logits [1, V] (float32)."""
    if engaged is not None: engaged.set_phase("prefill")  # kernel counters: prefill vs decode phase
    try:
        return _prefill(model, ids, cache, step)
    finally:
        if engaged is not None: engaged.set_phase("decode")


def _prefill(model, ids, cache, step=2048):
    lm = model.language_model
    T = ids.shape[1]
    _FAST.clear()  # free the previous prompt's decoder before prefill (it skewed the 1k prefill timing)
    import gc; gc.collect()  # FastDecoder self-references via its compiled step; clear() alone doesn't free it
    for s in range(0, T, step):
        e = min(s + step, T)
        if e == T:
            logits = lm(inputs=ids[:, s:e], cache=cache).logits[:, -1]
            mx.eval(logits)
            if FAST_DECODE:  # convert the stock cache into the compiled decoder's state
                from mlx_vlm.models.inkling2.fast_decode import FastDecoder, Unsupported
                try:
                    _FAST[id(cache)] = FastDecoder(lm).from_cache(cache)
                except Unsupported as e:  # sharded / too long: stock step_logits
                    print("FastDecoder unsupported, stock decode:", e, flush=True)
            return logits
        out = lm(inputs=ids[:, s:e], cache=cache, return_hidden=True, skip_logits=True)
        mx.eval(out.hidden_states[0])
        mx.clear_cache()


def step_logits(model, tok_id, cache):
    """One L=1 step. tok_id: mx.array [1,1]. Returns logits [1, V].
    INKLING_FAST_DECODE=1: runs the mx.compile'd FastDecoder step built in prefill()."""
    dec = _FAST.get(id(cache)) if FAST_DECODE else None
    if dec is not None:
        return dec.step(tok_id, compiled=FD_COMPILED)[0]
    out = model.language_model(inputs=tok_id, cache=cache)
    return out.logits[:, -1]


def logprob_stats(logits):
    """logits [1,V] -> (lp [V] f32, lse scalar). Lazy."""
    l = logits[0].astype(mx.float32)
    lse = mx.logsumexp(l)
    return l - lse, lse
