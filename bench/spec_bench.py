"""Greedy speculative-decode bench for Inkling 2 Mini's bundled DSpark drafter.

For each harness prompt, each requested drafter kind and each draft length n (verify block = n+1):
  e2e   : mlx-vlm's own generate_step(draft_model=..., draft_block_size=n+1, temperature=0) -> decode tok/s,
          accepted-per-round list (drafter.accept_lens), greedy tokens vs reference.
  split : a mirror of speculative/dflash.py::_dflash_rounds (greedy, B=1) with a sync after the draft and
          after the verify -> per-round draft / verify / walk+commit wall times. Syncs remove the
          draft/verify-build overlap, so split tok/s < e2e tok/s; use it for the cost split only.
  base  : generate_step without a drafter (same mlx-vlm path), for the speedup ratio.

Kinds: Inkling-2-Mini (inkling2_loop) always loads the inkling2_dspark drafter (round loop "dflash");
load_drafter(kind="mtp") is overridden to "dflash" (drafters/__init__.py resolve_drafter_kind). The
inkling_mtp drafter needs model.mtp.* tensors that this checkpoint does not have. Each requested kind is
loaded and its resolved kind recorded; duplicates are skipped.

Usage: python spec_bench.py [--ns 1,2,...,7] [--kinds dflash,mtp] [--only names] [--new 256]
                                [--modes e2e,split,base] [--tag name]
Writes one JSON line per (prompt, kind, n, mode) to $INKLING_RUNS/spec_bench.jsonl and prints a table.
Reference tokens: $INKLING_RUNS/ref/<name>.npz (make_ref.py). Missing refs are computed live with
the harness L=1 loop (ref_source="live").
Added (additive): every row gets `ptype` (chat/code/doc); after the table the script prints WORST and
BYPE lines and appends one summary row (mode="summary", prompt="(summary)") with, per (mode, kind, n): the worst prompt by
tok/s and by speedup vs base, and mean tok/s per prompt type. Chat is where spec loses."""
import argparse, json, os, statistics, time
from collections import Counter
import numpy as np
import mlx.core as mx
from common import *
import gpu_stats

OUT = os.path.join(RUNS, "spec_bench.jsonl")


# ---------------------------------------------------------------- pure helpers (unit-testable on the mini)
def first_divergence(got, ref):
    """Index of the first differing token over the common length, or None if the common prefix matches."""
    for i, (a, b) in enumerate(zip(got, ref)):
        if int(a) != int(b):
            return i
    return None


def accept_stats(accepted, n):
    """accepted: per-round accepted draft counts (0..n). Mean accepted drafts, mean tokens/round (+1 bonus),
    per-position acceptance rate, histogram {k: rounds}."""
    if not accepted:
        return {"rounds": 0}
    acc = [int(a) for a in accepted]
    hist = Counter(acc)
    # position i (1-based) accepted iff accepted >= i
    pos_rate = [sum(1 for a in acc if a >= i) / len(acc) for i in range(1, n + 1)]
    return {
        "rounds": len(acc),
        "mean_accepted": sum(acc) / len(acc),
        "mean_tokens_per_round": sum(acc) / len(acc) + 1,
        "full_block_rate": hist.get(n, 0) / len(acc),
        "pos_accept_rate": [round(r, 4) for r in pos_rate],
        "hist": {str(k): hist.get(k, 0) for k in range(n + 1)},
    }


def ptype_of(kind):
    """prompts.json kind -> chat / code / doc (doc_1p5k, doc_4k -> doc)."""
    return str(kind).split("_")[0]


def summarize_groups(rows):
    """rows: dicts with mode, kind (drafter kind or None for base), n, prompt, ptype, tok_s, speedup (or None).
    Per (mode, kind, n): worst prompt by tok/s, worst by speedup (if any), mean tok/s per prompt type, mean over all
    prompts. Pure function."""
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r.get("kind"), r["n"]), []).append(r)
    out = []
    for (mode, kind, n), rs in groups.items():
        w = min(rs, key=lambda r: r["tok_s"])
        sp = [r for r in rs if r.get("speedup") is not None]
        ws = min(sp, key=lambda r: r["speedup"]) if sp else None
        by = {}
        for r in rs:
            by.setdefault(r["ptype"], []).append(r["tok_s"])
        out.append(dict(mode=mode, kind=kind, n=n, prompts=len(rs),
                        mean_tok_s=sum(r["tok_s"] for r in rs) / len(rs),
                        worst_prompt=w["prompt"], worst_prompt_tok_s=w["tok_s"],
                        worst_speedup_prompt=ws["prompt"] if ws else None,
                        worst_speedup=ws["speedup"] if ws else None,
                        tok_s_by_type={t: sum(v) / len(v) for t, v in sorted(by.items())}))
    return out


def summarize_times(xs):
    if not xs:
        return None
    return {"mean_ms": 1e3 * sum(xs) / len(xs), "median_ms": 1e3 * statistics.median(xs)}


# ---------------------------------------------------------------- model-side helpers
def reference_tokens(model, processor, prompt, new):
    path = os.path.join(REF_DIR, prompt["name"] + ".npz")
    if os.path.exists(path):
        return [int(t) for t in np.load(path)["argmax"][:new]], "npz"
    ids = chat_ids(processor, prompt["text"])
    cache = model.language_model.make_cache()
    logits = prefill(model, ids, cache)
    toks = []
    for t in range(new):
        nxt = int(mx.argmax(logits[0]).item())
        toks.append(nxt)
        if t == new - 1:
            break
        logits = step_logits(model, mx.array([[nxt]]), cache)
    mx.clear_cache()
    return toks, "live"


def load_drafters(kinds):
    from mlx_vlm.speculative.drafters import load_drafter
    out, seen = [], set()
    for k in kinds:
        try:
            d, resolved = load_drafter(MODEL, None if k == "auto" else k)
        except Exception as e:  # recorded, not faked
            out.append((k, None, None, f"{type(e).__name__}: {e}"))
            continue
        if resolved in seen:
            out.append((k, resolved, None, f"duplicate of resolved kind {resolved!r}; skipped"))
            continue
        seen.add(resolved)
        out.append((k, resolved, d, None))
    return out


def run_e2e(model, ids, new, drafter=None, kind=None, block=None):
    """mlx-vlm generate_step; returns (tokens, decode_s after first token, accept_lens or None)."""
    from mlx_vlm.generate.ar import generate_step
    kw = dict(max_tokens=new, temperature=0.0)
    if drafter is not None:
        kw.update(draft_model=drafter, draft_kind=kind, draft_block_size=block)
    toks, t_first = [], None
    for tok, _ in generate_step(ids, model, None, None, **kw):
        toks.append(int(tok.item() if isinstance(tok, mx.array) else tok))
        if t_first is None:
            t_first = time.perf_counter()
    t_end = time.perf_counter()
    acc = list(getattr(drafter, "accept_lens", [])) if drafter is not None else None
    return toks, t_end - t_first, acc


def run_split(model, drafter, ids, new, block):
    """Mirror of _dflash_rounds (greedy, B=1) with explicit syncs for a per-round cost split.
    Uses the same mlx-vlm internals, so cache transactions / masked MoE / rollback are the real ones."""
    from mlx_vlm.speculative.common import generation_stream
    from mlx_vlm.speculative.dflash import (_dflash_verify_greedy, _reserve_dflash_target_cache)
    from mlx_vlm.speculative.common import _speculative_walk, _record_speculative_round
    from mlx_vlm.speculative.cache_state import commit_speculative_round, abort_speculative_round

    lm = model.language_model
    layer_ids = list(drafter.config.target_layer_ids)
    cache = lm.make_cache()
    # prefill with captures (harness chunking, 2048), same first token as the reference
    T, chunks = ids.shape[1], []
    for s in range(0, T, 2048):
        e = min(s + 2048, T)
        out = lm(inputs=ids[:, s:e], cache=cache, capture_layer_ids=layer_ids, skip_logits=e < T)
        mx.eval(out.hidden_states)
        chunks.append(out.hidden_states)
    hidden = mx.concatenate([mx.concatenate(parts, axis=1) for parts in zip(*chunks)], axis=-1)
    b = int(mx.argmax(out.logits[0, -1]).item())
    del chunks
    draft_cache = drafter.reset(model)
    with mx.stream(generation_stream):
        _reserve_dflash_target_cache(cache, block)
    sampler = lambda x: mx.argmax(x, axis=-1)
    toks, rounds = [b], []
    if os.environ.get("INKLING_FAST_VERIFY", "0") == "1":
        return _run_split_fast(lm, drafter, cache, hidden, draft_cache, b, new, block, layer_ids, sampler)
    t_start = time.perf_counter()
    while len(toks) < new:
        bs = min(block, new - len(toks) + 1)
        if bs <= 1:
            break
        t0 = time.perf_counter()
        draft = drafter.draft_block(b, hidden, draft_cache, bs, sampler, mx.int32)
        t1 = time.perf_counter(); mx.eval(draft); t2 = time.perf_counter()
        gdn = None
        try:
            with mx.stream(generation_stream):
                vin = mx.concatenate([mx.array([[b]], dtype=mx.int32), draft], axis=1)
                captured, gdn, target = _dflash_verify_greedy(lm, vin, cache, layer_ids, sampler)
                hidden = mx.concatenate(captured, axis=-1)
            t3 = time.perf_counter(); mx.eval(target, hidden); t4 = time.perf_counter()
            accepted, new_tokens = _speculative_walk(draft, target, new - len(toks))
            _record_speculative_round(drafter, accepted, bs - 1)
            if accepted < bs - 1:
                hidden = hidden[:, : accepted + 1, :]
            b = new_tokens[-1] if new_tokens else b
            with mx.stream(generation_stream):
                commit_speculative_round(lm, cache, gdn, accepted, bs)
            mx.eval(hidden)
            t5 = time.perf_counter()
        except BaseException:
            abort_speculative_round(gdn)
            raise
        rounds.append(dict(n=bs - 1, acc=accepted, draft_build=t1 - t0, draft_gpu=t2 - t1,
                           verify_build=t3 - t2, verify_wait=t4 - t3, commit=t5 - t4))
        toks.extend(new_tokens)
    return toks[:new], time.perf_counter() - t_start, rounds


def _run_split_fast(lm, drafter, cache, hidden, draft_cache, b, new, block, layer_ids, sampler):
    """INKLING_FAST_VERIFY=1: same round structure, target verify + accept + commit as ONE compiled
    FastDecoder graph (array-only rollback, expert-grouped NVFP4 MoE). The stock DSpark drafter still
    needs a host anchor and a variable-length hidden, so this keeps one sync per round (split mode
    syncs anyway); fast_decode.speculative_loop removes it once the drafter takes device inputs."""
    from mlx_vlm.models.inkling2.fast_decode import FastDecoder
    from mlx_vlm.speculative.common import _record_speculative_round
    dec = FastDecoder(lm, ring_extra=block - 1).from_cache(cache)
    del cache
    toks, rounds = [b], []
    t_start = time.perf_counter()
    while len(toks) < new:
        bs = min(block, new - len(toks) + 1)
        if bs <= 1:
            break
        t0 = time.perf_counter()
        draft = drafter.draft_block(b, hidden, draft_cache, bs, sampler, mx.int32)
        t1 = time.perf_counter(); mx.eval(draft); t2 = time.perf_counter()
        vin = mx.concatenate([mx.array([[b]], dtype=mx.int32), draft.astype(mx.int32)], axis=1)
        res = dec.spec_step(vin, captures=layer_ids)
        hidden = mx.concatenate(res["captures"], axis=-1)
        t3 = time.perf_counter(); mx.eval(res["emitted"], res["accepted"], hidden); t4 = time.perf_counter()
        accepted = int(res["accepted"][0].item())
        dec.observe([accepted])
        new_tokens = res["emitted"][0, : accepted + 1].tolist()[: new - len(toks)]
        _record_speculative_round(drafter, accepted, bs - 1)
        hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b
        t5 = time.perf_counter()
        rounds.append(dict(n=bs - 1, acc=accepted, draft_build=t1 - t0, draft_gpu=t2 - t1,
                           verify_build=t3 - t2, verify_wait=t4 - t3, commit=t5 - t4))
        toks.extend(new_tokens)
    return toks[:new], time.perf_counter() - t_start, rounds


def with_util(fn, *args):
    """Run fn(*args) under the GPU utilization sampler; returns (result, util summary)."""
    smp = gpu_stats.GpuSampler().start()
    try:
        r = fn(*args)
    finally:
        smp.stop()
    return r, gpu_stats.util_summary(smp.samples)


def spec_bw(n, st, tps):
    b = gpu_stats.spec_bytes_per_tok(n, st.get("mean_tokens_per_round"))
    if b is None:
        return None
    c = gpu_stats.ceiling()
    return dict(bytes_per_tok_gb=b, formula=gpu_stats.SPEC_FORMULA, drafter_gb=gpu_stats.DRAFTER_GB,
                verify_gb=gpu_stats.verify_gb(n + 1), ms_per_tok=1e3 / tps,
                floor_ms_per_tok=1e3 * b / c["ceiling_gbps"], **gpu_stats.eff(tps, b))


def record(row):
    os.makedirs(RUNS, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,2,3,4,5,6,7")
    ap.add_argument("--kinds", default="dflash,mtp")
    ap.add_argument("--modes", default="base,e2e,split")
    ap.add_argument("--only")
    ap.add_argument("--new", type=int, default=NEW_TOKENS)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    setup_memory()
    ns = [int(x) for x in a.ns.split(",")]
    modes = a.modes.split(",")
    model, processor = load_model()
    prompts = [p for p in load_prompts() if not a.only or p["name"] in a.only.split(",")]
    drafters = load_drafters(a.kinds.split(","))
    meta = dict(fork=fork_info(), env=env_flags(), tag=a.tag, ts=time.strftime("%FT%T"))
    for req, resolved, d, err in drafters:
        print(f"drafter kind={req!r} -> {resolved!r} {err or type(d).__module__}", flush=True)
        if err:
            record(dict(meta, mode="load", kind=req, resolved=resolved, error=err))
    live = [(req, res, d) for req, res, d, err in drafters if d is not None]
    if not live:
        raise SystemExit("no drafter loaded")

    # warm-up (kernel compile, fold) on the first prompt, short
    wids = chat_ids(processor, prompts[0]["text"])
    for _, res, d in live:
        for n in sorted({min(ns), max(ns)}):
            run_e2e(model, wids, 24, d, res, n + 1)
    run_e2e(model, wids, 24)
    mx.clear_cache()

    table, srows = [], []
    for p in prompts:
        pt = ptype_of(p["kind"])
        ids = chat_ids(processor, p["text"])
        ref, ref_src = reference_tokens(model, processor, p, a.new)
        base_tps = None
        if "base" in modes:
            (toks, dt, _), ub = with_util(run_e2e, model, ids, a.new)
            base_tps = (len(toks) - 1) / dt
            fd = first_divergence(toks, ref)
            srows.append(dict(mode="base", kind=None, n=0, prompt=p["name"], ptype=pt, tok_s=base_tps, speedup=None))
            record(dict(meta, mode="base", prompt=p["name"], ptype=pt, prompt_len=ids.shape[1], tok_s=base_tps,
                        first_div=fd, ref_source=ref_src, gpu_util=ub,
                        bw=dict(gpu_stats.decode_bytes(ids.shape[1]), **gpu_stats.eff(base_tps, gpu_stats.decode_bytes(ids.shape[1])["bytes_per_tok_gb"]))))
            table.append((p["name"], "base", 0, base_tps, None, fd))
            mx.clear_cache()
        for req, res, d in live:
            for n in ns:
                if "e2e" in modes:
                    (toks, dt, acc), ue = with_util(run_e2e, model, ids, a.new, d, res, n + 1)
                    tps = (len(toks) - 1) / dt
                    st = accept_stats(acc, n)
                    fd = first_divergence(toks, ref)
                    srows.append(dict(mode="e2e", kind=req, n=n, prompt=p["name"], ptype=pt, tok_s=tps,
                                      speedup=(tps / base_tps if base_tps else None)))
                    record(dict(meta, mode="e2e", prompt=p["name"], ptype=pt, prompt_len=ids.shape[1], kind=req,
                                resolved=res, n=n, tok_s=tps, speedup=(tps / base_tps if base_tps else None),
                                first_div=fd, n_tokens=len(toks), ref_source=ref_src, accept=st, gpu_util=ue,
                                bw=spec_bw(n, st, tps)))
                    table.append((p["name"], f"e2e/{res}", n, tps, st.get("mean_tokens_per_round"), fd))
                    mx.clear_cache()
                if "split" in modes:
                    (toks, dt, rounds), us = with_util(run_split, model, d, ids, a.new, n + 1)
                    st = accept_stats([r["acc"] for r in rounds], n)
                    fd = first_divergence(toks, ref)
                    parts = {k: summarize_times([r[k] for r in rounds])
                             for k in ("draft_build", "draft_gpu", "verify_build", "verify_wait", "commit")}
                    srows.append(dict(mode="split", kind=req, n=n, prompt=p["name"], ptype=pt, tok_s=(len(toks) - 1) / dt,
                                      speedup=(((len(toks) - 1) / dt) / base_tps if base_tps else None)))
                    record(dict(meta, mode="split", prompt=p["name"], ptype=pt, prompt_len=ids.shape[1], kind=req,
                                resolved=res, n=n, tok_s=(len(toks) - 1) / dt, first_div=fd,
                                ref_source=ref_src, accept=st, per_round=parts, gpu_util=us,
                                bw=spec_bw(n, st, (len(toks) - 1) / dt)))
                    table.append((p["name"], f"split/{res}", n, (len(toks) - 1) / dt,
                                  st.get("mean_tokens_per_round"), fd))
                    print("   split ms/round: " + " ".join(
                        f"{k}={v['median_ms']:.1f}" for k, v in parts.items() if v), flush=True)
                    mx.clear_cache()
                print(f"{p['name']} {res} n={n} done", flush=True)

    print(f"\n{'prompt':<14}{'mode':<14}{'n':>3}{'tok/s':>9}{'tok/rnd':>9}{'first_div':>10}")
    for name, mode, n, tps, tpr, fd in table:
        print(f"{name:<14}{mode:<14}{n:>3}{tps:>9.1f}{(tpr or 0):>9.2f}{str(fd):>10}")
    groups = summarize_groups(srows)
    print()
    for g in groups:
        sp = f", worst speedup {g['worst_speedup']:.2f}x on {g['worst_speedup_prompt']}" if g["worst_speedup"] is not None else ""
        print(f"WORST {g['mode']}/{g['kind']} n={g['n']}: {g['worst_prompt']} {g['worst_prompt_tok_s']:.1f} tok/s{sp}; "
              f"mean {g['mean_tok_s']:.1f}")
        print(f"BYTYPE {g['mode']}/{g['kind']} n={g['n']}: " + " ".join(f"{t}={v:.1f}" for t, v in g["tok_s_by_type"].items()) + " tok/s")
    record(dict(meta, mode="summary", prompt="(summary)", groups=groups))
    print(f"\nrows appended to {OUT}")


if __name__ == "__main__":
    main()
