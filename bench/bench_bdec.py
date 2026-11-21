"""Batched greedy decode bench. Same-session comparison of decode paths at B>1.

Rows: B distinct raw-doc slices (offset 97*i) of --ctx tokens (equal length), or with --lp lengths ctx-37*(i%5)
(left-padded as batch_generate does). One batched prefill per (mode, B), then --new greedy L=1 steps, tokens fed
back on device (no host round trip). Pipelined like mlx-lm generate: step t+1 is enqueued (async_eval) before the
host waits on step t; the host timestamp when step t's tokens are ready gives per-step latency.

Modes (comma list, run in the order given, repeated --reps times interleaved):
  stock : model.language_model(inputs=y, cache=BatchKVCache)   (the path batch_generate uses; INKLING_FAST flags apply)
  fdb   : fd with INKLING_BATCHFAST=--bf set while the decoder is built (same-session base vs change; fd runs with it unset)
  fd    : FastDecoder(lm).from_cache(cache).step(y)             (compiled; refuses padded rows unless
                                                                  INKLING_FAST_DECODE_PADDED=1 -> row says refused)
  bg    : mlx_vlm batch_generate on prompt 0 x B (the old bench.py number; sampler/detokenizer included)
Reported per (mode, B): aggregate tok/s = B*steps/wall, per-stream tok/s = steps/wall, p50/p95/max step ms
(= per-token latency of every stream), prefill s. Median over reps of the aggregate; spread_pct.
--route-stats: (stock mode, uncompiled) unique experts per MoE layer per step at each B (hooks language._route;
  run with no INKLING_FAST flags so the stock router is used).
Appends rows to $INKLING_RUNS/bench_bdec.jsonl."""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
from common import *

ROUTES = []


def rows_for(processor, B, T, lp):
    tok = tokenizer_of(processor)
    docs = [p["text"] for p in load_prompts() if p["kind"].startswith("doc")]
    ids = []
    for d in docs:
        ids += tok.encode(d, add_special_tokens=False)
    while len(ids) < 97 * B + T:
        ids += ids
    if lp:
        return [ids[i * 97: i * 97 + T - 37 * (i % 5)] for i in range(B)]
    return [ids[i * 97: i * 97 + T] for i in range(B)]


def prefill_batch(lm, rows, chunk_tokens=2048):
    """Batched prefill in sequence chunks of <= chunk_tokens total tokens (B=16 x 512 in one call peaked at 43.5 GB)."""
    from mlx_vlm.generate.ar import _left_pad_prompts, _make_cache
    lengths = [len(r) for r in rows]
    mem_guard(len(rows), max(lengths), "bench_bdec prefill")
    x = _left_pad_prompts(rows) if len(set(lengths)) > 1 else mx.array(rows)
    cache = _make_cache(lm, [max(lengths) - n for n in lengths])
    T = x.shape[1]; c = max(64, chunk_tokens // len(rows))
    starts = list(range(0, T, c))
    for s0 in starts[:-1]:
        out = lm(inputs=x[:, s0:s0 + c], cache=cache, return_hidden=True, skip_logits=True)
        mx.eval(out.hidden_states[0]); mx.clear_cache()
    x = x[:, starts[-1]:]
    lg = lm(inputs=x, cache=cache).logits[:, -1]
    y = mx.argmax(lg, axis=-1).astype(mx.int32)
    mx.eval(y)
    return cache, y


def decode(stepfn, y, steps):
    """stepfn(y [B]) -> next y [B] (lazy). Returns host arrival times of each step's tokens."""
    times = []
    nxt = stepfn(y); mx.async_eval(nxt)
    for _ in range(steps):
        cur = nxt
        nxt = stepfn(cur); mx.async_eval(nxt)
        mx.eval(cur); times.append(time.perf_counter())
    mx.eval(nxt)
    return times


def run_mode(model, processor, mode, B, a):
    lm = model.language_model
    if mode == "bg":
        from bench import batch_bench
        text = load_prompts()[0]["text"] + " Please answer at length."
        r = batch_bench(model, processor, B, a.new, text)
        return dict(agg=r["agg_decode_tps"], per_stream=r["agg_decode_tps"] / B, p50_ms=None, p95_ms=None)
    rows = rows_for(processor, B, a.ctx, a.lp)
    t0 = time.perf_counter()
    cache, y = prefill_batch(lm, rows)
    pf = time.perf_counter() - t0
    if mode == "stock":
        def stepfn(t):
            return mx.argmax(lm(inputs=t[:, None], cache=cache).logits[:, -1], axis=-1).astype(mx.int32)
    elif mode in ("fd", "fdb"):
        from mlx_vlm.models.inkling2.fast_decode import FastDecoder, Unsupported
        old = os.environ.get("INKLING_BATCHFAST")
        os.environ["INKLING_BATCHFAST"] = a.bf if mode == "fdb" else ""  # read in FastDecoder.__init__
        try:
            dec = FastDecoder(lm).from_cache(cache)
        except Unsupported as e:
            return dict(refused=str(e))
        finally:
            if old is None: os.environ.pop("INKLING_BATCHFAST", None)
            else: os.environ["INKLING_BATCHFAST"] = old
        del cache

        def stepfn(t):
            return dec.step(t[:, None], compiled=FD_COMPILED)[1]
    else:
        raise SystemExit(f"unknown mode {mode}")
    # warm the compiled graph / kernels on this shape outside the timing
    for _ in range(a.warm):
        y = stepfn(y)
    mx.eval(y)
    times = decode(stepfn, y, a.new)
    dt = np.diff(np.array(times)) * 1e3
    wall = times[-1] - times[0]
    n = len(times) - 1
    return dict(agg=B * n / wall, per_stream=n / wall, p50_ms=float(np.percentile(dt, 50)),
                p95_ms=float(np.percentile(dt, 95)), max_ms=float(dt.max()), mean_ms=float(dt.mean()), prefill_s=pf)


def route_stats(model, processor, a):
    import mlx_vlm.models.inkling2.language as lang
    orig = lang._route

    def hooked(*args, **kw):
        r = orig(*args, **kw)
        ROUTES.append(r[0])
        return r
    lang._route = hooked
    lm = model.language_model
    out = {}
    for B in [int(b) for b in a.batches.split(",")]:
        cache, y = prefill_batch(lm, rows_for(processor, B, a.ctx, a.lp))
        uniq = []
        for _ in range(a.stat_steps):
            ROUTES.clear()
            lg = lm(inputs=y[:, None], cache=cache).logits[:, -1]
            y = mx.argmax(lg, axis=-1).astype(mx.int32)
            mx.eval(y, *ROUTES)
            uniq.append([len(np.unique(np.array(r).reshape(-1))) for r in ROUTES])
        u = np.array(uniq, dtype=np.float64)  # [steps, n_moe_layers]
        out[B] = dict(mean_unique=float(u.mean()), p95_unique=float(np.percentile(u, 95)),
                      routes=B * model.language_model.config.num_experts_per_tok, n_layers=int(u.shape[1]),
                      per_layer_mean=[round(float(x), 2) for x in u.mean(0)])
        print(f"ROUTE B={B} unique experts/layer mean {u.mean():.2f} (p95 {np.percentile(u, 95):.1f}) "
              f"of {out[B]['routes']} routes, layers {u.shape[1]}", flush=True)
        del cache; mx.clear_cache()
    lang._route = orig
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=""); ap.add_argument("--batches", default="4,8,16")
    ap.add_argument("--modes", default="stock,fd"); ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--new", type=int, default=128); ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--lp", action="store_true")
    ap.add_argument("--route-stats", action="store_true"); ap.add_argument("--bf", default="all"); ap.add_argument("--stat-steps", type=int, default=32)
    a = ap.parse_args()
    setup_memory(); model, processor = load_model()
    print("env", env_flags(), "fork", fork_info(), "lp", a.lp, flush=True)
    res = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=a.tag, fork=fork_info(), env=env_flags(), ctx=a.ctx,
               new=a.new, lp=a.lp, rows={})
    if a.route_stats:
        res["route_stats"] = route_stats(model, processor, a)
    else:
        modes = [m for m in a.modes.split(",") if m]
        # global warm-up: one short run per mode at the smallest B (kernels, compile caches)
        for m in modes:
            run_mode(model, processor, m, int(a.batches.split(",")[0]), argparse.Namespace(**{**vars(a), "new": 8}))
            mx.clear_cache()
        for B in [int(b) for b in a.batches.split(",")]:
            acc = {m: [] for m in modes}
            for _ in range(a.reps):
                for m in modes:
                    r = run_mode(model, processor, m, B, a); acc[m].append(r); mx.clear_cache()
            for m in modes:
                rs = acc[m]
                if "refused" in rs[0]:
                    row = dict(refused=rs[0]["refused"])
                    print(f"B={B:2d} {m:5s} REFUSED: {row['refused']}", flush=True)
                else:
                    aggs = [r["agg"] for r in rs]
                    best = rs[int(np.argsort(aggs)[len(aggs) // 2])]
                    row = dict(best, agg_all=aggs, spread_pct=100 * (max(aggs) - min(aggs)) / np.median(aggs))
                    p = lambda v: "-" if v is None else f"{v:.2f}"
                    print(f"B={B:2d} {m:5s} agg {np.median(aggs):7.1f} tok/s  per-stream {row['per_stream']:6.2f}  "
                          f"step p50 {p(row['p50_ms'])} p95 {p(row['p95_ms'])} ms  spread {row['spread_pct']:.1f}%  "
                          f"aggs {[round(x, 1) for x in aggs]}", flush=True)
                res["rows"][f"{m}-B{B}"] = row
    res["peak_gb"] = mx.get_peak_memory() / 1e9
    print(f"peak {res['peak_gb']:.1f} GB", flush=True)
    with open(os.path.join(RUNS, "bench_bdec.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
