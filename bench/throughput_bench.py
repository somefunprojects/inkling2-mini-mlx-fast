"""Aggregate-throughput bench for many-request / data-generation use. Usage:
  throughput_bench.py [--tag T] [--ns 4,8,16,32] [--per-n 5] [--no-warmup] [--honour-eos] [--workload prompts_tput.json]

ENGINE: continuous batching, the fork's own mlx_vlm.generate.ar.BatchGenerator (the engine behind mlx_vlm.server),
driven directly: insert() on arrival, then decode_step() / prefill_step() in a loop (streaming order: tokens are published
before the next prefill chunk, as server/generation.py does). Greedy, compute_logprobs=False (server default).
Prefill is chunked and interleaved with decode (budget 2048 prompt tokens, capped at 256 while peers decode); up to 8 prompts
are prefilled together; finished rows leave the decode batch and queued prompts join it (extend) -- that IS continuous batching.

WORKLOAD (prompts_tput.json, build_prompts_tput.py; stdlib text only): CLOSED LOOP with N requests in flight. All N start at t=0
(a burst); whenever one finishes the next request from the fixed-seed list arrives immediately (so the steady state keeps the
batch at N). Total M = per-n * N requests (default 5N: 20/40/80/160). Mix 50% chat ~100 tok / 30% code ~1k / 20% doc ~4k
prompt tokens, output 64..512 forced (EOS ignored so every request generates exactly max_tokens; --honour-eos to stop at EOS).

METRICS (wall = first insert -> last token; no GPU sampler thread anywhere in the timed region):
  total_tps = generated tokens / wall            prefill_tps = prompt tokens / time spent inside prefill_step() calls
  decode_only_tps = generated tokens / (wall - prefill time)  (what the decode steps alone sustained)
  latency p50/p95 per request (arrival -> last token); ttft p50/p95 (arrival -> first token); tpot (per-token latency) p50/p95 =
  gaps between consecutive tokens of one request, pooled over requests, INCLUDING stalls where a prefill chunk ran between
  decode steps; peak_gb = mx peak memory over the timed run.
Timestamps are taken right after decode_step() returns (the batch sync-evals the previous token there), so a token's time is
accurate to one decode step. Appends one row to $INKLING_RUNS/throughput.jsonl (fork sha + dirty flag + INKLING*/MLX_* env)."""
import argparse, gc, hashlib, json, os, platform, statistics, sys, time
import mlx.core as mx
from common import *

TPUT = os.path.join(RUNS, "throughput.jsonl")


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_workload(path):
    with open(path) as f:
        w = json.load(f)
    return w


def prep_requests(processor, reqs):
    """Chat-template each text as one user turn -> token id lists (CPU, outside the timed region)."""
    out = []
    for r in reqs:
        ids = chat_ids(processor, r["text"])[0].tolist()
        out.append(dict(id=r["id"], kind=r["kind"], ids=ids, prompt_tokens=len(ids), max_tokens=r["max_tokens"]))
    return out


def make_gen(model, processor, N, honour_eos):
    from mlx_vlm.generate import ar
    from mlx_vlm.generate.common import DEFAULT_PREFILL_STEP_SIZE, _chunked_prefill_enabled
    chunked = _chunked_prefill_enabled(model, input_ids=None, inputs_embeds=None, draft_model=None, draft_kind=None,
                                       prefill_kwargs={})
    step = DEFAULT_PREFILL_STEP_SIZE if chunked else None
    gen = ar.BatchGenerator(model.language_model, processor, completion_batch_size=N, prefill_batch_size=min(8, N),
                            prefill_step_size=step, compute_logprobs=False, greedy_sampling=True)
    cfg = dict(completion_batch_size=N, prefill_batch_size=min(8, N), prefill_step_size=step, chunked_prefill=bool(chunked))
    return gen, cfg


def run_closed_loop(model, processor, reqs, N, honour_eos, label):
    """Run len(reqs) requests with N in flight. Returns dict of raw records + aggregate numbers."""
    tok = tokenizer_of(processor)
    sc = tok.stopping_criteria
    saved_eos = list(sc.eos_token_ids)
    est = mem_guard(N, max(r["prompt_tokens"] + r["max_tokens"] for r in reqs), f"throughput {label}")  # hard memory cap
    print(f"mem_guard {label}: est peak {est:.1f} GB (cap {MEM_CAP_GB})", flush=True)
    if not honour_eos:
        sc.eos_token_ids = []  # never stop on EOS: each request generates exactly max_tokens
    gen, cfg = make_gen(model, processor, N, honour_eos)
    try:
        rec = {}      # uid -> dict
        queue = list(reqs)
        inflight = 0
        tok_times = {}  # uid -> [t of each token]

        def submit():
            nonlocal inflight
            r = queue.pop(0)
            mem_guard(inflight + 1, r["prompt_tokens"] + r["max_tokens"], f"submit req {r['id']}")
            emb = model.get_input_embeddings(mx.array(r["ids"])[None]).inputs_embeds
            mx.eval(emb)
            (uid,) = gen.insert([r["ids"]], max_tokens=r["max_tokens"], prompt_kwargs=[dict(inputs_embeds=emb)])
            rec[uid] = dict(req=r["id"], kind=r["kind"], prompt_tokens=r["prompt_tokens"], max_tokens=r["max_tokens"],
                            t_arrive=time.perf_counter(), t_first=None, t_last=None, ntok=0)
            tok_times[uid] = []
            inflight += 1

        mx.reset_peak_memory()
        if engaged is not None: engaged.reset()
        gc.collect()
        t_start = time.perf_counter()
        for _ in range(min(N, len(queue))):
            submit()
        t_prefill = 0.0
        n_prefill_calls = 0
        n_decode_steps = 0
        ntok_total = 0
        while gen.has_work or queue:
            resp = gen.decode_step()
            now = time.perf_counter()
            n_decode_steps += 1 if resp else 0
            for r in resp:
                d = rec[r.uid]
                if d["t_first"] is None:
                    d["t_first"] = now
                d["t_last"] = now; d["ntok"] += 1; tok_times[r.uid].append(now); ntok_total += 1
                if r.finish_reason is not None:
                    inflight -= 1
            while inflight < N and queue:  # closed loop: a finished request is replaced at once
                submit()
            t0 = time.perf_counter()
            prog = gen.prefill_step()
            t_prefill += time.perf_counter() - t0; n_prefill_calls += 1
        wall = time.perf_counter() - t_start
        peak = mx.get_peak_memory() / 1e9
        st = gen.stats()
    finally:
        sc.eos_token_ids = saved_eos
        try:
            gen.close()
        except Exception:
            pass
        del gen
        gc.collect(); mx.clear_cache()
    lat = [d["t_last"] - d["t_arrive"] for d in rec.values()]
    ttft = [d["t_first"] - d["t_arrive"] for d in rec.values()]
    gaps = []
    for ts in tok_times.values():
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    prompt_tokens = sum(d["prompt_tokens"] for d in rec.values())
    gen_tokens = sum(d["ntok"] for d in rec.values())
    out = dict(N=N, requests=len(rec), wall_s=wall, gen_tokens=gen_tokens, prompt_tokens=prompt_tokens,
               total_tps=gen_tokens / wall, prefill_s=t_prefill, prefill_tps=prompt_tokens / t_prefill if t_prefill else None,
               decode_only_tps=gen_tokens / (wall - t_prefill) if wall > t_prefill else None,
               stats_prompt_tps=st.prompt_tps, stats_prompt_time_s=st.prompt_time,
               req_latency_p50_s=pct(lat, 50), req_latency_p95_s=pct(lat, 95),
               ttft_p50_s=pct(ttft, 50), ttft_p95_s=pct(ttft, 95),
               tpot_p50_ms=1e3 * pct(gaps, 50), tpot_p95_ms=1e3 * pct(gaps, 95), tpot_mean_ms=1e3 * statistics.fmean(gaps),
               peak_gb=peak, decode_steps_with_tokens=n_decode_steps, prefill_calls=n_prefill_calls, engine_cfg=cfg,
               short_requests=[uid for uid, d in rec.items() if d["ntok"] != d["max_tokens"]][:10])
    out["requests_detail"] = [dict(d, t_arrive=d["t_arrive"] - t_start, t_first=d["t_first"] - t_start, t_last=d["t_last"] - t_start)
                              for d in rec.values()]
    return out


def warmup(model, processor, reqs, N, honour_eos):
    """Untimed: same engine, N requests with max_tokens 4..4+N so the decode batch shrinks N -> 1 (compiles every B)."""
    pre = [dict(r, max_tokens=4 + i) for i, r in enumerate(reqs[:N])]
    w = run_closed_loop(model, processor, pre, N, honour_eos, "warmup")
    print(f"warmup N={N}: {w['wall_s']:.1f} s, {w['gen_tokens']} tok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=""); ap.add_argument("--ns", default="4,8,16,32")
    ap.add_argument("--per-n", type=int, default=5, help="total requests per run = per-n * N")
    ap.add_argument("--no-warmup", action="store_true"); ap.add_argument("--honour-eos", action="store_true")
    ap.add_argument("--workload", default=os.path.join(HERE, "prompts_tput.json"))
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    setup_memory(); model, processor = load_model()
    w = load_workload(a.workload)
    reqs = prep_requests(processor, w["requests"])
    need = max(a.per_n * n for n in ns)
    assert len(reqs) >= need, f"workload has {len(reqs)} requests, need {need}"
    import mlx_vlm
    base = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=a.tag, fork=fork_info(), env=env_flags(),
                engine="continuous batching: mlx_vlm.generate.ar.BatchGenerator (insert/decode_step/prefill_step), closed loop",
                workload=dict(file=os.path.basename(a.workload), sha=w.get("sha"), seed=w.get("seed"), per_n=a.per_n,
                              eos=("honoured" if a.honour_eos else "ignored (forced max_tokens)")),
                gpu_sampler="off", mlx=mx.__version__ if hasattr(mx, "__version__") else None,
                mlx_vlm=getattr(mlx_vlm, "__version__", None), python=platform.python_version())
    print("fork", base["fork"], "env", base["env"], flush=True)
    print("engine:", base["engine"], flush=True)
    os.makedirs(RUNS, exist_ok=True)
    for N in ns:
        M = a.per_n * N
        sub = reqs[:M]
        if not a.no_warmup:
            warmup(model, processor, reqs, N, a.honour_eos)
        res = run_closed_loop(model, processor, sub, N, a.honour_eos, f"N={N}")
        row = dict(base, **{k: v for k, v in res.items()})
        if engaged is not None:
            try:
                row["engaged"] = engaged.snapshot()
            except Exception as e:
                row["engaged"] = f"error {e}"
        with open(TPUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"N={N:2d} M={M:3d}: total {res['total_tps']:.1f} tok/s | decode-only {res['decode_only_tps']:.1f} | prefill {res['prefill_tps']:.0f} tok/s "
              f"| latency p50/p95 {res['req_latency_p50_s']:.1f}/{res['req_latency_p95_s']:.1f} s | ttft p50/p95 {res['ttft_p50_s']:.2f}/{res['ttft_p95_s']:.2f} s "
              f"| tpot p50/p95 {res['tpot_p50_ms']:.0f}/{res['tpot_p95_ms']:.0f} ms | peak {res['peak_gb']:.1f} GB | wall {res['wall_s']:.0f} s "
              f"| gen {res['gen_tokens']} prompt {res['prompt_tokens']}", flush=True)
        if res["short_requests"]:
            print("  note: requests with ntok != max_tokens:", res["short_requests"], flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()
