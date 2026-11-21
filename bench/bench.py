"""Speed bench. Usage: bench.py [--tag NAME] [--skip-batch] [--batches 4,8,16] [--prefill-step 2048]
                           [--sampler on|off] [--reps N] [--require-engaged]
Appends one JSON line to $INKLING_RUNS/bench.jsonl.
Added (all additive): --sampler off times the decode runs WITHOUT the GPU util sampler thread and
takes gpu_util in one extra untimed pass (its tok/s goes to decode_tps_ctx*_sampler_pass); gpu_sampler is recorded.
--reps N (default 3) -> decode_tps_ctx*_{min,max,spread_pct,noise_pct}; diffs below noise_pct = max(spread, 1%) are noise.
ENGAGED lines + `engaged` record (engaged.py); --require-engaged / INKLING_REQUIRE_ENGAGED=1 exits 3 after writing
the jsonl when a flag that should run at B=1 L=1 had 0 decode hits."""
import argparse, json, os, statistics, time
import mlx.core as mx
from common import *
import gpu_stats

def decode_run(model, ids, n, step, util=None):
    cache = model.language_model.make_cache()
    t0 = time.perf_counter(); lg = prefill(model, ids, cache, step); pf = time.perf_counter() - t0
    y = mx.argmax(lg, axis=-1); mx.eval(y)
    smp = gpu_stats.GpuSampler().start() if util is not None else None
    t0 = time.perf_counter()
    nxt = mx.argmax(step_logits(model, y[None], cache), axis=-1); mx.async_eval(nxt)
    for _ in range(n - 1):
        cur = nxt
        nxt = mx.argmax(step_logits(model, cur[None], cache), axis=-1); mx.async_eval(nxt)
    mx.eval(nxt); dt = time.perf_counter() - t0
    if smp is not None: util.append(smp.stop())
    return pf, n / dt

def batch_bench(model, processor, B, new, text):
    from mlx_vlm.generate.ar import batch_generate
    r = batch_generate(model, processor, prompts=[text] * B, max_tokens=new, verbose=False)
    s = r.stats
    return dict(batch=B, agg_decode_tps=s.generation_tps, gen_tokens=s.generation_tokens,
                prompt_tps=s.prompt_tps, peak_gb=mx.get_peak_memory() / 1e9)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--batches", default="4,8,16"); ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--sampler", choices=("on", "off"), default="on",
                    help="on (default, comparable with old rows): GPU util sampler runs inside the timed decode; "
                         "off: timed decode runs clean, util comes from one extra untimed pass")
    ap.add_argument("--reps", type=int, default=3, help="timed decode repetitions per context (default 3)")
    ap.add_argument("--require-engaged", action="store_true",
                    help="exit 3 if an INKLING_FAST flag that should run at B=1 L=1 has 0 kernel hits")
    a = ap.parse_args()
    setup_memory(); model, processor = load_model()
    if engaged is not None: engaged.reset()
    res = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=a.tag, fork=fork_info(), env=env_flags(),
               prefill_step=a.prefill_step, gpu_sampler=a.sampler, reps=a.reps)
    # warm-up (compiles kernels, pages weights)
    decode_run(model, raw_ids(processor, 512), 32, a.prefill_step)
    for ctx in (512, 4096):
        ids = raw_ids(processor, ctx); mx.reset_peak_memory(); pfs, dts, utl = [], [], []
        for _ in range(a.reps):
            pf, d = decode_run(model, ids, a.new, a.prefill_step, utl if a.sampler == "on" else None)
            pfs.append(ctx / pf); dts.append(d); mx.clear_cache()
        if a.sampler == "off":  # separate untimed pass: only here does the sampler thread run
            _, d_s = decode_run(model, ids, a.new, a.prefill_step, utl); mx.clear_cache()
            res[f"decode_tps_ctx{ctx}_sampler_pass"] = d_s
        res[f"decode_tps_ctx{ctx}"] = statistics.median(dts); res[f"decode_tps_ctx{ctx}_runs"] = dts
        med = res[f"decode_tps_ctx{ctx}"]; spread = 100 * (max(dts) - min(dts)) / med
        res[f"decode_tps_ctx{ctx}_min"] = min(dts); res[f"decode_tps_ctx{ctx}_max"] = max(dts)
        res[f"decode_tps_ctx{ctx}_spread_pct"] = spread; res[f"decode_tps_ctx{ctx}_noise_pct"] = max(spread, 1.0)
        res[f"prefill_tps_{ctx}"] = statistics.median(pfs); res[f"peak_gb_ctx{ctx}"] = mx.get_peak_memory() / 1e9
        u = gpu_stats.merge_util(utl); res[f"gpu_util_ctx{ctx}"] = u
        b = gpu_stats.decode_bytes(ctx); tps = res[f"decode_tps_ctx{ctx}"]
        res[f"bw_ctx{ctx}"] = dict(b, **gpu_stats.eff(tps, b["bytes_per_tok_gb"]),
            gbps_floor_bytes=tps * b["bytes_per_tok_floor_gb"], ms_per_tok=1e3 / tps,
            floor_ms_per_tok=1e3 * b["bytes_per_tok_floor_gb"] / gpu_stats.ceiling()["ceiling_gbps"],
            note="bytes/token from PLAN.md sec.2 (stock graph 6.6 GB short / 7.8 GB at 4k; floor 5.6/5.9); gbps uses stock-graph bytes")
        print(f"ctx {ctx}: gpu util mean {u['mean']} min {u['min']} ({u['n']} samples)", flush=True)
        print(f"ctx {ctx}: decode {res[f'decode_tps_ctx{ctx}']:.1f} tok/s, prefill {res[f'prefill_tps_{ctx}']:.0f} tok/s", flush=True)
        print(f"ctx {ctx}: decode min/median/max {min(dts):.1f}/{med:.1f}/{max(dts):.1f} tok/s over {a.reps} reps, "
              f"spread {spread:.1f}%, noise threshold {max(spread, 1.0):.1f}% (sampler {a.sampler})", flush=True)
    # prefill at 1k (own measurement)
    ids = raw_ids(processor, 1024); pfs = []
    for _ in range(max(4, a.reps + 1)):
        cache = model.language_model.make_cache(); t0 = time.perf_counter(); prefill(model, ids, cache, a.prefill_step)
        pfs.append(1024 / (time.perf_counter() - t0)); mx.clear_cache()
    res["prefill_tps_1024"] = statistics.median(pfs[1:])  # first is warm-up
    res["prefill_tps_1024_min"] = min(pfs[1:]); res["prefill_tps_1024_max"] = max(pfs[1:])
    print(f"prefill 1k: {res['prefill_tps_1024']:.0f} tok/s", flush=True)
    # engaged-kernel counters: snapshot BEFORE the batched section (B>1 kernels would blur the B=1 L=1 answer)
    eng_ok = True
    if engaged is not None:
        res["engaged"] = engaged.snapshot()
        eng_ok = engaged.report(require=True if a.require_engaged else None)
    # batched aggregate decode via mlx-vlm batch_generate; report failure honestly
    res["batch"] = {}
    if not a.skip_batch:
        text = " ".join(load_prompts()[0]["text"] for _ in range(1)) + " Please answer at length."
        try:
            batch_bench(model, processor, 2, 16, text)  # warm-up
            for B in [int(x) for x in a.batches.split(",")]:
                mem_guard(B, 4096 + 128, "bench batch")  # prompt ~4k + 128 new
                mx.reset_peak_memory()
                try:
                    res["batch"][str(B)] = batch_bench(model, processor, B, 128, text)
                    print("batch", B, res["batch"][str(B)], flush=True)
                except Exception as e:
                    res["batch"][str(B)] = {"error": f"{type(e).__name__}: {e}"[:300]}; print("batch", B, "FAILED", e, flush=True)
                mx.clear_cache()
        except Exception as e:
            res["batch"] = {"unsupported": f"{type(e).__name__}: {e}"[:300]}; print("batch generation unsupported:", e, flush=True)
    res["peak_gb_overall"] = mx.get_peak_memory() / 1e9
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "bench.jsonl"), "a") as f: f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1))
    if not eng_ok:
        print("ENGAGED FAIL: exiting 3 (--require-engaged); jsonl row was written", flush=True)
        raise SystemExit(3)

if __name__ == "__main__":
    main()
