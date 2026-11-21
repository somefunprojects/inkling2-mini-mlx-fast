"""Spec decode vs plain decode at long context. Usage:
  spec_longctx.py --ctxs 4096,16384,32768 [--n 3] [--new 256] [--reps 2] [--spec|--no-spec] [--tag T]
Prompts: stdlib source (longctx_text.build_ids, variant a, raw un-templated ids, nested prefixes of one text; PSF, nothing model-written).
Prefill chunk 1024 everywhere (mem_guard counts the chunk x ctx transient; cap 55 GB). No GPU sampler. Per context:
  plain : harness FastDecoder/stock loop (as bench.py decode_run): prefill, then `new` greedy tokens, async_eval pipelined; tok/s excludes prefill.
  spec  : spec_bench2.run_pipe (DSpark dflash drafter, FAST_VERIFY, one host sync per round) at n = --n draft tokens (block n+1); needs
          INKLING_FAST_VERIFY=1 and a fork with the drafter wiring (specfast). Reports tok/s, tokens/round (= mean accepted+1), round ms
          (mean wall of a round) and token agreement with the plain run of the SAME process (spec is NONBITWISE-OK, not bitwise).
One warm-up run per context (24 tokens, same prompt) compiles that context's buckets before timing. Rows -> $INKLING_RUNS/spec_longctx.jsonl."""
import argparse, gc, json, os, statistics, sys, time
import mlx.core as mx
from common import *
import common

CHUNK = 1024
OUT = os.path.join(RUNS, "spec_longctx.jsonl")


def make_ids(n):
    from tokenizers import Tokenizer
    from longctx_text import build_ids
    tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    return mx.array(build_ids(tk, n, "a")[:n])[None]


def plain_run(model, ids, new):
    cache = model.language_model.make_cache()
    t0 = time.perf_counter(); lg = prefill(model, ids, cache, CHUNK); y = mx.argmax(lg, axis=-1); mx.eval(y); pf = time.perf_counter() - t0
    toks = [y.reshape(-1)]
    t0 = time.perf_counter()
    nxt = mx.argmax(step_logits(model, y[None], cache), axis=-1); mx.async_eval(nxt)
    for _ in range(new - 2):
        cur = nxt; toks.append(cur.reshape(-1))
        nxt = mx.argmax(step_logits(model, cur[None], cache), axis=-1); mx.async_eval(nxt)
    toks.append(nxt.reshape(-1)); out = mx.concatenate(toks); mx.eval(out); dt = time.perf_counter() - t0
    fd = common._FAST.get(id(cache)) is not None if common.FAST_DECODE else False
    res = (out.tolist(), (len(toks) - 1) / dt, ids.shape[1] / pf, fd)
    del cache; common._FAST.clear(); gc.collect(); mx.clear_cache()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ctxs", default="4096,16384,32768"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--new", type=int, default=256); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--tag", default="")
    ap.add_argument("--spec", dest="spec", action="store_true", default=True); ap.add_argument("--no-spec", dest="spec", action="store_false")
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    est = mem_guard(1, max(ctxs) + a.new + 64, "spec_longctx", chunk=CHUNK)
    print(f"mem_guard: est peak {est:.1f} GB (cap {MEM_CAP_GB}), chunk {CHUNK}", flush=True)
    setup_memory(); model, processor = load_model()
    d = None
    if a.spec:
        assert os.environ.get("INKLING_FAST_VERIFY", "0") == "1", "spec needs INKLING_FAST_VERIFY=1"
        import spec_bench as sb, spec_bench2 as sb2
        sb2.prefill_capture = None  # replaced below with a chunk-1024 version
        (_, res, d, err), = sb.load_drafters(["dflash"])
        if err: raise SystemExit(err)

        def prefill_capture(lm, drafter, ids):  # sb2.prefill_capture with CHUNK instead of 2048
            layer_ids = list(drafter.config.target_layer_ids); cache = lm.make_cache(); T, chunks = ids.shape[1], []
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                out = lm(inputs=ids[:, s:e], cache=cache, capture_layer_ids=layer_ids, skip_logits=e < T)
                mx.eval(out.hidden_states); chunks.append(out.hidden_states)
            hidden = mx.concatenate([mx.concatenate(parts, axis=1) for parts in zip(*chunks)], axis=-1)
            return cache, hidden, int(mx.argmax(out.logits[0, -1]).item()), layer_ids
        sb2.prefill_capture = prefill_capture
    base = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=a.tag, fork=fork_info(), env=env_flags(), chunk=CHUNK, new=a.new, n=a.n, gpu_sampler="off")
    print("fork", base["fork"], "env", base["env"], flush=True)
    for ctx in ctxs:
        ids = make_ids(ctx)
        plain_run(model, ids, 24)  # warm-up (compile this context's buckets)
        prs = [plain_run(model, ids, a.new) for _ in range(a.reps)]
        ptoks = prs[0][0]; ptps = statistics.median(r[1] for r in prs)
        row = dict(base, ctx=ctx, plain_tps=ptps, plain_runs=[r[1] for r in prs], prefill_tps=statistics.median(r[2] for r in prs), fd_active=prs[0][3])
        line = f"ctx {ctx:6d}: plain {ptps:6.1f} tok/s (runs {[round(r[1], 1) for r in prs]}, FD {prs[0][3]}, prefill {row['prefill_tps']:.0f} tok/s)"
        if a.spec:
            sb2.run_pipe(model, d, ids, 24, a.n + 1); mx.clear_cache()
            runs = []
            for _ in range(a.reps):
                toks, dt, rounds = sb2.run_pipe(model, d, ids, a.new, a.n + 1)
                tpr = statistics.fmean(r["acc"] + 1 for r in rounds); rms = 1e3 * statistics.fmean(r["round"] for r in rounds)
                agree = sum(int(x == y) for x, y in zip(toks, ptoks)); first = next((i for i, (x, y) in enumerate(zip(toks, ptoks)) if x != y), -1)
                runs.append(dict(tps=len(toks) / dt, tokens_per_round=tpr, round_ms=rms, rounds=len(rounds), agree=agree, first_div=first, total=len(toks)))
                mx.clear_cache()
            row["spec"] = runs; row["spec_tps"] = statistics.median(r["tps"] for r in runs)
            row["spec_tokens_per_round"] = statistics.fmean(r["tokens_per_round"] for r in runs); row["spec_round_ms"] = statistics.fmean(r["round_ms"] for r in runs)
            row["speedup_vs_plain_same_process"] = row["spec_tps"] / ptps
            agrees = ["%d/%d" % (r["agree"], r["total"]) for r in runs]; firsts = [r["first_div"] for r in runs]
            line += (f" | spec n={a.n} {row['spec_tps']:6.1f} tok/s (x{row['speedup_vs_plain_same_process']:.2f}) tok/round {row['spec_tokens_per_round']:.2f} "
                     f"round {row['spec_round_ms']:.1f} ms | tokens agree with plain {agrees} first_div {firsts}")
        print(line, flush=True)
        with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
        del ids; gc.collect(); mx.clear_cache()


if __name__ == "__main__":
    main()
