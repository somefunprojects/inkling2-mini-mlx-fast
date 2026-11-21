"""spec_bench with a pipelined FAST_VERIFY loop.

Mode `pipe` (INKLING_FAST_VERIFY=1 only): same round as spec_bench's split mode (DSpark draft_block ->
FastDecoder.spec_step -> accept), but with ONE host sync per round: the verify graph is built right
after the (lazy) draft graph, so its host build overlaps the draft's GPU time; the host then waits only
for `accepted` (needed to slice the drafter's hidden rows). Tokens are identical to split mode (same
ops, same order); only the sync points differ. Reports per-round wall and accept stats.
Optional adaptive n (--adapt): per round choose n from the recent pos-1 accept rate (EMA); every L it
can use must be gated separately.

Usage: python spec_bench2.py [--ns 3,4] [--only names] [--new 256] [--modes pipe,split] [--tag t]
Writes rows (mode "pipe"/"split") to $INKLING_RUNS/spec_bench.jsonl, prints a table + means."""
import argparse, os, statistics, sys, time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_bench as sb  # noqa: E402
from common import *  # noqa: E402,F401,F403
import gpu_stats  # noqa: E402


def prefill_capture(lm, drafter, ids):
    layer_ids = list(drafter.config.target_layer_ids)
    cache = lm.make_cache()
    T, chunks = ids.shape[1], []
    for s in range(0, T, 2048):
        e = min(s + 2048, T)
        out = lm(inputs=ids[:, s:e], cache=cache, capture_layer_ids=layer_ids, skip_logits=e < T)
        mx.eval(out.hidden_states)
        chunks.append(out.hidden_states)
    hidden = mx.concatenate([mx.concatenate(parts, axis=1) for parts in zip(*chunks)], axis=-1)
    b = int(mx.argmax(out.logits[0, -1]).item())
    return cache, hidden, b, layer_ids


def run_pipe(model, drafter, ids, new, block, adapt=None):
    """adapt: None or (n_lo, n_hi, thr): use n_hi while EMA(pos-1 accept) >= thr else n_lo."""
    from mlx_vlm.models.inkling2.fast_decode import FastDecoder
    from mlx_vlm.speculative.common import _record_speculative_round
    lm = model.language_model
    cache, hidden, b, layer_ids = prefill_capture(lm, drafter, ids)
    draft_cache = drafter.reset(model)
    sampler = lambda x: mx.argmax(x, axis=-1)
    n_max = max(block - 1, adapt[1] if adapt else 0)
    dec = FastDecoder(lm, ring_extra=n_max).from_cache(cache)
    del cache
    toks, rounds, ema = [b], [], 0.7
    t_start = time.perf_counter()
    while len(toks) < new:
        n = block - 1
        if adapt:
            n = adapt[1] if ema >= adapt[2] else adapt[0]
        bs = min(n + 1, new - len(toks) + 1)
        if bs <= 1:
            break
        t0 = time.perf_counter()
        draft = drafter.draft_block(b, hidden, draft_cache, bs, sampler, mx.int32)
        ta = time.perf_counter()
        vin = mx.concatenate([mx.array([[b]], dtype=mx.int32), draft.astype(mx.int32)], axis=1)
        res = dec.spec_step(vin, captures=layer_ids)
        tb = time.perf_counter()
        hid = mx.concatenate(res["captures"], axis=-1)
        mx.async_eval(res["emitted"], res["accepted"], hid)
        t1 = time.perf_counter()
        accepted = int(res["accepted"][0].item())  # the one sync
        t2 = time.perf_counter()
        dec.observe([accepted])
        new_tokens = res["emitted"][0, : accepted + 1].tolist()[: new - len(toks)]
        _record_speculative_round(drafter, accepted, bs - 1)
        hidden = hid[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b
        ema = 0.8 * ema + 0.2 * (1.0 if accepted >= 1 else 0.0)
        rounds.append(dict(n=bs - 1, acc=accepted, build=t1 - t0, dbuild=ta - t0, vbuild=tb - ta,
                           wait=t2 - t1,
                           round=time.perf_counter() - t0))
        toks.extend(new_tokens)
    return toks[:new], time.perf_counter() - t_start, rounds


def run_adapt0(model, drafter, ids, new, block, thr, burst):
    """Adaptive speculation with a plain-decode fallback (n=0): spec rounds at n=block-1 while the EMA of
    tokens/round >= thr; otherwise `burst` plain FastDecoder steps (bitwise L=1 path, with drafter
    captures so the drafter context stays complete), then one probe spec round."""
    from mlx_vlm.models.inkling2.fast_decode import FastDecoder
    from mlx_vlm.speculative.common import _record_speculative_round
    lm = model.language_model
    cache, hidden, b, layer_ids = prefill_capture(lm, drafter, ids)
    draft_cache = drafter.reset(model)
    sampler = lambda x: mx.argmax(x, axis=-1)
    dec = FastDecoder(lm, ring_extra=block - 1).from_cache(cache)
    del cache
    toks, rounds, ema = [b], [], float(block)
    t_start = time.perf_counter()
    while len(toks) < new:
        t0 = time.perf_counter()
        if ema < thr:
            rows, cur = [], mx.array([[b]], dtype=mx.int32)
            for _ in range(min(burst, new - len(toks))):
                _, nxt, caps = dec.step_captures(cur, layer_ids)
                rows.append(mx.concatenate(caps, axis=-1))
                cur = nxt.reshape(1, 1)
                mx.async_eval(cur)
                toks.append(cur)
            hidden = mx.concatenate(rows, axis=1)
            toks[-len(rows):] = [int(t.item()) for t in toks[-len(rows):]]
            b = toks[-1]
            rounds.append(dict(n=0, acc=-1, round=time.perf_counter() - t0, steps=len(rows)))
            ema = thr  # probe: next round speculates
            continue
        bs = min(block, new - len(toks) + 1)
        if bs <= 1:
            break
        draft = drafter.draft_block(b, hidden, draft_cache, bs, sampler, mx.int32)
        vin = mx.concatenate([mx.array([[b]], dtype=mx.int32), draft.astype(mx.int32)], axis=1)
        res = dec.spec_step(vin, captures=layer_ids)
        hid = mx.concatenate(res["captures"], axis=-1)
        mx.async_eval(res["emitted"], res["accepted"], hid)
        accepted = int(res["accepted"][0].item())
        dec.observe([accepted])
        new_tokens = res["emitted"][0, : accepted + 1].tolist()[: new - len(toks)]
        _record_speculative_round(drafter, accepted, bs - 1)
        hidden = hid[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b
        ema = 0.7 * ema + 0.3 * (accepted + 1)
        rounds.append(dict(n=bs - 1, acc=accepted, round=time.perf_counter() - t0))
        toks.extend(new_tokens)
    return toks[:new], time.perf_counter() - t_start, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="3,4")
    ap.add_argument("--modes", default="pipe")
    ap.add_argument("--only")
    ap.add_argument("--new", type=int, default=NEW_TOKENS)
    ap.add_argument("--adapt", default="", help="n_lo,n_hi,thr  (adds a row tagged n=-1)")
    ap.add_argument("--adapt0", default="", help="thr,burst: plain-decode fallback mode 'adapt0' at each n")
    ap.add_argument("--sampler", choices=("on", "off"), default="off",
                    help="off (default): no GPU util sampler thread during the timed runs (it costs ~3.7%)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.sampler == "off":
        sb.with_util = lambda fn, *args: (fn(*args), None)
    assert os.environ.get("INKLING_FAST_VERIFY", "0") == "1", "pipe mode needs INKLING_FAST_VERIFY=1"
    setup_memory()
    ns = [int(x) for x in a.ns.split(",") if x]
    modes = a.modes.split(",")
    model, processor = load_model()
    prompts = [p for p in load_prompts() if not a.only or p["name"] in a.only.split(",")]
    (_, res, d, err), = sb.load_drafters(["dflash"])
    if err:
        raise SystemExit(err)
    meta = dict(fork=fork_info(), env=env_flags(), tag=a.tag, ts=time.strftime("%FT%T"), gpu_sampler=a.sampler)
    adapt = None
    if a.adapt:
        lo, hi, thr = a.adapt.split(",")
        adapt = (int(lo), int(hi), float(thr))
    # warm-up: compile every L used
    wids = chat_ids(processor, prompts[0]["text"])
    for n in sorted(set(ns) | ({adapt[0], adapt[1]} if adapt else set())):
        run_pipe(model, d, wids, 24, n + 1)
        if "adapt0" in modes and n > 0:
            run_adapt0(model, d, wids, 24, n + 1, 1e9, 4)  # compiles step_captures
    mx.clear_cache()
    table = []
    for p in prompts:
        ids = chat_ids(processor, p["text"])
        ref, ref_src = sb.reference_tokens(model, processor, p, a.new)
        cfgs = [(n, None) for n in ns] + ([(-1, adapt)] if adapt else [])
        for n, ad in cfgs:
            for mode in modes:
                if mode == "adapt0":
                    thr, burst = a.adapt0.split(",")
                    (toks, dt, rounds), ut = sb.with_util(run_adapt0, model, d, ids, a.new, n + 1,
                                                          float(thr), int(burst))
                    parts = {"round": sb.summarize_times([r["round"] for r in rounds if r["n"] > 0])}
                    rounds_spec = [r for r in rounds if r["n"] > 0] or [dict(n=n, acc=0)]
                    plain = sum(r.get("steps", 0) for r in rounds)
                elif mode == "pipe":
                    (toks, dt, rounds), ut = sb.with_util(run_pipe, model, d, ids, a.new,
                                                          (n if n > 0 else ad[1]) + 1, ad)
                    parts = {k: sb.summarize_times([r[k] for r in rounds])
                             for k in ("build", "dbuild", "vbuild", "wait", "round")}
                else:
                    (toks, dt, rounds), ut = sb.with_util(sb.run_split, model, d, ids, a.new, n + 1)
                    parts = {k: sb.summarize_times([r[k] for r in rounds])
                             for k in ("draft_build", "draft_gpu", "verify_build", "verify_wait", "commit")}
                if mode != "adapt0":
                    rounds_spec, plain = rounds, 0
                nn = max(r["n"] for r in rounds_spec)
                st = sb.accept_stats([r["acc"] for r in rounds_spec], nn)
                st["plain_steps"] = plain
                fd = sb.first_divergence(toks, ref)
                tps = (len(toks) - 1) / dt
                sb.record(dict(meta, mode=mode, prompt=p["name"], prompt_len=ids.shape[1], kind="dflash",
                               resolved=res, n=n, adapt=a.adapt if n < 0 else None, tok_s=tps, first_div=fd,
                               ref_source=ref_src, accept=st, per_round=parts, gpu_util=ut))
                table.append((p["name"], mode, n, tps, st.get("mean_tokens_per_round"), fd))
                print(f"{p['name']} {mode} n={n} {tps:.1f} tok/s  tok/rnd {st.get('mean_tokens_per_round', 0):.2f}  "
                      + " ".join(f"{k}={v['median_ms']:.1f}" for k, v in parts.items() if v), flush=True)
                mx.clear_cache()
    print(f"\n{'prompt':<14}{'mode':<7}{'n':>3}{'tok/s':>9}{'tok/rnd':>9}{'first_div':>10}")
    for name, mode, n, tps, tpr, fd in table:
        print(f"{name:<14}{mode:<7}{n:>3}{tps:>9.1f}{(tpr or 0):>9.2f}{str(fd):>10}")
    for mode in modes:
        for n in sorted({r[2] for r in table}):
            xs = [r[3] for r in table if r[1] == mode and r[2] == n]
            if xs:
                kinds = {}
                for r in table:
                    if r[1] == mode and r[2] == n:
                        kinds.setdefault(r[0].split("_")[0], []).append(r[3])
                print(f"MEAN {mode} n={n}: {statistics.mean(xs):.1f} tok/s  "
                      + " ".join(f"{k}={statistics.mean(v):.1f}" for k, v in kinds.items()))


if __name__ == "__main__":
    main()
