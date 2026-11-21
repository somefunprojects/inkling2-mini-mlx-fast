"""Gate: teacher-force the reference tokens through current code; compare to saved refs.
Teacher forcing makes KL meaningful past a divergence; argmax-match at every step <=> free-run greedy identical.

Two bars:
  BITWISE        every step's full-vocab fp32 logits equal the stock full ref (--full-ref, built once with
                 `check.py --save-full` on UNMODIFIED stock code). Reports max|dlogit| and non-bitwise step count.
  NONBITWISE-OK  not bitwise, but overall mean KL(top-64) <= --kl-floor (stock prefill-chunk noise floor,
                 2.68e-3 measured for chunk 1024). Also needs a separate FP32-exactness test on the tiny
                 model, which this script cannot judge (a reminder is printed).
  FAIL           otherwise.
Exit 0 iff a bar is met (--require-bitwise: only BITWISE). The old per-prompt lines and the final
"GATE PASS/FAIL" line are kept; bitwise columns are appended per prompt.
Added after the GATE line: median KL, max-row KL with (prompt, step), top-1 flip count, and the
ENGAGED lines (engaged.py: kernel-call counters per INKLING_FAST flag, decode phase). --require-engaged (or env
INKLING_REQUIRE_ENGAGED=1): exit 3 (if the gate itself passed) when a flag that should run at B=1 L=1 has 0 hits.
Usage: check.py [--ref DIR] [--full-ref DIR] [--only a,b] [--kl-floor 2.68e-3] [--require-bitwise] [--require-engaged]
       check.py --save-full DIR      (stock only: record full logits per step; no gate)"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
from common import *

KL_NOISE_FLOOR = 2.68e-3  # stock, prefill chunk 2048 -> 1024 (stock-vs-stock noise floor)
FULL_REF_DIR = os.path.join(REF_DIR, "full")

def kl_topk(ref_ids, ref_lp, new_full_lp):
    """KL(ref||new) on the ref top-k support, both renormalised. new_full_lp: [V]."""
    p = np.exp(ref_lp.astype(np.float64)); p /= p.sum()
    q_lp = new_full_lp[ref_ids].astype(np.float64)
    q_lp -= np.log(np.exp(q_lp).sum())
    return float(np.sum(p * (np.log(p) - q_lp)))

def kl_full(ref_lp, new_lp):
    r = ref_lp.astype(np.float64); n = new_lp.astype(np.float64)
    return float(np.sum(np.exp(r) * (r - n)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default=REF_DIR); ap.add_argument("--only")
    ap.add_argument("--full-ref", default=FULL_REF_DIR)
    ap.add_argument("--save-full", help="record every step's fp32 logits here (run on stock)")
    ap.add_argument("--kl-floor", "--kl-gate", dest="kl_floor", type=float, default=KL_NOISE_FLOOR)
    ap.add_argument("--require-bitwise", action="store_true")
    ap.add_argument("--require-engaged", action="store_true",
                    help="exit 3 if an INKLING_FAST flag that should run at B=1 L=1 has 0 kernel hits (env INKLING_REQUIRE_ENGAGED=1)")
    ap.add_argument("--prefill-chunk", type=int, default=2048, help="prefill step (refs used 2048)"); a = ap.parse_args()
    setup_memory(); model, processor = load_model()
    if engaged is not None: engaged.reset()
    print("fork", fork_info(), "env", env_flags(), "prefill_chunk", a.prefill_chunk); fail = False; allkl = []
    if a.save_full:
        os.makedirs(a.save_full, exist_ok=True)
    bw_missing, bw_bad_steps, bw_steps, bw_maxd = False, 0, 0, 0.0
    per_prompt, n_flips, n_rows = [], 0, 0
    for p in load_prompts():
        if a.only and p["name"] not in a.only.split(","): continue
        path = os.path.join(a.ref, p["name"] + ".npz")
        if not os.path.exists(path): print("MISSING ref", path); fail = True; continue
        ref = np.load(path)
        ids = chat_ids(processor, p["text"])
        if ids.shape[1] != int(ref["prompt_len"]): print(p["name"], "PROMPT LEN MISMATCH"); fail = True; continue
        # reference tokens (argmax of ref run == the tokens it fed back)
        rtoks = ref["argmax"]; N = len(rtoks)
        fpath = os.path.join(a.full_ref, p["name"] + ".npy")
        full = None
        if a.save_full:
            full_out = np.lib.format.open_memmap(os.path.join(a.save_full, p["name"] + ".npy"), mode="w+",
                                                 dtype=np.float32, shape=(N, ref["full_lp"].shape[-1]))
        elif os.path.exists(fpath):
            full = np.load(fpath, mmap_mode="r")
        else:
            bw_missing = True
        cache = model.language_model.make_cache(); logits = prefill(model, ids, cache, step=a.prefill_chunk)
        am_new, kls = [], []; klf = {}; nb, maxd, first_nb = 0, 0.0, -1
        for t in range(N):
            lg32 = logits[0].astype(mx.float32)
            lp, _ = logprob_stats(logits); mx.eval(lp, lg32); lp = np.array(lp); lg = np.array(lg32)
            if a.save_full:
                full_out[t] = lg
            elif full is not None:
                r = full[t]
                if not np.array_equal(r, lg):
                    nb += 1; first_nb = t if first_nb < 0 else first_nb
                    maxd = max(maxd, float(np.nanmax(np.abs(r.astype(np.float64) - lg))))
            am_new.append(int(lp.argmax()))
            kls.append(kl_topk(ref["top_ids"][t], ref["top_lp"][t], lp))
            if t in ref["full_steps"]:
                klf[t] = kl_full(ref["full_lp"][list(ref["full_steps"]).index(t)], lp)
            if t == N - 1: break
            logits = step_logits(model, mx.array([[int(rtoks[t])]]), cache); mx.eval(logits)
        if a.save_full:
            full_out.flush(); del full_out
        am_new = np.array(am_new); bad = np.nonzero(am_new != rtoks)[0]
        first = int(bad[0]) if len(bad) else -1
        kls = np.array(kls); allkl.append(kls); per_prompt.append((p["name"], kls)); n_flips += len(bad); n_rows += N
        bwcol = ("" if a.save_full else " | bitwise n/a (no full ref)" if full is None else
                 f" | bitwise {N - nb}/{N} steps, first_nonbitwise={first_nb}, max|dlogit|={maxd:.3g}")
        print(f'{p["name"]:12s} first_div={first:4d} mismatches={len(bad):3d} KLtop64 mean={kls.mean():.2e} max={kls.max():.2e} '
              f'| full-KL ' + " ".join(f"{k}:{v:.1e}" for k, v in klf.items()) + bwcol, flush=True)
        if full is not None:
            bw_bad_steps += nb; bw_steps += N; bw_maxd = max(bw_maxd, maxd)
        mx.clear_cache()
    if a.save_full:
        print("saved full refs to", a.save_full, "(no gate)"); sys.exit(1 if fail else 0)
    m = float(np.concatenate(allkl).mean()) if allkl else float("nan")
    print(f"overall mean KL {m:.3e} (noise floor {a.kl_floor:g})")
    if bw_missing:
        print(f"BITWISE: unknown (missing full refs in {a.full_ref}; build with stock: check.py --save-full {a.full_ref})")
    else:
        print(f"BITWISE: non-bitwise steps {bw_bad_steps}/{bw_steps}, max|dlogit| {bw_maxd:.3g}")
    bitwise = not fail and not bw_missing and bw_bad_steps == 0
    if bitwise:
        print("BAR BITWISE")
    elif not fail and not a.require_bitwise and m <= a.kl_floor:
        print(f"BAR NONBITWISE-OK (mean KL {m:.3e} <= floor {a.kl_floor:g}). REMINDER: also needs an FP32-exactness "
              "test on the tiny model (ordering-only difference); check.py cannot judge that.")
    else:
        fail = True
        print("BAR FAIL" + (" (--require-bitwise)" if a.require_bitwise else f" (mean KL {m:.3e} > floor or ref error)"))
    print("GATE", "FAIL" if fail else "PASS")
    # --- added lines (after the existing summary) ---
    if allkl:
        cat = np.concatenate(allkl)
        wn, wk = max(per_prompt, key=lambda t: float(t[1].max())); ws = int(np.argmax(wk))
        print(f"median KL {np.median(cat):.3e} (KLtop64 over {len(cat)} rows)")
        print(f"max-row KL {float(wk.max()):.3e} at (prompt={wn}, step={ws})")
        print(f"top-1 flips {n_flips}/{n_rows} steps (argmax differs from ref)")
    ok = True
    if engaged is not None:
        ok = engaged.report(require=True if a.require_engaged else None)
    sys.exit(1 if fail else (0 if ok else 3))

if __name__ == "__main__":
    main()
