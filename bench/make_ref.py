"""Record reference greedy decode (256 tokens) + logprobs for each prompt.
Usage: make_ref.py [--out DIR] [--only name,name]"""
import argparse, os, time
import numpy as np
import mlx.core as mx
from common import *

def run_prompt(model, processor, text, new=NEW_TOKENS, forced=None):
    """Greedy (or teacher-forced with `forced` tokens) decode. Returns dict of numpy arrays."""
    ids = chat_ids(processor, text)
    cache = model.language_model.make_cache()
    logits = prefill(model, ids, cache)
    toks, top_ids, top_lp, lses, full = [], [], [], [], {}
    for t in range(new):
        lp, lse = logprob_stats(logits)
        ti = mx.argpartition(-lp, TOPK - 1)[:TOPK]
        am = mx.argmax(lp)
        mx.eval(am, ti, lse, lp)
        ti_np = np.array(ti); lp_np = np.array(lp)
        order = np.argsort(-lp_np[ti_np], kind="stable")
        ti_np = ti_np[order]
        top_ids.append(ti_np.astype(np.int32)); top_lp.append(lp_np[ti_np].astype(np.float32))
        lses.append(float(lse))
        if t in FULL_STEPS: full[t] = lp_np.copy()
        nxt = int(am) if forced is None else int(forced[t])
        toks.append(int(am))  # the model's own argmax at this step
        if t == new - 1: break
        logits = step_logits(model, mx.array([[nxt]]), cache)
        mx.eval(logits)
    return dict(argmax=np.array(toks, np.int32), top_ids=np.stack(top_ids), top_lp=np.stack(top_lp),
                lse=np.array(lses, np.float32), full_steps=np.array(sorted(full), np.int32),
                full_lp=np.stack([full[k] for k in sorted(full)]), prompt_len=np.int32(ids.shape[1]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=REF_DIR); ap.add_argument("--only")
    a = ap.parse_args(); setup_memory(); os.makedirs(a.out, exist_ok=True)
    model, processor = load_model()
    for p in load_prompts():
        if a.only and p["name"] not in a.only.split(","): continue
        t0 = time.time(); r = run_prompt(model, processor, p["text"])
        np.savez_compressed(os.path.join(a.out, p["name"] + ".npz"), fork=str(fork_info()), env=str(env_flags()), **r)
        print(f'{p["name"]}: prompt_len={int(r["prompt_len"])} {time.time()-t0:.1f}s', flush=True)
        mx.clear_cache()

if __name__ == "__main__":
    main()
