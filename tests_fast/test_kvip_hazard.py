"""megafix (REVIEW_MEGA2 F1): FastDecoder steps fed KNOWN tokens with no host sync between them.
KVIP writes the KV rows in place, invisible to MLX; across steps only an explicit dependency (the
qsm kernel's ``dep`` input) orders step t+1's write after step t's SDPA read of that slot.
Modes, all with forced tokens (no data path from step t's logits to step t+1):
  sync    mx.eval(logits) after every step            (the reference)
  lazy    chain all steps, one mx.eval of every logits at the end
  last    chain all steps, eval ONLY the last step's logits (step t's qsm must still run)
  async   mx.async_eval(logits) per step, eval at the end
Each mode with kvip on and off; every mode's logits must be bitwise equal to sync/kvip-off.
Tiny model (TINY_WIDE=1 D 1024 or TINY_REAL=1 D 2304); run with the full INKLING_FAST list.
Env: T0S (default 5,40,1000), STEPS (default 48), REPS (default 3)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny
import mlx.core as mx
from mlx_vlm.models.inkling2.fast_decode import FastDecoder

if os.environ.get("TINY_WIDE", "0") == "1":
    _o = tiny.tiny_config
    def _wide(**kw):
        c = _o(**kw); c.hidden_size = 1024; c.head_dim = c.swa_head_dim = 128; return c
    tiny.tiny_config = _wide
if os.environ.get("TINY_REAL", "0") == "1":
    _o2 = tiny.tiny_config
    def _real(**kw):
        c = _o2(**kw); c.hidden_size = 2304
        c.num_attention_heads = c.swa_num_attention_heads = 24
        c.num_key_value_heads = c.swa_num_key_value_heads = 8
        c.head_dim = c.swa_head_dim = 128; import dataclasses; c.loop_config = dataclasses.replace(c.loop_config, gate_rank=64); return c
    tiny.tiny_config = _real

STEPS = int(os.environ.get("STEPS", "48"))
REPS = int(os.environ.get("REPS", "3"))
lm = tiny.build(dtype=mx.bfloat16, window=1024, rel_extent=1024, n_floor=8192)
for layer in lm.model.layers:
    sm = layer.mlp.switch_mlp
    g, u = sm.gate_proj, sm.up_proj
    if hasattr(g, "_scale_name") and g._scale_name == u._scale_name:
        u[u._scale_name] = g[g._scale_name]

fails = []
for T0 in [int(t) for t in os.environ.get("T0S", "5,40,1000").split(",")]:
    mx.random.seed(T0)
    ids = mx.random.randint(0, 500, (1, T0))
    forced = mx.random.randint(0, 500, (STEPS, 1, 1)).astype(mx.int32)
    mx.eval(ids, forced)

    def run(mode, kvip):
        cache = lm.make_cache()
        mx.eval(lm(inputs=ids, cache=cache).logits)
        dec = FastDecoder(lm).from_cache(cache)
        assert dec.mega_kvip, "mega_kvip not enabled (INKLING_FAST?)"
        dec.mega_kvip = kvip
        mx.eval(dec.state)
        outs = []
        for i in range(STEPS):
            lg, _ = dec.step(forced[i])
            outs.append(lg)
            if mode == "sync":
                mx.eval(lg)
            elif mode == "async":
                mx.async_eval(lg)
        if mode == "last":
            mx.eval(outs[-1])
            return outs[-1:]
        mx.eval(outs)
        return outs

    ref = run("sync", False)
    for mode in ("sync", "lazy", "last", "async"):
        for kvip in (False, True):
            for rep in range(REPS):
                got = run(mode, kvip)
                want = ref[-1:] if mode == "last" else ref
                ok = all(mx.array_equal(a, b).item() for a, b in zip(want, got))
                if not ok:
                    bad = sum(not mx.array_equal(a, b).item() for a, b in zip(want, got))
                    fails.append(f"T0={T0} {mode} kvip={kvip} rep={rep}: {bad} steps differ")
                    print("FAIL", fails[-1], flush=True)
        print(f"T0={T0} mode={mode}: done", flush=True)
print("KVIP_HAZARD", "PASS" if not fails else f"FAIL ({len(fails)})")
