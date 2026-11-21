"""Stock-path decode step time per INKLING_FAST flag on the e2e random model.

Indicative only (M4 Pro, random 2304-wide model, 30 visits, 16 experts).
Usage: python bench_ideas.py [flagset ...]   (each flagset runs in a subprocess)
"""
import json, os, subprocess, sys, time
os.environ.setdefault("E2E_CFG", json.dumps({"intermediate_size": 1152, "loop_config": {
    "core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 64}}))
HERE = os.path.dirname(os.path.abspath(__file__))


def worker():
    sys.path.insert(0, HERE)
    import numpy as np
    import test_e2e
    import mlx.core as mx
    lm, cfg = test_e2e.build()
    out = {}
    for B in (1, 4):
        rng = np.random.default_rng(0)
        c = lm.make_cache()
        lg = lm(mx.array(rng.integers(0, 8000, size=(B, 90))), cache=c).logits[:, -1]
        y = mx.argmax(lg, -1)
        ts = []
        for s in range(45):
            t0 = time.perf_counter()
            lg = lm(y[:, None], cache=c).logits[:, -1]
            y = mx.argmax(lg, -1)
            mx.eval(y)
            ts.append(time.perf_counter() - t0)
        out[B] = float(np.median(ts[5:]) * 1e3)
    print(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(); sys.exit(0)
    sets = sys.argv[1:] or ["", "route", "moesort", "ctrlbatch", "xpass_keys", "xpass", ""]
    for f in sets:
        env = dict(os.environ); env.pop("INKLING_FAST", None)
        if f: env["INKLING_FAST"] = f
        r = subprocess.run([sys.executable, __file__, "--worker"], env=env, capture_output=True, text=True)
        try:
            d = json.loads(r.stdout.strip().splitlines()[-1])
            print(f"{f or 'stock':40s} B=1 {d['1']:7.2f} ms/step   B=4 {d['4']:7.2f} ms/step", flush=True)
        except Exception:
            print(f, "FAILED", r.stderr[-800:])
