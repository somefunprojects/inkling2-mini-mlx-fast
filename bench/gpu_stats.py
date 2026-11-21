"""GPU-efficiency helpers: GPU utilization sampler, bytes-per-token model, bandwidth ceiling.
macOS only. Additive: bench.py / spec_bench.py add fields, never change existing ones."""
import json, os, re, subprocess, threading, time

RUNS = os.path.abspath(os.environ.get("INKLING_RUNS", "runs"))
_UTIL = re.compile(r'"Device Utilization %"=(\d+)')


class GpuSampler:
    """Polls `ioreg -r -d 1 -c IOAccelerator` Device Utilization % every `period` s in a thread.
    Note: each poll spawns ioreg (a few ms CPU); this adds a little host load, so tok/s may shift by ~0-1%."""
    def __init__(self, period=0.15):
        self.period, self.samples, self._stop, self._t = period, [], threading.Event(), None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True,
                                     text=True, timeout=2).stdout
                m = _UTIL.search(out)
                if m: self.samples.append(int(m.group(1)))
            except Exception:
                pass
            self._stop.wait(self.period)

    def start(self):
        self.samples = []; self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start(); return self

    def stop(self):
        self._stop.set()
        if self._t: self._t.join(timeout=3)
        return self.samples


def util_summary(samples):
    if not samples: return {"mean": None, "min": None, "n": 0}
    return {"mean": sum(samples) / len(samples), "min": min(samples), "n": len(samples)}


def merge_util(list_of_samples):
    return util_summary([s for ss in list_of_samples for s in ss])


# ---- bandwidth ceiling: measured by membw.py (membw.json), else 1 TB/s assumption
def ceiling():
    try:
        with open(os.path.join(RUNS, "membw.json")) as f:
            d = json.load(f)
        return {"ceiling_gbps": d["read_gbps"], "ceiling_source": "measured membw.json"}
    except Exception:
        return {"ceiling_gbps": 1000.0, "ceiling_source": "assumed 1 TB/s"}


# ---- bytes per token (PLAN.md section 2; GB = 1e9 bytes)
STOCK_BYTES = {"short": 6.6, "4k": 7.8}   # current stock graph (FP32 head copy, KV copies)
FLOOR_BYTES = {"short": 5.6, "4k": 5.9}   # exact minimum + KV reads (no copies, BF16 head, cached router)
DENSE_GB = 2.58 + 0.17 + 0.14             # attention + controllers + router, read once per forward
HEAD_GB = 1.85                            # FP32 head copy (0.93 if BF16)
EXPERT_GB = {"nvfp4": 4.48e-3, "mxfp8": 8.21e-3}
N_EXPERTS, TOPK_E, N_NVFP4, N_MXFP8 = 128, 8, 44, 4
DRAFTER_GB = 2.3


def decode_bytes(ctx):
    k = "4k" if ctx >= 2048 else "short"
    return {"bytes_per_tok_gb": STOCK_BYTES[k], "bytes_per_tok_floor_gb": FLOOR_BYTES[k]}


def unique_experts(L):
    """Expected unique experts per layer for L positions under UNIFORM routing (upper bound; real routing is skewed)."""
    return N_EXPERTS * (1 - (1 - TOPK_E / N_EXPERTS) ** L)


def verify_gb(L):
    u = unique_experts(L)
    return DENSE_GB + HEAD_GB + u * (N_NVFP4 * EXPERT_GB["nvfp4"] + N_MXFP8 * EXPERT_GB["mxfp8"])


def spec_bytes_per_tok(n, mean_tokens_per_round):
    """(drafter 2.3 GB + verify(L=n+1) GB) / accepted tokens per round. KV reads ignored (small).
    Verify = dense weights + head once + uniform-routing unique experts (upper bound on bytes)."""
    if not mean_tokens_per_round: return None
    return (DRAFTER_GB + verify_gb(n + 1)) / mean_tokens_per_round


SPEC_FORMULA = ("bytes_per_tok = (2.3 GB drafter + verify_gb(L=n+1)) / mean_tokens_per_round; verify_gb = 2.89 dense "
                "+ 1.85 head + E[unique experts](L, uniform 8-of-128) * (44*4.48MB + 4*8.21MB); upper bound on bytes")


def eff(tok_s, bytes_gb):
    c = ceiling()
    gbps = tok_s * bytes_gb
    return dict(gbps=gbps, pct_of_ceiling=100 * gbps / c["ceiling_gbps"], **c)
