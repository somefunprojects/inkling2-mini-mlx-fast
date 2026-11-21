"""Engaged-kernel probe (guards against a silent fallback to stock kernels). No fork edits.

The fork has no per-kernel USED counters, so the harness counts them itself: `install()` (called from common.py at
import, before any mlx_vlm import) wraps `mx.fast.metal_kernel` so every custom kernel the fork builds counts its calls
by kernel name, split by phase ("prefill" while common.prefill runs, "decode" afterwards, i.e. L=1 steps).
Kernel names are mapped to INKLING_FAST flags in FLAG_KERNELS below.

Counts are PYTHON-LEVEL calls. Under mx.compile (FastDecoder.step) the python body runs once at trace time, so the
count is "call sites in the traced step graph" (e.g. 48 layers) and NOT a per-token count; eager paths count every
call. Either way 0 in the decode phase means the kernel never entered a decode graph.

report(): `ENGAGED: flag=count ...` for every flag in INKLING_FAST that has a counter, `ENGAGED-NO-COUNTER: ...` for
the flags without one, `ENGAGED-MISSING: ...` for counted flags with 0 decode hits that should run at B=1 L=1
(superseded flags are exempt, see SUPERSEDED). Set INKLING_ENGAGED_PROBE=0 to disable the wrapper entirely.

CPU-only self-test (no mlx needed): python engaged.py --selftest"""
import collections, os, re, sys

COUNTS = {"prefill": collections.Counter(), "decode": collections.Counter()}
_PHASE = ["decode"]


def set_phase(p):
    _PHASE[0] = p


def reset():
    for c in COUNTS.values():
        c.clear()


class _Counted:
    """Callable wrapper around an mx.fast.metal_kernel object; counts calls by name and phase."""
    def __init__(self, kern, name):
        self._k, self._name = kern, name

    def __call__(self, *a, **k):
        COUNTS[_PHASE[0]][self._name] += 1
        return self._k(*a, **k)

    def __getattr__(self, x):
        return getattr(self._k, x)


def install():
    if os.environ.get("INKLING_ENGAGED_PROBE", "1") == "0":
        return False
    import mlx.core as mx
    orig = mx.fast.metal_kernel
    if getattr(orig, "_engaged_probe", False):
        return True

    def metal_kernel(*a, **k):
        kern = orig(*a, **k)
        return _Counted(kern, k.get("name", a[0] if a else "?"))
    metal_kernel._engaged_probe = True
    mx.fast.metal_kernel = metal_kernel
    return True


# kernel-name regex -> flags it is evidence for (a "_dp" suffix = mega_dpar variant of post/pre/attn_res)
FLAG_KERNELS = [
    (r"^inkling2_head_gemv_exact", ["head"]),
    (r"^inkling2_head_gemv_bf16w", ["head_approx"]),
    (r"^inkling2_slot_mask$", ["mask"]),
    (r"^inkling2_route_topk$", ["route"]),
    (r"^inkling2_qkv_clip$", ["qkvclip"]),
    (r"^inkling2_exact_mask$", ["xmask"]),
    (r"^inkling2_sconv_tail$", ["sconvtail"]),
    (r"^inkling2_ring_gather2$", ["mega_rg2"]),
    (r"^inkling2_attn_res_\d+", ["attnres"]),
    (r"^inkling2_attn_res_\d+_dp$", ["mega_dpar"]),
    (r"^inkling2_mega_ctrl_(down|blend)$", ["mega_ctrl"]),
    (r"^inkling2_mega_norm\d+$", ["mega_norm"]),
    (r"^inkling2_mega_dsconv$", ["mega_dsconv"]),
    (r"^inkling2_mega_router$", ["mega_moe"]),
    (r"^inkling2_mega_post\d+", ["mega_post"]),
    (r"^inkling2_mega_post\d+_dp$", ["mega_dpar"]),
    (r"^inkling2_mega_qsm2?$", ["mega_qsm"]),
    (r"^inkling2_mega_qsm2$", ["mega_qsm2"]),
    (r"^inkling2_mega_sconv2$", ["mega_sconv2"]),
    (r"^inkling2_mega_oproj2?$", ["mega_oproj"]),
    (r"^inkling2_mega_oproj2$", ["mega_oproj2"]),
    (r"^inkling2_mega_pre\d+", ["mega_pre"]),
    (r"^inkling2_mega_pre\d+_dp$", ["mega_dpar"]),
    (r"^inkling2_mega_sdpa", ["mega_sdpa"]),
]
COUNTED_FLAGS = sorted({f for _, fl in FLAG_KERNELS for f in fl})

# Counted flags that do NOT have to fire at B=1 L=1 decode (prefill/verify/B>1/unrelated-path only).
NOT_B1_DECODE = {
    "head_approx",  # rows 2..15 only; `head` handles rows == 1
}

# A counted flag has legitimately 0 hits when one of these (enabled AND engaged) replaces its kernel:
# mega_qsm fuses qkvclip + sink_proj + exact mask; mega_post fuses the sconv tail; mega_oproj2/qsm2/pre fuse the rest;
# mega_moe's router kernel replaces route_topk (route_topk still ran on integ b136ea8: both counted); xmask/mega_qsm replace the slot mask (mask is the pre-FD decode path).
SUPERSEDED = {
    "qkvclip": ["mega_qsm"],
    "xmask": ["mega_qsm"],
    "sconvtail": ["mega_post"],
    "route": ["mega_moe"],
    "mask": ["xmask", "mega_qsm"],
    "attnres": ["mega_post", "mega_pre"],
    "mega_rg2": ["mega_sdpa"],  # mega_sdpa reads the ring in place (IM-30: removes ring_gather); seen 0 hits on integ b136ea8
}


def enabled_flags():
    """Flags of the running fork if it is imported (authoritative, "all" expanded), else parsed from env."""
    m = sys.modules.get("mlx_vlm.models.inkling2.fast_kernels")
    if m is not None and hasattr(m, "FLAGS"):
        return set(m.FLAGS)
    fl = {t.strip() for t in os.environ.get("INKLING_FAST", "").split(",") if t.strip()}
    if "all" in fl:
        fl = {"head", "mask", "glue", "fuse", "fusemoe"} | (fl - {"all"})
    return fl


def replica_ok():
    m = sys.modules.get("mlx_vlm.models.inkling2.fast_kernels")
    try:
        return bool(m.replica_ok()) if m is not None else None
    except Exception:
        return None


def flag_counts(counts):
    out = collections.Counter()
    for name, n in counts.items():
        for rx, fl in FLAG_KERNELS:
            if re.search(rx, name):
                for f in fl:
                    out[f] += n
    return out


def analyse(flags, decode, prefill, replica=None):
    """Pure function (unit-testable). Returns dict(flags=[(flag, decode_count, prefill_count)], no_counter=[...],
    missing=[...], exempt=[...])."""
    d, p = flag_counts(decode), flag_counts(prefill)
    rows, no_counter, missing, exempt = [], [], [], []
    for f in sorted(flags):
        if f not in COUNTED_FLAGS:
            no_counter.append(f)
            continue
        rows.append((f, d.get(f, 0), p.get(f, 0)))
        if d.get(f, 0) == 0 and f not in NOT_B1_DECODE:
            sup = [s for s in SUPERSEDED.get(f, []) if s in flags and d.get(s, 0) > 0]
            (exempt if sup else missing).append(f if not sup else f"{f}(by {sup[0]})")
    return dict(flags=rows, no_counter=no_counter, missing=missing, exempt=exempt, replica_ok=replica)


def snapshot():
    """Plain-dict copy of the current counters + the analysis, for jsonl records."""
    fl = enabled_flags()
    a = analyse(fl, COUNTS["decode"], COUNTS["prefill"], replica_ok())
    return dict(decode={f: n for f, n, _ in a["flags"]}, prefill={f: n for f, _, n in a["flags"]},
                no_counter=a["no_counter"], missing=a["missing"], exempt=a["exempt"], replica_ok=a["replica_ok"],
                kernels_decode=dict(COUNTS["decode"]), kernels_prefill=dict(COUNTS["prefill"]),
                note="python-level kernel calls; under mx.compile = call sites in the traced step graph")


def report(require=None, out=print):
    """Print the ENGAGED lines. Returns True when engagement is OK (or not required)."""
    if require is None:
        require = os.environ.get("INKLING_REQUIRE_ENGAGED", "0") == "1"
    s = snapshot()
    out("ENGAGED: " + (" ".join(f"{f}={n}" for f, n in s["decode"].items()) or "(no counted flags enabled)")
        + "  [decode-phase kernel calls; compiled step = traced call sites]")
    out("ENGAGED-PREFILL: " + " ".join(f"{f}={n}" for f, n in s["prefill"].items()))
    out("ENGAGED-NO-COUNTER: " + (",".join(s["no_counter"]) or "-")
        + "  (no kernel counter for these INKLING_FAST flags; not checked)")
    out(f"ENGAGED-REPLICA-OK: {s['replica_ok']}")
    out("ENGAGED-KERNELS: " + " ".join(f"{k}={v}" for k, v in sorted(s["kernels_decode"].items())))
    if s["exempt"]:
        out("ENGAGED-EXEMPT: " + ",".join(s["exempt"]) + "  (0 hits because a fusing flag replaced the kernel)")
    ok = not s["missing"]
    if s["missing"]:
        out(("ENGAGED FAIL: " if require else "ENGAGED WARNING: ") + "enabled flags with 0 decode hits: "
            + ",".join(s["missing"]) + ("" if require else "  (--require-engaged would exit non-zero)"))
    else:
        out("ENGAGED OK: every counted flag fired in the decode phase")
    return ok or not require


def _selftest():
    mega = {"mega_ctrl", "mega_sdpa", "mega_qsm", "qkvclip", "xmask", "fuse", "mask", "head_approx"}
    good = collections.Counter({"inkling2_mega_ctrl_down": 48, "inkling2_mega_ctrl_blend": 48,
                                "inkling2_mega_sdpa1v2": 36, "inkling2_mega_qsm2": 48})
    a = analyse(mega, good, collections.Counter())
    assert a["missing"] == [] and "fuse" in a["no_counter"], a
    assert sorted(a["exempt"]) == ["mask(by mega_qsm)", "qkvclip(by mega_qsm)", "xmask(by mega_qsm)"], a
    bad = collections.Counter({"inkling2_mega_ctrl_down": 48, "inkling2_mega_qsm": 48})  # sdpa fell back
    assert analyse(mega, bad, collections.Counter())["missing"] == ["mega_sdpa"]
    # prefill-only hits do not count as engaged at decode
    assert "mega_sdpa" in analyse({"mega_sdpa"}, collections.Counter(), collections.Counter({"inkling2_mega_sdpa1v2": 3}))["missing"]
    # a superseding flag that itself has 0 hits does not exempt
    assert analyse({"qkvclip", "mega_qsm"}, collections.Counter(), collections.Counter())["missing"] == ["mega_qsm", "qkvclip"]
    # _dp suffix feeds mega_dpar
    assert flag_counts(collections.Counter({"inkling2_mega_post5_dp": 2, "inkling2_mega_pre5": 1}))["mega_dpar"] == 2
    print("engaged selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
