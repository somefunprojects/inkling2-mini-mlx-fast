"""Long-context text for the long-context gate/bench (longctx-bench). Python stdlib source only (PSF licence); no
model-generated text. Deterministic: variant "a" walks the stdlib in sorted order, "b" in reverse-sorted order, so the
two prompts differ. CPU only (tokenizers). build_ids(tk, n, variant) -> first n token ids of the concatenation."""
import os, sysconfig

def _files(variant):
    root = sysconfig.get_paths()["stdlib"]
    out = []
    for r, d, f in os.walk(root):
        d[:] = sorted(x for x in d if x not in ("site-packages", "test", "tests", "idlelib", "turtledemo", "__pycache__"))
        for x in sorted(f):
            if x.endswith(".py"): out.append(os.path.join(r, x))
    out.sort()
    return out if variant == "a" else out[::-1]

def build_text(target_chars, variant="a"):
    parts, n = [], 0
    for p in _files(variant):
        try: s = open(p, encoding="utf-8").read()
        except Exception: continue
        parts.append(f"# ===== {os.path.basename(p)} =====\n{s}"); n += len(parts[-1])
        if n >= target_chars: break
    return "\n\n".join(parts)

def build_ids(tk, n, variant="a"):
    """tk: tokenizers.Tokenizer. First n ids of the stdlib concatenation (~5 chars/token headroom x1.3)."""
    text = build_text(int(n * 5.5), variant)
    ids = tk.encode(text)
    ids = ids.ids if hasattr(ids, 'ids') else list(ids)  # tokenizers.Tokenizer -> Encoding; HF tokenizer -> list
    assert len(ids) >= n, f"corpus too small: {len(ids)} < {n}"
    return ids[:n]
