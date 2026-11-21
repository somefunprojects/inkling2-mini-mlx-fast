"""Build prompts.json for check.py / make_ref.py / bench.py / spec_bench*.py / bench_bdec.py (CPU only, tokenizer only).

Usage: INKLING_MODEL=/path/to/Inkling-2-Mini python build_prompts.py [--out prompts.json]

Every prompt is a token-exact window of Python standard-library source (PSF licence); no instruction text is added.
This is NOT the prompt set behind the published numbers (that set also held short chat/code prompts that are not
redistributed here), so absolute tok/s will differ somewhat, mostly for speculative decoding, which is prompt-sensitive.
Any file of the same shape works: {"prompts": [{"name": str, "kind": "chat"|"code"|"doc_<x>", "text": str}, ...]}.
"""
import argparse, importlib, json, os
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = [  # (name, kind, stdlib module, raw tokens)
    ("code_textwrap", "code", "textwrap", 400),
    ("code_heapq", "code", "heapq", 400),
    ("doc_1p5k_a", "doc_1p5k", "fractions", 1400),
    ("doc_1p5k_b", "doc_1p5k", "shlex", 1400),
    ("doc_4k_a", "doc_4k", "argparse", 3900),
    ("doc_4k_b", "doc_4k", "_pydecimal", 3900),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "prompts.json")); a = ap.parse_args()
    tk = Tokenizer.from_file(os.path.join(os.environ.get("INKLING_MODEL", "Inkling-2-Mini"), "tokenizer.json"))
    out = []
    for name, kind, mod, n in SPEC:
        ids = tk.encode(open(importlib.import_module(mod).__file__).read()).ids[:n]
        text = tk.decode(ids)
        out.append(dict(name=name, kind=kind, source=f"stdlib:{mod}", text=text, raw_tokens=len(tk.encode(text).ids)))
        print(name, out[-1]["raw_tokens"])
    json.dump({"note": "stdlib (PSF) source windows; raw_tokens excludes the chat template", "prompts": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
