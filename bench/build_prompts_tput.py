"""Build prompts_tput.json: the fixed mixed workload for throughput_bench.py (CPU only, tokenizer only).
Usage: python build_prompts_tput.py [--seed 20261121] [--pool 200]

Text sources are permissively licensed ONLY, nothing model-written and nothing self-written:
  chat (~100 tok) and doc (~4k tok): windows of the English reference text in `pydoc_data.topics` (Python stdlib, PSF licence)
  code (~1k tok): windows of stdlib .py source files (PSF licence), starting at a line boundary
No instruction wrapper is added (throughput does not depend on what the model says; output length is forced by max_tokens).
Windows are cut in token space with the Inkling tokenizer, so `raw_tokens` is exact before the chat template (the bench
chat-templates each text as one user turn, which adds ~10-20 tokens).
Request mix (fixed seed): kind 50% chat / 30% code / 20% doc, target lengths uniform in
chat 60-140, code 700-1300, doc 3200-4800 tokens; max_tokens from {64,128,192,256,384,512} with weights .20/.25/.15/.20/.10/.10."""
import argparse, glob, hashlib, json, os, random, sys
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = [("chat", 0.5, (60, 140)), ("code", 0.3, (700, 1300)), ("doc", 0.2, (3200, 4800))]
OUT_LENS = [64, 128, 192, 256, 384, 512]
OUT_W = [0.20, 0.25, 0.15, 0.20, 0.10, 0.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--pool", type=int, default=200)
    a = ap.parse_args()
    tk = Tokenizer.from_file(os.path.join(os.environ.get("INKLING_MODEL", "Inkling-2-Mini"), "tokenizer.json"))
    rng = random.Random(a.seed)

    import pydoc_data.topics as topics
    topic_text = [(k, v) for k, v in sorted(topics.topics.items()) if len(v) > 12000]  # long topics only (PSF docs)
    libdir = os.path.dirname(os.__file__)  # the stdlib directory
    files = sorted(f for f in glob.glob(os.path.join(libdir, "*.py")) + glob.glob(os.path.join(libdir, "*", "*.py"))
                   if not any(x in f for x in ("/test", "/idlelib", "site-packages", "/encodings", "/pydoc_data", "/lib2to3/fixes"))
                   and os.path.getsize(f) > 12000 and open(f, errors="ignore").read().count("def ") > 15)
    assert topic_text and files, (len(topic_text), len(files))

    def window(text, n, line_start):
        ids = tk.encode(text).ids
        if len(ids) <= n:
            return text, len(ids)
        s = rng.randrange(0, len(ids) - n)
        w = tk.decode(ids[s:s + n])
        if line_start and "\n" in w[:200]:
            w = w[w.index("\n") + 1:]  # drop the partial first line
        return w, len(tk.encode(w).ids)

    reqs = []
    for i in range(a.pool):
        kind = rng.choices([k[0] for k in KINDS], [k[1] for k in KINDS])[0]
        lo, hi = next(k[2] for k in KINDS if k[0] == kind)
        n = rng.randint(lo, hi)
        if kind == "code":
            f = rng.choice(files)
            src = open(f, errors="ignore").read()
            text, raw = window(src, n, True); source = os.path.relpath(f, libdir)
        else:  # chat / doc: windows of the stdlib reference text; doc concatenates topics if one is too short
            name, t = rng.choice(topic_text)
            while len(tk.encode(t).ids) < n:  # rare: join with the next long topic
                name2, t2 = rng.choice(topic_text); t += "\n\n" + t2; name += "+" + name2
            text, raw = window(t, n, False); source = "pydoc_data.topics:" + name
        reqs.append(dict(id=i, kind=kind, source=source, raw_tokens=raw, max_tokens=rng.choices(OUT_LENS, OUT_W)[0], text=text))
    sha = hashlib.sha256(json.dumps([(r["kind"], r["text"], r["max_tokens"]) for r in reqs]).encode()).hexdigest()[:12]
    json.dump(dict(note="throughput_bench workload; stdlib (PSF) text only; see build_prompts_tput.py", seed=a.seed,
                   python=sys.version.split()[0], sha=sha, requests=reqs),
              open(os.path.join(HERE, "prompts_tput.json"), "w"), indent=0)
    from collections import Counter
    c = Counter(r["kind"] for r in reqs)
    print("pool", a.pool, "sha", sha, dict(c), "mean raw tokens", sum(r["raw_tokens"] for r in reqs) / len(reqs),
          "mean max_tokens", sum(r["max_tokens"] for r in reqs) / len(reqs))
    print("first 20:", [(r["kind"], r["raw_tokens"], r["max_tokens"]) for r in reqs[:20]])


if __name__ == "__main__":
    main()
