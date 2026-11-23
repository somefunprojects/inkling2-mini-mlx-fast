# inkling2-mini-mlx-fast

> **Archived (November 2026).** We've stopped working on this. For running Inkling 2 Mini on a Mac we now use [mlx-serve](https://github.com/ddalcu/mlx-serve), which serves many requests at once. Everything here is kept as a record and isn't maintained.
>
> The branches: `main` has the plain fast decode plus greedy speculative decoding at B=1 (the old `specfast` branch, merged in). `batch` has the B>1 batched decode kernels; it was never merged into `main` and lives only on its own branch.

This is a small speed overlay for running the Inkling 2 Mini model (text path only) on Apple Silicon with MLX, built on top of `mlx-vlm`. It keeps the official released weights completely untouched and cuts per-token GPU overhead with a compiled decode loop and custom Metal kernels. On our Mac, that roughly doubles plain decoding speed compared with stock `mlx-vlm`, while keeping identical logits in the single-prompt (B=1), default-setting plain runs we list as bit-exact (see the note below the table; drafting, batches, and one known non-default case differ). Everything sits behind opt-in flags, so default runs still behave like stock `mlx-vlm`.

## What we measured

Here is what we measured on our own Mac (Apple M5 Ultra, 64-core GPU, 96 GB unified memory, MLX 0.32.4, greedy decoding, unchanged weights):

| Mode | Context | Drafting | Baseline (tok/s) | Ours (tok/s) | Bit-exact? |
|---|---|---|---|---|---|
| Plain (`main`) | 512 | Off | 37.3 (stock `mlx-vlm`) | 75.6 (2.03x) | Yes, as tested (B=1) |
| Plain (`main`) | 4K | Off | 35.4 (stock `mlx-vlm`) | 73.5 (2.08x) | Yes, as tested (B=1) |
| Speculative (`main`), 8 prompts | 43-64, ~1.5K and ~4K | On (DSpark, n=3) | 78.6 mean (plain) | 101.9 mean (1.30x) | No; pooled mean KL 1.72e-3 on these prompts only (bar 2.68e-3; one prompt 3.24e-3). An earlier check near 1K context was over the bar (see [DETAILS.md](DETAILS.md)) |
| Speculative (`main`), code-writing prompts, 2,048-token replies | ~1K / ~4.3K | On (DSpark, n=3) | 78.1 / 77.3 median (plain) | 118.8 / 113.7 median (1.52x / 1.47x) | No; KL not measured on these prompts |
| Batched, B=4/8/16 (`batch`), aggregate tok/s, distinct prompts | 512 | Off | 153.7 / 232.6 / 301.6 (our plain path; stock `mlx-vlm` wasn't timed in this test) | 222.2 / 321.8 / 420.1 | No; mean KL within the 2.68e-3 bar in 24-step gates at 256, 1K and 4K context |

The two speculative rows (both speed columns) and the drafting figures below were timed on the merged code. The other speeds in the table are from before the merge, and the two plain rows ran with the GPU-utilisation sampler on, which cost our side 3-4% (its cost to stock wasn't measured). The merge gate's Try-it run timed our side of those two again with it off (78.4 / 76.5 tok/s at 512 / 4K); stock wasn't timed again.

For reference, stock `mlx-lm` (the fork at commit 0c4c991) on the same machine decodes at 45.0 tok/s at 4K context, so our plain decode is about 1.6-1.7x stock `mlx-lm`. Prefill measured the same way (4,096 tokens in 1,024-token chunks, fresh cache, no `lm_head`) is 3,344 tok/s for stock `mlx-lm` and 3,122 for ours, so ours is about 7% slower. Other engines on the same machine can be faster than our plain B=1 decode: [inkling-infer by razorback16 is faster once context passes 1K, 109 tok/s on our machine](https://github.com/razorback16/inkling-infer). Details and methods are in [DETAILS.md](DETAILS.md).

"Bit-exact" means identical logits to unmodified `mlx-vlm` at every step we checked: plain B=1 decode (drafting off) with the full flag list and MLX's default settings, on the prompts and lengths in [DETAILS.md](DETAILS.md). The speculative and batched modes change the math, so logits differ and generated tokens can change. We compare their mean KL with a 2.68e-3 bar (our tree with no flags at a 1024-token prefill chunk against stock at 2048, averaged over 8 prompts; not two identical runs). For drafting, that bar is on the pooled mean across prompts, so a single prompt can sit above it (pooled mean 1.72e-3 over 8 prompts and 2,033 steps; one prompt alone was 3.24e-3).

Drafting (n=3 means three drafted tokens per round), in short:

- It isn't always faster. On code prompts, the gain shrank beyond 4K (1.44x at 4K, 1.38x at 16K, 1.25x at 32K; 512-token replies). Around 115-123K tokens it was slower than plain over a whole 128-token reply, at 46.2 vs 52.9 tok/s median (5 runs), mostly because of a slow first drafting round (about 0.6 s); after that round the five runs were too mixed to call faster or slower. See [DETAILS.md](DETAILS.md) for the table of other shapes.
- It changes greedy text: on 50 questions (64 tokens), 34 of 50 replies were identical to plain and 16 differed; asking again matched 50 of 50 times with drafting off and with it on. It is greedy only, and was gated on real weights and timed one prompt at a time (B=1) only.
- The fast decoder refuses prompts of 131072 tokens or more. If the prompt plus the requested reply runs two or more tokens past that limit, the `FastDecoder` drafting loops in `bench/` crash (and can at one token past); the library's `speculative_loop` switches to plain decoding instead (tested on a tiny model only).

This project is a work in progress, and corrections or bug reports are always welcome. See [DETAILS.md](DETAILS.md) for long-context runs, prefill timings, correctness gates, known limits, and all runtime flags.

## Try it

You need Python 3.12 and a local copy of the Inkling 2 Mini weights. These commands set up `main` and run the plain decoding benchmark:

```bash
git clone https://github.com/somefunprojects/inkling2-mini-mlx-fast
cd inkling2-mini-mlx-fast
python3.12 -m venv .venv
. .venv/bin/activate
pip install mlx==0.32.4 "mlx-vlm @ git+https://github.com/Neywa-Labs/mlx-vlm.git@fd6404fb3261102600541127d59103dac65c2756"
./install.sh .venv/lib/python3.12/site-packages

export INKLING_MODEL=/path/to/Inkling-2-Mini
python bench/build_prompts.py
export INKLING_FAST=$(sed -n 's/^ALL31=//p' run_tests.sh)
export INKLING_FAST_DECODE=1
python bench/bench.py --skip-batch --sampler off
```

It prints plain decode speed in tok/s at 512 and 4K context; it doesn't print generated text. Drafting is now in `main`. It needs a draft loop (`bench/spec_bench2.py` has one; `generate_fast` doesn't draft). `INKLING_SPECFAST=all` turns on the faster verify kernels we timed, and the `FastDecoder` drafting benchmarks also need `INKLING_FAST_VERIFY=1`. Faster batched decode stays on the separate `batch` branch.

Check [DETAILS.md](DETAILS.md) for the test runner, kernel flags, and benchmark scripts.

## Licences

New files in this repo are MIT (`LICENSE`). Modified `mlx-vlm` files keep their upstream MIT (`LICENSE`) and Apache-2.0 (`LICENSE-APACHE`) licences.

## Thank you

We learned a lot from generous public work:

- [MLX](https://github.com/ml-explore/mlx) for the framework and Metal kernels several of ours mirror.
- [mlx-vlm](https://github.com/Neywa-Labs/mlx-vlm) (Prince Canuma / Neywa-Labs) for the Inkling 2 port, and Hugging Face Transformers for `language.py`.
- [EXO Labs](https://x.com/exolabs/status/2123953913342935486) for showing how much headroom the M5 Ultra has, and exo PR #2530 for fused MoE combine ideas (ideas only, no code copied).
- alesha-pro for [maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) and [inference-optimization-playbook](https://github.com/alesha-pro/inference-optimization-playbook).
- mlx-lm PR #2086 and mlx-community [Inkling-2-Mini-4bit](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit).
- [oMLX](https://omlx.ai/benchmarks/performance/rj6pgg85) for the M5 Ultra benchmark table format.
- realugbun for the [llama.cpp inkling2-speed fork](https://github.com/realugbun/llama.cpp) (fusion ideas and depth-table layout).
