# inkling2-mini-mlx-fast

This is a small speed overlay for running the Inkling 2 Mini model (text path only) on Apple Silicon with MLX, built on top of `mlx-vlm`. It keeps the official released weights completely untouched and cuts per-token GPU overhead with a compiled decode loop and custom Metal kernels. On our Mac, that roughly doubles plain decoding speed compared with stock `mlx-vlm` while producing the exact same output, and everything sits behind opt-in flags so default runs still behave like stock `mlx-vlm`.

## What we measured

Here is what we measured on our own Mac (Apple M5 Ultra, 64-core GPU, 96 GB unified memory, MLX 0.32.4, greedy decoding, unchanged weights):

| Mode | Context | Drafting | Baseline (tok/s) | Ours (tok/s) | Bit-exact? |
|---|---|---|---|---|---|
| Plain (`main`) | 512 | Off | 37.3 (stock `mlx-vlm`) | 75.6 (2.03x) | Yes |
| Plain (`main`) | 4K | Off | 35.4 (stock `mlx-vlm`) | 73.5 (2.08x) | Yes |
| Speculative (`specfast`) | <=4K | On | 78.2 / 76.3 (plain) | 102.7 mean (1.31x) | No, within noise bar |
| Batched, B=4/8/16 (`batch`), aggregate tok/s, distinct prompts | 512 | Off | 153.7 / 232.6 / 301.6 (our plain path; stock `mlx-vlm`: 111.1 / 175.6 / 249.7) | 222.2 / 321.8 / 420.1 | No, within noise bar (up to 4K context) |

For reference, stock `mlx-lm` (the fork at commit 0c4c991) on the same machine decodes at 45.0 tok/s at 4K context, so our plain decode is about 1.6-1.7x stock `mlx-lm`. Prefill measured the same way (4,096 tokens in 1,024-token chunks, fresh cache, no `lm_head`) is 3,344 tok/s for stock `mlx-lm` and 3,122 for ours, so ours is about 7% slower. Other engines on the same machine are faster at B=1 decode: [inkling-infer by razorback16 is faster once context passes 1K, 109 tok/s on our machine](https://github.com/razorback16/inkling-infer). Details and methods are in [DETAILS.md](DETAILS.md).

"Bit-exact" means every decoding step produces identical logits to unmodified `mlx-vlm`; the speculative and batched modes reorder some math and allow tiny differences that stay within stock `mlx-vlm`'s own run-to-run noise. This project is a work in progress, and corrections or bug reports are always welcome. See [DETAILS.md](DETAILS.md) for long-context runs, prefill timings, correctness gates, known limits, and all runtime flags.

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

It prints decode speed in tok/s at 512 and 4K context; it doesn't print generated text. The `batch` and `specfast` branches have the batched and speculative modes.

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
