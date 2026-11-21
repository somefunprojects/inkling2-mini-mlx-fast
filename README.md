# inkling2-mini-mlx-fast

This is a small speed overlay for running the Inkling 2 Mini model (text path only) on Apple Silicon with MLX, built on top of `mlx-vlm`. It keeps the official released weights completely untouched and cuts per-token GPU overhead with a compiled decode loop and custom Metal kernels. On our Mac, that roughly doubles plain decoding speed while producing the exact same output, and everything sits behind opt-in flags so default runs still behave like stock `mlx-vlm`.

## What we measured

Here is what we measured on our own Mac (Apple M5 Ultra, 64-core GPU, 96 GB unified memory, MLX 0.32.4, greedy decoding, unchanged weights):

| Mode | Context | Drafting | Baseline (tok/s) | Ours (tok/s) | Bit-exact? |
|---|---|---|---|---|---|
| Plain (`main`) | 512 | Off | 37.3 (stock) | 75.6 (2.03x) | Yes |
| Plain (`main`) | 4K | Off | 35.4 (stock) | 73.5 (2.08x) | Yes |
| Speculative (`specfast`) | <=4K | On | 78.2 / 76.3 (plain) | 102.7 mean (1.31x) | No, within noise bar |
| Batched, B=4/8/16 (`batch`) | 512 | Off | 155.2 / 234.9 / 303.0 (our plain path; stock: 111.1 / 175.6 / 249.7) | 208.6 / 287.8 / 344.5 | No, within noise bar |

"Bit-exact" means every decoding step produces identical logits to unmodified `mlx-vlm`; the speculative and batched modes reorder some math and allow tiny differences that stay within stock `mlx-vlm`'s own run-to-run noise. This project is a work in progress, and corrections or bug reports are always welcome. See [DETAILS.md](DETAILS.md) for long-context runs, prefill timings, correctness gates, known limits, and all runtime flags.

## Try it

You need Python 3.12, MLX 0.32.4, and a pinned checkout of `mlx-vlm`. Pick a branch (`main` for plain decode, `batch` for batched decode, or `specfast` for speculative decoding), install `mlx-vlm`, and run the overlay script against the directory holding `mlx_vlm/`:

```bash
pip install "mlx-vlm @ git+https://github.com/Neywa-Labs/mlx-vlm.git@fd6404fb3261102600541127d59103dac65c2756"
./install.sh /path/to/dir-containing-mlx_vlm
INKLING_FAST_DECODE=1 python your_script.py
```

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
