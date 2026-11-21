# Details

Full setup instructions, benchmark tables, correctness checks, known limits, and licence notes for `inkling2-mini-mlx-fast`.

## Setup, branches, and flags

The overlay targets Python 3.12, MLX 0.32.4, and `mlx-vlm` at commit `fd6404fb3261102600541127d59103dac65c2756`. That commit reports version 0.8.0, though its Inkling 2 files differ from the PyPI 0.8.0 wheel. The bitwise kernels are pinned to MLX 0.32.4 and this GPU architecture; running on another MLX version raises an error unless `INKLING_FAST_UNVERIFIED_OK=1` is set.

Three branches hold the reviewed code as-is (only licence headers and comment wording were edited):

- `main`: plain B=1 decode and stock-compatible batching.
- `batch`: `main` plus `INKLING_BATCHFAST` kernels for B>1 decode.
- `specfast`: `main` plus `INKLING_SPECFAST` kernels for speculative decoding with the model's bundled DSpark drafter.

Running `./install.sh <dir containing mlx_vlm/>` checks the upstream `language.py` SHA256 (`--force` skips the check), backs up overwritten files as `*.orig`, and copies `tests_fast/`, `tests.txt`, and `run_tests.sh` into `<dir>`. Run `./run_tests.sh python` from `<dir>` to execute the tiny random-weight unit tests (no model download needed; requires a Metal GPU).

Runtime flags (default behaviour with no flags is stock `mlx-vlm`):

- `INKLING_FAST_DECODE=1` and `INKLING_FAST=<31-flag list>`: turns on the compiled `FastDecoder` and fused Metal kernels (the full 31-flag list is `$ALL31` in `run_tests.sh`).
- `INKLING_BATCHFAST=all`: enables batched decode kernels (`batch` branch).
- `INKLING_FAST_VERIFY=1 INKLING_SPECFAST=all`: enables speculative verify and fast kernels (`specfast` branch).
- `INKLING_FAST_DECODE_PADDED=1`: opts into unequal-length (left-padded) batches.
- `INKLING_FAST_UNVERIFIED_OK=1`: bypasses the MLX version and GPU arch check.

To run benchmarks, build prompts with `INKLING_MODEL=/path/to/Inkling-2-Mini INKLING_RUNS=./runs python bench/build_prompts.py` (reads `INKLING_PROMPTS`, defaulting to `bench/prompts.json`), then run scripts in `bench/` (such as `bench/bench.py`, `bench/bench_bdec.py`, or `bench/spec_bench.py`) with `bench/` and the `mlx-vlm` tree on `PYTHONPATH`. `bench/common.py` refuses jobs estimated above 55 GB.

## Test machine and weights

Every number below was measured on a Mac with an Apple M5 Ultra (64-core GPU, `applegpu_g17d`, 96 GB unified memory), macOS 27.2.1, Python 3.12, MLX 0.32.4, and `mlx-vlm` at `fd6404fb`. Decoding is greedy, B=1 unless noted, and the GPU-utilisation sampler is off during timing. "Stock" means unmodified `mlx-vlm` at that commit with no flags.

Weights are the shipped Inkling 2 Mini release with no requantization: NVFP4 experts in layers 0-19, MXFP8 experts in layers 20-23, and bf16 for attention, controllers, router, embedding, and LM head.

## Full benchmark results

### Single-stream decode (B=1, tok/s)

| Path (branch) | Context | Drafting | Baseline | Ours | Speedup | Exactness |
|---|---|---|---|---|---|---|
| Plain (`main`) | 512 | Off | 37.3 (stock) | 75.6 (rerun 75.9) | 2.03x | BITWISE |
| Plain (`main`) | 4K | Off | 35.4 (stock) | 73.5 (rerun 73.7) | 2.08x | BITWISE |
| Plain, raw code (`main`)* | 32K | Off | 32.4 (stock) | 70.2 | 2.17x | BITWISE |
| Plain, raw code (`main`)* | 128K | Off | 22.3 (stock) | 53.3 | 2.40x | No verdict |
| Spec, n=3 (`specfast`) | <=4K | On | 78.2 / 76.3 (plain 512 / 4K) | 102.7 mean (chat 86.0, code 112.3, doc 106.3) | 1.31x | NONBITWISE-OK |
| Spec, raw code, n=3 (`specfast`)* | 4K | On | 77.0 (plain) | 131.3 | 1.71x | Unchecked |
| Spec, raw code, n=3 (`specfast`)* | 16K | On | 73.3 (plain) | 120.3 | 1.64x | Unchecked |
| Spec, raw code, n=3 (`specfast`)* | 32K | On | 68.8 (plain) | 98.7 | 1.44x | Unchecked |

*Long-context raw-code rows come from our results files and were not independently reviewed (and have no KL gate for the speculative rows). Short-context speculative runs used 8 prompts x 256 new tokens up to 4K context; an earlier author run measured 100.7 tok/s mean (chat 84.5, code 111.9, doc 103.2) against 78.1 plain. Because speculative and plain runs diverge after token 15-217, differences under 5% between configs are noise.

### Prefill (drafting off, tok/s)

Prefill is essentially unchanged; the real speedups are in decode.

- 512 / 1K / 4K (harness): 2375 / 2729 / 2920 vs 2206 / 2873 / 2766 stock. The 1K dip is a harness artifact; measured without it, prefill is about 2940 vs 2790.
- Model-level prefill: about 2900-2980 vs about 2846 at 1K, and about 2905 vs about 2752 at 4K.
- Long context: 2370 vs 2281 at 32K, and about 1264 vs about 1228 at 128K.

### Batched decode (`batch` branch, drafting off, aggregate tok/s)

Measured on equal-length rows of overlapping 512-token Python stdlib slices (all code, no chat, about 80% overlap between neighbours), 128 decode steps:

| Context | B | FastDecoder (no BATCHFAST) | Ours (`BATCHFAST=all`) | Drafting | Exactness |
|---|---|---|---|---|---|
| 512 | 4 | 155.2 | 208.6 | Off | NONBITWISE-OK |
| 512 | 8 | 234.9 | 287.8 | Off | NONBITWISE-OK |
| 512 | 16 | 303.0 | 344.5 (reps 340.5 / 348.4) | Off | NONBITWISE-OK |
| 2K | 4 | 147.7 | 199.6 | Off | NONBITWISE-OK |
| 2K | 16 | 273.8 | 330.9 | Off | NONBITWISE-OK |

For rough reference, stock `mlx-vlm` batched decode measured 111.1 / 175.6 / 249.7 tok/s at B=4 / 8 / 16 in a separate session, so only the two columns in the table above are a direct same-session comparison. At B=1 with `BATCHFAST` enabled, output stays bitwise exact (256/256 steps across 8 prompts).

## Correctness gates

We use two labels:

- `BITWISE`: logits identical to stock at every step. This holds for plain B=1 decode (0 differing steps out of 2048 across 8 prompts x 256), ~1K and ~8K starts (0/512), equal-length batches without `BATCHFAST`, 32K (2 prompts), and 64K (1 prompt). With `SPECFAST` off, the `specfast` branch also matches stock 512/512. At 96K and 128K stock is not reproducible against itself, so there is no verdict.
- `NONBITWISE-OK`: top-64 teacher-forced KL against stock stays under 2.68e-3, which is the stock-vs-stock noise measured when only the prefill chunk size changes from 2048 to 1024 (8 prompts x 256 steps, calibrated on prompts up to 4K; not meaningful at 96K+).
  - `BATCHFAST` at B>1 measures 2.1e-4 to 7.2e-4 mean KL at 256 context, with worst cases of 1.55e-3 at 1K and 1.61e-3 at 4K.
  - `SPECFAST` measures 1.72e-3 KL64 at n=3 and 1.52e-3 at n=4. Tiny FP32 verify tests pass with TF32 off (row diffs up to 1.9e-6).

## Known limits

- Context past 131072 tokens: plain decode falls back to stock; the speculative path currently crashes past 131072 (a fix exists and will land in a later update).
- Verify-pass KL near context 1000-1256: KL exceeds the 2.68e-3 bar even on the base (non-`specfast`) path, reaching 6.82e-3 for `specfast` at n=3 before settling back down at 2000. Our original gate prompts did not cross ~1024.
- Batched decode gaps: gated only up to 4K context (8K-32K could cross the bar), B=32 never run, B>16 runs partly on stock paths without warning, no chat batch test, MXFP8 MoE layers still use stock kernels, and `FastDecoder` is not wired into `batch_generate` or the server.
- Unequal-length (left-padded) batches are refused unless `INKLING_FAST_DECODE_PADDED=1` is set, and that mode is not bitwise (KL about 1e-3).
- A reference cycle keeps KV buffers alive until Python GC runs, which matters for long-running servers.
- `attnres` at L>1 is not FP32-tested.
- Multi-turn KV reuse for agent-style serving is still WIP and not in this repo.
- Prompt set caveat: the shipped prompt builder (stdlib-source windows) is not the exact short chat/code prompt set behind the numbers above (which is not redistributed), so expect somewhat different absolute speeds, especially for speculative decoding.

## Repo layout and licences

- `mlx_vlm/models/inkling2/`: new files (`fast_decode.py`, `fast_kernels.py`, `mega_kernels.py`, `verify_dedup.py`, plus `batch_kernels.py` on `batch` and `spec_attn.py`, `spec_kernels.py`, `spec_moe.py` on `specfast`), modified upstream `language.py` and `kernels.py`, and `tests_fast/`.
- `mlx_vlm/quantization/nvfp4.py`: modified upstream file caching a per-expert scale factor.
- `tests_fast/`, `tests.txt`, `run_tests.sh`, `bench/`, `install.sh`, `UPSTREAM_LANGUAGE_SHA256`, `LICENSE`, `LICENSE-APACHE`.

Licences:

- Our new files: MIT, Copyright (c) 2026 somefunprojects (`LICENSE`).
- Modified `mlx_vlm/models/inkling2/kernels.py` and `mlx_vlm/quantization/nvfp4.py`: MIT, Copyright (c) 2025 Prince Canuma (upstream notice in `LICENSE`).
- Modified `mlx_vlm/models/inkling2/language.py`: Apache-2.0, Hugging Face Team code adapted for MLX in `mlx-vlm`, keeping its header plus a "Modifications Copyright 2026 somefunprojects" notice (`LICENSE-APACHE`).
- Upstream `mlx-vlm` (Neywa-Labs/mlx-vlm, formerly Blaizzy/mlx-vlm): MIT.
