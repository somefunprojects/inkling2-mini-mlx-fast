# Details

Full setup instructions, benchmark tables, correctness checks, known limits, and licence notes for `inkling2-mini-mlx-fast`.

## Setup, branches, and flags

The overlay targets Python 3.12, MLX 0.32.4, and `mlx-vlm` at commit `fd6404fb3261102600541127d59103dac65c2756`. That commit reports version 0.8.0, though its Inkling 2 files differ from the PyPI 0.8.0 wheel. The bitwise kernels are pinned to MLX 0.32.4 and this GPU architecture; running on another MLX version raises an error unless `INKLING_FAST_UNVERIFIED_OK=1` is set.

Three branches hold the reviewed code as-is (only licence headers and comment wording were edited):

- `main`: plain B=1 decode and batching that stays compatible with stock `mlx-vlm`.
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

Every number below was measured on a Mac with an Apple M5 Ultra (64-core GPU, `applegpu_g17d`, 96 GB unified memory), macOS 27.2.1, Python 3.12, MLX 0.32.4, and `mlx-vlm` at `fd6404fb`. Decoding is greedy, B=1 unless noted, and the GPU-utilisation sampler is off during timing. "Stock `mlx-vlm`" means unmodified `mlx-vlm` at that commit with no flags.

Weights are the shipped Inkling 2 Mini release with no requantization: NVFP4 experts in layers 0-19, MXFP8 experts in layers 20-23, and bf16 for attention, controllers, router, embedding, and LM head.

## Full benchmark results

### Single-stream decode (B=1, tok/s)

| Path (branch) | Context | Drafting | Baseline | Ours | Speedup | Exactness |
|---|---|---|---|---|---|---|
| Plain (`main`) | 512 | Off | 37.3 (stock `mlx-vlm`) | 75.6 (rerun 75.9) | 2.03x | BITWISE |
| Plain (`main`) | 4K | Off | 35.4 (stock `mlx-vlm`) | 73.5 (rerun 73.7) | 2.08x | BITWISE |
| Plain, raw code (`main`)* | 32K | Off | 32.4 (stock `mlx-vlm`) | 70.2 | 2.17x | BITWISE |
| Plain, raw code (`main`)* | 128K | Off | 22.3 (stock `mlx-vlm`) | 53.3 | 2.40x | No verdict |
| Spec, n=3 (`specfast`) | <=4K | On | 78.2 / 76.3 (plain 512 / 4K) | 102.7 mean (chat 86.0, code 112.3, doc 106.3) | 1.31x | NONBITWISE-OK |
| Spec, raw code, n=3 (`specfast`)* | 4K | On | 77.0 (plain) | 131.3 | 1.71x | Unchecked |
| Spec, raw code, n=3 (`specfast`)* | 16K | On | 73.3 (plain) | 120.3 | 1.64x | Unchecked |
| Spec, raw code, n=3 (`specfast`)* | 32K | On | 68.8 (plain) | 98.7 | 1.44x | Unchecked |

*Long-context raw-code rows come from our results files and were not independently reviewed (and have no KL gate for the speculative rows). Short-context speculative runs used 8 prompts x 256 new tokens up to 4K context; an earlier author run measured 100.7 tok/s mean (chat 84.5, code 111.9, doc 103.2) against 78.1 plain. Because speculative and plain runs diverge after token 15-217, differences under 5% between configs are noise.

### Prefill (drafting off, tok/s)

Prefill is essentially unchanged; the real speedups are in decode.

- 512 / 1K / 4K (harness): 2375 / 2729 / 2920 vs 2206 / 2873 / 2766 for stock `mlx-vlm`. The 1K dip is a harness artifact; measured without it, prefill is about 2940 vs 2790.
- Model-level prefill: about 2900-2980 vs about 2846 at 1K, and about 2905 vs about 2752 at 4K.
- Long context: 2370 vs 2281 at 32K, and about 1264 vs about 1228 at 128K.

### Batched decode (`batch` branch, drafting off, aggregate tok/s)

Measured on equal-length rows of 512 tokens that share no tokens with each other (distinct prompts: even rows are natural-language passages, odd rows non-overlapping code and documentation slices), 128 decode steps, sampler off, 3 repetitions in one session, spread at most 3.3%:

| Context | B | FastDecoder (no BATCHFAST) | Ours (`BATCHFAST=all`) | Stock `mlx-vlm` | Drafting | Exactness |
|---|---|---|---|---|---|---|
| 512 | 4 | 153.7 | 222.2 | 111.1 | Off | NONBITWISE-OK |
| 512 | 8 | 232.6 | 321.8 | 175.6 | Off | NONBITWISE-OK |
| 512 | 16 | 301.6 | 420.1 | 249.7 | Off | NONBITWISE-OK |

The two middle columns come from the same session on the same rows. The stock `mlx-vlm` column is from a separate session. At B=16 the batched MoE kernels (one pass over each expert's weights for all rows routed to it) account for most of the gain over the previous batched path (354.4 tok/s at B=16 on the same rows). The shipped harness (`bench/bench_bdec.py`) builds overlapping stdlib-source slices instead; on overlapping rows B=16 measured about 0.8% higher. At B=1 with `BATCHFAST` enabled, output stays bitwise exact.

Only the NVFP4 MoE layers use the new batched kernels. The 4 MXFP8 MoE layers still use stock `mlx-vlm` kernels. The batched MoE path falls back to stock `mlx-vlm` above B=16 (with a one-time warning on stderr).

### Reference points from other engines (same machine)

We ran two other engines on the same Mac to see where we stand. They use their own prompt handling, so treat the comparison as rough. Batch 1, bf16 KV cache, median of 3 repetitions after 1 warm-up, spread at most 0.3%. Weights are the same released Inkling 2 Mini checkpoint, loaded read-only. Prompts are token ids from our stdlib-source prompt builder.

Stock `mlx-lm` (the `jundot/mlx-lm` fork at commit 0c4c991, which `inkling-infer` pins):

| Metric | Stock `mlx-lm` | Ours (`main`) |
|---|---|---|
| Decode, 16-token prompt | 45.8 | 82.0 |
| Decode, 4000-token context | 45.0 | 76.7 |
| Prefill, 4,096 tokens, 1,024-token chunks, no `lm_head` | 3,344 | 3,122 |

Our plain decode is therefore about 1.6-1.7x stock `mlx-lm` (73.5 / 45.0 with the harness numbers above, 76.7 / 45.0 in this same-session check). Measured the same way, our prefill is about 7% slower than stock `mlx-lm`: a fresh 4,096-token cache, 1,024-token chunks, hidden state and cache evaluated per chunk, `lm_head` never run, median of 5 repetitions. Per 1,024-token chunk that is 290 / 310 / 316 / 305 ms for stock `mlx-lm` and 306 / 331 / 332 / 338 ms for ours (chunk offsets 0 / 1K / 2K / 3K). Timing the two components separately points to attention: stock `mlx-lm` uses one fused relative-attention kernel, while ours builds a dense mask and calls SDPA. The MoE experts take the same time in both (about 130 ms per chunk), and our short convolution is faster.

Other prefill figures differ by definition and are not directly comparable with this row: stock `mlx-lm` reaches up to about 3.7K tok/s with a 2,048-token chunk and no `lm_head`, and the prefill figures in the Prefill section above come from a harness that also computes the last-row logits.

[inkling-infer](https://github.com/razorback16/inkling-infer) (repo commit 87d0e22), run locally with its own scripts and unmodified code: decode is 109.1 tok/s at 4000-token context, faster than ours (76.7 in the same check), and about 67 tok/s for prompts under 1K (ours: 82.0). Its fast path appears to engage once the sliding-window ring buffers fill, around 1K of context. Its prefill kernels measured 1.9-2.2K tok/s here, below ours and below its own stock path. Its decode is not bitwise with stock `mlx-lm` (teacher-forced mean KL 1.3-1.8e-3 over 41 steps, 2-3 argmax flips per run). We read its code for safety only and copied nothing.

## Correctness gates

We use two labels:

- `BITWISE`: logits identical to stock `mlx-vlm` at every step. This holds for plain B=1 decode (0 differing steps out of 2048 across 8 prompts x 256), ~1K and ~8K starts (0/512), equal-length batches without `BATCHFAST`, 32K (2 prompts), and 64K (1 prompt). With `SPECFAST` off, the `specfast` branch also matches stock `mlx-vlm` 512/512. At 96K and 128K stock `mlx-vlm` is not reproducible against itself, so there is no verdict.
- `NONBITWISE-OK`: top-64 teacher-forced KL against stock `mlx-vlm` stays under 2.68e-3, which is the run-to-run noise of stock `mlx-vlm` against itself measured when only the prefill chunk size changes from 2048 to 1024 (8 prompts x 256 steps, calibrated on prompts up to 4K; not meaningful at 96K+).
  - `BATCHFAST` at B>1 measures 2.1e-4 to 7.2e-4 mean KL at 256 context, with worst cases of 1.55e-3 at 1K and 1.61e-3 at 4K.
  - `SPECFAST` measures 1.72e-3 KL64 at n=3 and 1.52e-3 at n=4. Tiny FP32 verify tests pass with TF32 off (row diffs up to 1.9e-6).

## Known limits

- Context past 131072 tokens: plain decode falls back to stock `mlx-vlm`; the speculative path currently crashes past 131072 (a fix exists and will land in a later update).
- Verify-pass KL near context 1000-1256: KL exceeds the 2.68e-3 bar even on the base (non-`specfast`) path, reaching 6.82e-3 for `specfast` at n=3 before settling back down at 2000. Our original gate prompts did not cross ~1024.
- Batched decode gaps: gated only up to 4K context (8K-32K could cross the bar), B=32 never run, B>16 runs the MoE layers on stock `mlx-vlm` kernels (one-time warning), no chat batch test, MXFP8 MoE layers still use stock `mlx-vlm` kernels, and `FastDecoder` is not wired into `batch_generate` or the server.
- Unequal-length (left-padded) batches are refused unless `INKLING_FAST_DECODE_PADDED=1` is set, and that mode is not bitwise (KL about 1e-3).
- Prefill is about 7% slower than stock `mlx-lm` when both are timed the same way (see the reference-points section); the difference is in attention.
- A reference cycle keeps KV buffers alive until Python GC runs, which matters for long-running servers.
- `attnres` at L>1 is not FP32-tested.
- Multi-turn KV reuse for agent-style serving is still WIP and not in this repo.
- Prompt set caveat: the shipped prompt builder (stdlib-source windows) is not the exact short chat/code prompt set behind the numbers above (which is not redistributed), so expect somewhat different absolute speeds, especially for speculative decoding.

## Repo layout and licences

- `mlx_vlm/models/inkling2/`: new files (`fast_decode.py`, `fast_kernels.py`, `mega_kernels.py`, `verify_dedup.py`, plus `batch_kernels.py` on `batch` only, and `spec_attn.py`, `spec_kernels.py`, `spec_moe.py` on both `batch` and `specfast`; `main` has none of them. `batch` carries the spec files because its B-row kernels reuse them, but the speculative path is documented and reviewed on `specfast` only), modified upstream `language.py` and `kernels.py`, and `tests_fast/`.
- `mlx_vlm/quantization/nvfp4.py`: modified upstream file caching a per-expert scale factor.
- `tests_fast/`, `tests.txt`, `run_tests.sh`, `bench/`, `install.sh`, `UPSTREAM_LANGUAGE_SHA256`, `LICENSE`, `LICENSE-APACHE`.

Licences:

- Our new files: MIT, Copyright (c) 2026 somefunprojects (`LICENSE`).
- Modified `mlx_vlm/models/inkling2/kernels.py` and `mlx_vlm/quantization/nvfp4.py`: MIT, Copyright (c) 2025 Prince Canuma (upstream notice in `LICENSE`).
- Modified `mlx_vlm/models/inkling2/language.py`: Apache-2.0, Hugging Face Team code adapted for MLX in `mlx-vlm`, keeping its header plus a "Modifications Copyright 2026 somefunprojects" notice (`LICENSE-APACHE`).
- Upstream `mlx-vlm` (Neywa-Labs/mlx-vlm, formerly Blaizzy/mlx-vlm): MIT.
