# Details

Full setup instructions, benchmark tables, correctness checks, known limits, and licence notes for `inkling2-mini-mlx-fast`.

## Setup, branches, and flags

The overlay targets Python 3.12, MLX 0.32.4, and `mlx-vlm` at commit `fd6404fb3261102600541127d59103dac65c2756`. That commit reports version 0.8.0, though its Inkling 2 files differ from the PyPI 0.8.0 wheel. The bitwise kernels are pinned to MLX 0.32.4 and this GPU architecture; running on another MLX version raises an error unless `INKLING_FAST_UNVERIFIED_OK=<your MLX version>` is set.

Three branches hold the code:

- `main`: plain B=1 decode and batched decode (see Known limits).
- `batch`: `main` plus `INKLING_BATCHFAST` kernels for B>1 decode.
- `specfast`: `main` plus `INKLING_SPECFAST` kernels for speculative decoding with the model's bundled DSpark drafter.

Running `./install.sh <dir containing mlx_vlm/>` checks the upstream `language.py` SHA256 (`--force` skips the check), backs up overwritten files as `*.orig`, and copies `tests_fast/`, `tests.txt`, and `run_tests.sh` into `<dir>`. Run `./run_tests.sh python` from `<dir>` to execute the tiny random-weight unit tests (no model download needed; requires a Metal GPU).

Runtime flags (existing scripts with no flags run stock `mlx-vlm`):

- `INKLING_FAST=<31-flag list>`: enables the fused Metal kernels used by `FastDecoder` (the full 31-flag list is `$ALL31` in `run_tests.sh`). Installing alone does not change an existing script, but setting `INKLING_FAST` also switches on kernels in the normal `mlx-vlm` path, where batched output can then differ from stock (see Known limits). For the compiled decoder, call `generate_fast(model, processor, prompt)` from `mlx_vlm.models.inkling2.fast_decode`, or build a `FastDecoder`. `generate_fast` takes one prompt (raw text, no chat template, or token ids), returns token ids, and stops at EOS only if given `eos_token_ids`. Its prefill keeps the last token in the last chunk, while `mlx_vlm.generate` feeds it separately, so their outputs can differ. `INKLING_FAST_DECODE=1` is read only by the scripts in `bench/`, where it makes them build a `FastDecoder`; it does not affect other scripts.
- `INKLING_BATCHFAST=all`: enables batched decode kernels (`batch` branch).
- `INKLING_FAST_VERIFY=1 INKLING_SPECFAST=all`: enables speculative verify and fast kernels (`specfast` branch).
- `INKLING_FAST_DECODE_PADDED=1`: opts into unequal-length (left-padded) batches.
- `INKLING_FAST_UNVERIFIED_OK=<your MLX version>`: skips the startup check; `mega_*` kernels stay off on an unverified MLX or GPU.

To run benchmarks, build prompts with `INKLING_MODEL=/path/to/Inkling-2-Mini INKLING_RUNS=./runs python bench/build_prompts.py` (reads `INKLING_PROMPTS`, defaulting to `bench/prompts.json`), then run scripts in `bench/` (such as `bench/bench.py`, `bench/bench_bdec.py`, or `bench/spec_bench.py`) with `bench/` and the `mlx-vlm` tree on `PYTHONPATH`. `bench/common.py` refuses jobs estimated above 55 GB.

## Test machine and weights

Every number below was measured on a Mac with an Apple M5 Ultra (64-core GPU, `applegpu_g17d`, 96 GB unified memory), macOS 27.2.1, Python 3.12, MLX 0.32.4, and `mlx-vlm` at `fd6404fb`. Decoding is greedy, B=1 unless noted. The GPU-utilisation sampler is off during timing, except for the plain 512 and 4K rows of the single-stream table (both columns and the reruns; the same rows are in the README): their script ran it inside the timed loop. "Stock `mlx-vlm`" means unmodified `mlx-vlm` at that commit with no flags.

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

The FastDecoder and `BATCHFAST` columns were measured on equal-length rows of 512 tokens that share no tokens with each other (distinct prompts: even rows are natural-language passages, odd rows non-overlapping code and documentation slices), 128 decode steps, sampler off, 3 repetitions per B, spread at most 3.3%:

| Context | B | FastDecoder (no BATCHFAST) | Ours (`BATCHFAST=all`) | Not stock (our flags on, overlapping rows) | Drafting | Exactness |
|---|---|---|---|---|---|---|
| 512 | 4 | 153.7 | 222.2 | 111.1 | Off | NONBITWISE-OK (gated at 256/1K/4K) |
| 512 | 8 | 232.6 | 321.8 | 175.6 | Off | NONBITWISE-OK (gated at 256/1K/4K) |
| 512 | 16 | 301.6 | 420.1 | 249.7 | Off | NONBITWISE-OK (gated at 256/1K/4K) |

The two middle columns come from the same session on the same rows. The column to their right is not stock, and earlier versions of this page were wrong to call it that: it is the normal `mlx-vlm` path with the same 31 `INKLING_FAST` flags on, from a separate session at an earlier commit, on overlapping rows (512-token slices 97 tokens apart), 2 repetitions. Stock wasn't timed in this loop at B>1. At B=16 the batched MoE kernels (one pass over each expert's weights for all rows routed to it) account for most of the gain over the previous batched path (354.4 tok/s at B=16 on the same rows). The shipped harness (`bench/bench_bdec.py`) builds overlapping stdlib-source slices instead; on overlapping rows B=16 measured about 0.8% higher. At B=1 with `BATCHFAST` enabled, output stayed bitwise exact in the runs we checked (MLX default settings).

Only the NVFP4 MoE layers use the new batched kernels. The 4 MXFP8 MoE layers, and all MoE layers above B=16 (with a warning on stderr), use the normal `mlx-vlm` MoE path instead, which keeps any `INKLING_FAST` kernels on, so it is not the same as stock.

### Reference points from other engines (same machine)

We ran two other engines on the same Mac to see where we stand. They use their own prompt handling, so treat the comparison as rough. Batch 1, bf16 KV cache. Decode is the median of 3 repetitions after a warm-up, with a spread of at most 1% in the figures shown. Weights are the same released Inkling 2 Mini checkpoint, loaded read-only. Prompts are token ids from our stdlib-source prompt builder.

Stock `mlx-lm` (the `jundot/mlx-lm` fork at commit 0c4c991, which `inkling-infer` pins):

| Metric | Stock `mlx-lm` | Ours (`main`) |
|---|---|---|
| Decode, 16-token prompt | 45.8 | 82.0 |
| Decode, 4000-token context | 45.0 | 76.7 |
| Prefill, 4,096 tokens, 1,024-token chunks, no `lm_head` | 3,344 | 3,122 |

Our plain decode is therefore about 1.6-1.7x stock `mlx-lm` (73.5 / 45.0 with the harness numbers above, 76.7 / 45.0 from separate runs on the same prompt ids). Measured the same way, our prefill is about 7% slower than stock `mlx-lm`: a fresh 4,096-token cache, 1,024-token chunks, hidden state and cache evaluated per chunk, `lm_head` never run, median of 5 repetitions. Per 1,024-token chunk that is 290 / 310 / 316 / 305 ms for stock `mlx-lm` and 306 / 331 / 332 / 338 ms for ours (chunk offsets 0 / 1K / 2K / 3K). Timing the two components separately points to attention: stock `mlx-lm` uses one fused relative-attention kernel, while ours builds a dense mask and calls SDPA. The MoE experts take the same time in both (about 130 ms per chunk), and our short convolution is faster.

Other prefill figures differ by definition and are not directly comparable with this row: stock `mlx-lm` reaches up to about 3.7K tok/s with a 2,048-token chunk and no `lm_head`, and the prefill figures in the Prefill section above come from a harness that also computes the last-row logits.

[inkling-infer](https://github.com/razorback16/inkling-infer) (repo commit 87d0e22), run locally with unmodified code, timed by our own greedy driver around its kernels: decode is 109.1 tok/s at 4000-token context, faster than ours (76.7 on the same prompt ids, in an earlier separate run), and about 67 tok/s for prompts under 1K (ours: 82.0). Its fast path appears to engage once the sliding-window ring buffers fill, around 1K of context. Its prefill kernels measured 1.9-2.2K tok/s here, below ours and below its own stock path. Its decode is not bitwise with stock `mlx-lm` (teacher-forced mean KL 1.3-1.8e-3 over 41 steps, 2-3 argmax flips per run). We read its code for safety only and copied nothing.

## Correctness gates

We use two labels:

- `BITWISE`: fp32 logits identical to stock `mlx-vlm` at every checked step. We claim it only for these plain B=1 runs, with the full flag list, `MLX_ENABLE_TF32` unset and stock prefilled by the same loop as ours (2048-token chunks, last token included; not `mlx_vlm.generate`'s split): 8 prompts x 256 steps, starting at 43-64, ~1.5K and ~4K tokens (0/2048 differing), the same 3 texts, each cut to 300, 512 and 768 tokens, x 256 decode steps per cut (on `main`: 0/2,313 logit rows differing, prefill logits included, both in `FastDecoder` and on the normal `mlx-vlm` path with the flags), ~1K and ~8K starts (0/512), 32K (2 prompts x 128 steps) and 64K (1 prompt x 128 steps). With `SPECFAST` off, the `specfast` branch also matches stock `mlx-vlm` 512/512. Batched decode is not claimed `BITWISE`: see Known limits. At 96K and 128K stock does not reproduce itself, so there is no verdict. Separately, on the tiny random-weight model with the full flag list (`BATCHFAST` and `SPECFAST` off), B=1 matched stock in all 336 short-prompt runs: 1, 2, 3, 4, 8, 20 and 40 tokens, two seeds, all three branches, `MLX_ENABLE_TF32` unset and 0, two decoder settings, two repeats each, 89-128 steps compared.
- `NONBITWISE-OK`: mean teacher-forced KL over a gate run is at most 2.68e-3 (`BATCHFAST`: full vocabulary, against batched decode on our tree with no flags; other gates and the bar itself: top-64). Single prompts and steps can be higher. The bar is our tree with no flags (an earlier commit) at prefill chunk 1024 against stock at 2048, averaged over 8 prompts x 256 steps up to 4K (the four short prompts differ by 0, the four longer ones by 5.0e-3 to 5.7e-3). It is not the difference between identical runs, and not meaningful at 96K+.
  - `BATCHFAST` at B>1 measures 2.1e-4 to 7.2e-4 mean KL at 256 context, with worst cases of 1.55e-3 at 1K and 1.61e-3 at 4K. Each gate was 24 steps; single steps reach 0.20. The gate rows were not the distinct prompts of the speed table: they were overlapping document slices, offset by 97 tokens.
  - `SPECFAST` measures 1.72e-3 KL64 at n=3 and 1.52e-3 at n=4.

## Known limits

- Correction: earlier versions of this page listed equal-length batches without `BATCHFAST` under `BITWISE`; that was wrong.
- Batches (B>1) are not bitwise. With `INKLING_FAST` set, they can differ from stock both in `FastDecoder` and on the normal `mlx-vlm` path (no `FastDecoder`). The cause is unknown and we have no workaround. On a tiny random-weight bfloat16 model with equal-length rows, the largest differences we saw, by environment:
  - `MLX_ENABLE_TF32` unset, no `BATCHFAST`: max |dlogit| 1.23e-2, with no top-token changes in that case (4-token rows, B=8, 100 of 124 steps differing).
  - `MLX_ENABLE_TF32=0`, no `BATCHFAST`: max |dlogit| 0.894, and up to 8 top-token changes per case. In one case (4-token rows, B=4) the prefill logits already differed.
  - `batch` with `BATCHFAST=all`, `MLX_ENABLE_TF32` unset: max |dlogit| 0.728 and 18 top-token changes (4-token rows, B=8).
  On real weights, `BATCHFAST=all` with `MLX_ENABLE_TF32` unset differed at every decode step of every row in the gates under Correctness gates: max |dlogit| 7.34 (B=16, 4K) and at most 3 top-token changes per case (B=16, 1K: 3 of 384 step-row pairs). Without `BATCHFAST`, left-padded real-weight batches also differed in `FastDecoder` (opt-in, see the unequal-length limit below); equal-length ones were compared in a few configurations only, too few to say whether they can differ.
- Test gaps: `run_tests.sh` passes with an empty or missing test list, some shipped tests don't check that the kernel they cover ran, and `test_batch_fd.py` can miss wrong output.
- The gate at 300, 512 and 768 tokens (see Correctness gates) doesn't cover three things: whether the 10 flags without a kernel counter engaged (`ctrlbatch`, `fuse`, `fusemoe`, `glue`, `mega_attnres`, `mega_kvip`, `mega_kvz`, `moesort`, `msgpack`, `xpass`; of the other 21, the `xmask` and `mega_rg2` kernels never ran here: in `FastDecoder` other flags' kernels replace them), batches (B>1), and `mlx_vlm.generate`. It compared logits only; the speed numbers on this page were not timed again.
- Decoding past 131072 tokens: plain decode falls back to the normal `mlx-vlm` path, which keeps any `INKLING_FAST` kernels on, so it is not the same as stock and has the batch limit above. At B=1 that hand-off is untested for bitwise output (`test_spec_fallback` on `specfast` compares tokens only, in FP32). A prompt already too long for that limit is refused by `FastDecoder.from_cache`; `generate_fast` then uses the normal path. The speculative path currently crashes past 131072 (a fix exists and will land in a later update).
- Verify-pass KL near context 1000-1256: KL exceeds the 2.68e-3 bar even on the base (non-`specfast`) path, reaching 6.82e-3 for `specfast` at n=3 before settling back down at 2000. Our original gate prompts did not cross ~1024.
- Batched decode gaps: gated only up to 4K context (8K-32K could cross the bar), B=32 never run, B>16 runs the MoE layers on the normal `mlx-vlm` path, with any `INKLING_FAST` kernels on (with a warning), no chat batch test, MXFP8 MoE layers don't use the batched kernels, and `FastDecoder` is not wired into `batch_generate` or the server.
- Unequal-length (left-padded) batches are refused in the default exact mode unless `INKLING_FAST_DECODE_PADDED=1` is set, and that mode is not bitwise (KL about 1e-3).
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
