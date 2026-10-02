# Paged KV Cache Inference Engine

Inference-only Llama 3 implementation with a C++17 block allocator, copy-on-write
prefix sharing, and a split-K CUDA decode-attention kernel. Python/PyTorch handles
model orchestration and matrix multiplications; it does not delegate generation
to Transformers. Transformers is used for tokenization and as an independent test oracle.

## Validation status

The real Meta Llama 3 8B checkpoint has been tested on an **A100 80GB PCIe**
with PyTorch 2.8.0+cu128. The current implementation passed **103 GPU tests**.
At batch 32, prompt length 128 and 128 generated tokens, three-iteration runs
measured **1,378–1,405 output tokens/s**, including prefill and host scheduling.
Loading and warmup are excluded, and EOS is disabled. Batched causal prefill and
reusing RoPE factors across layers improved the original 530 tokens/s result.

Exact greedy output agreement passed for 8,192 generated tokens across random
and fixed-text workloads **when both cache layouts use the same tiled kernel**.
The independent SDPA baseline still differs (409 random-workload and 1,451
fixed-text token mismatches). The controlled comparison isolates cache layout;
it does not establish independent-attention bitwise parity.

Under a **38 GiB PyTorch allocation budget on the 80GB GPU**, the fixed mixed-length
workload supports 154 paged versus 44 contiguous sequences (**3.5x**), with **99.6%
less unused reserved KV storage** at the matched batch of 44. Both capacity
candidates also completed actual prompt prefill and generation. This is not an
A100 40GB hardware measurement, and the 4x target remains unmet.

Attention at batch 32 / context 2048 / BF16 reached **55.0% of forced flash SDPA
throughput on A100** (0.494 ms versus 0.272 ms). The A40 reached **83.3–84.1%**
with the same tiled kernel; the 75% target is established only on that A40 workload.
Raw results, including failed parity comparisons, are in
`validation/a100-8b-2026-10-02/` and `validation/a40-tiled-2026-10-02/`.

### Earlier iterations

The baseline implementation at commit `6010b59` passed all 29 correctness tests on an NVIDIA A40 with PyTorch 2.8.0+cu128
and CUDA 12.8 on October 1, 2026, including CUDA compilation, paged attention
in FP32/FP16/BF16, and small-model greedy output parity. All 29 also passed in
an earlier RTX 4090 run. CPU logits agree with an independent Hugging Face
Llama implementation on a small random model. The native allocator passed
10,000 randomized operations normally and with UBSan; local macOS ASan stalled
and remains unvalidated (Linux CI is configured for ASan/UBSan).

The initial A40 attention benchmark at batch 32, context 2048, BF16,
32 query heads / 8 KV heads / head dimension 128 measured 2.693 ms for this
kernel versus 0.485 ms for forced flash SDPA: **18.0% of flash throughput**.
This is one initial 100-iteration measurement, not a tuned performance result.
Raw benchmark and test records are in `validation/a40-2026-10-01/`.

The multi-warp iteration at `a85b408` passed all **53 tests** on an A40 on
October 2, 2026 UTC (October 1 Pacific). At the same benchmark shape, the
4-warp variant measured **1.313 ms**, versus **2.208 ms** for the one-warp
baseline rerun on the same host: **1.68x throughput / 40.6% lower latency**.
It reached **38.4% of forced flash SDPA throughput** (0.504 ms). These are
medians of five 100-iteration samples. The earlier 18.0% result came from a
different session; use the same-session baseline of 22.2% for this comparison.
Raw results and reproduction commands are in
`validation/a40-multiwarp-2026-10-02/`.

This is a static-batch inference engine, not an HTTP server. Equal-length prompts
without full-page prefix sharing use batched causal prefill; ragged and shared-prefix
requests retain tokenwise prefill. Decode uses host scheduling and fresh scratch
allocations. Standard unscaled Llama RoPE is supported, not Llama 3.1 scaled RoPE.

## Install and test

Requires Python 3.10+, a C++17 compiler, and PyTorch 2.5+. A CUDA toolkit with
`nvcc` and a CUDA-enabled PyTorch build are additionally required for GPU execution.
No CUDA compiler is needed for CPU tests. The CUDA extension compiles lazily on
first use, using [PyTorch's extension loader](https://docs.pytorch.org/docs/stable/cpp_extension.html).

```bash
python3 -m venv .venv
source .venv/bin/activate
# On a GPU host, install the appropriate CUDA build of torch first.
pip install -e '.[test,model,cuda]'
pytest -q
```

GPU tests skip explicitly when no CUDA device exists. On a CUDA host, the same
command compiles and tests the kernel for FP32/FP16/BF16, four head sizes, GQA,
ragged contexts, noncontiguous physical pages, and partition boundaries.

The standalone allocator sanitizer test has no Python/PyTorch dependency:

```bash
cmake -S . -B build -DENABLE_SANITIZERS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Generate from a checkpoint

Obtain an authorized local Hugging Face safetensors snapshot of
[Meta Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), including
`config.json`, tokenizer files, and all checkpoint shards. This project neither
bundles model weights nor automatically downloads them. A BF16 8B model needs
roughly 16 GB for weights before KV storage and temporary tensors.

```bash
paged-generate /path/to/Meta-Llama-3-8B \
  --prompt 'The key to a good database is' \
  --prompt 'The key to a good database is careful' \
  --max-new-tokens 64 --max-context 4096 --blocks 1024
```

Each prompt is a separate sequence. The engine prefills the common prefix once,
forks its block table, and copies a shared partial block only when a branch writes
into it. Fully populated prefix pages remain shared. Greedy generation stops at
EOS or the token limit and returns generated token IDs plus decoded text. Prompts
are raw completion text; no chat template is implicitly applied.

CPU execution is available via `--device cpu --backend reference`, but the full
8B checkpoint is expensive; unit tests use a small random model instead.

## Architecture and contracts

- `cpp/block_pool.h`: fixed-size physical block pool, free list, per-sequence
  logical page tables, reference counts, atomic batch capacity checks, and COW.
  Fork is O(prefix blocks). A decode reservation is O(batch), excluding metadata
  allocation. Freeing a sequence returns each unreferenced page to the pool.
- `kv_cache/cache.py`: device K/V storage shaped `[layers, pages, block_size,
  kv_heads, head_dim]`. Copies span every layer before either branch writes.
  A reservation returns physical token slots and padded device block tables.
- `kv_cache/csrc/decode.cu`: tiled partition attention computes QK with
  16-lane groups, reduces softmax over shared-memory logits, then accumulates V
  in FP32. Head dimensions are specialized; power-of-two page lookup uses shifts
  and masks, with division fallback for other page sizes. A second kernel merges
  partition states. The original one/four/eight-warp online kernels remain available.
  GQA maps query heads to KV heads, and empty partitions have zero mass.
- `kv_cache/model.py`: checkpoint loading, RMSNorm, split-half RoPE, grouped-query
  attention, SwiGLU, residuals, and the language-model head. Checkpoint names and
  shapes must match; unsupported model configurations fail explicitly.
- `kv_cache/engine.py`: ragged prefill, shared-prefix forking, batched greedy
  decoding, EOS release, and cleanup on exceptions.

All cache operations are single-owner: serialize calls and use one CUDA stream.
Metadata is trusted output of the allocator; the low-level CUDA operator validates
shapes, types, and devices but does not synchronize to validate every page index
or sequence length. Do not pass arbitrary metadata directly to the kernel.
Supported CUDA head dimensions are multiples of 32 up to 256.

Allocator OOM/context failures leave the entire batch unchanged. A failure after
reservation (for example, device OOM during a model layer) may leave partially
written cache state: discard the affected sequences. `Engine.generate` does this
in its `finally` cleanup. Cache sizes and the request context limit are explicit;
there is no eviction, swapping, quantization, or continuous admission scheduler.

## Reproduce measurements

Run commands from the repository root. JSON output is generated by the actual
run under `results/` (ignored by Git); selected raw sessions are archived under `validation/`.

```bash
# CPU accounting, not an empirical GPU memory/concurrency result:
python -m benchmarks.memory --lengths 128,257,512,769,1024,1537,2048,3073

# CUDA attention only, BF16, 32 query heads / 8 KV heads / 128 head dimension:
python -m benchmarks.attention --batch 32 --context 2048

# Full checkpoint, same synthetic prompts and greedy token comparison:
python -m benchmarks.generation /path/to/Meta-Llama-3-8B \
  --batch 32 --prompt-length 128 --new-tokens 128 --max-context 4096
```

The attention benchmark forces PyTorch's **flash** SDPA backend and fails if that
backend is unavailable; it never silently substitutes another baseline. It warms
up and uses CUDA events. The throughput ratio is `flash_ms / paged_ms`. Dense K/V
construction and cache initialization are excluded; custom temporary allocation
is included. The benchmark reports GPU, PyTorch/CUDA versions, shape, dtype,
latencies, and numerical error.

The generation benchmark reports wall time, generated tokens/s, peak allocated
GPU memory, reserved KV bytes, and exact token agreement. Wall time includes
prefill and Python scheduling, and excludes checkpoint loading and warmup.
The dense baseline preallocates maximum context per sequence and uses PyTorch
SDPA automatic dispatch. Its storage-gather cost is included, so use the separate
attention benchmark for a kernel-only comparison. EOS is disabled for a fixed
amount of work. A mismatch makes the benchmark fail, even if throughput is high.

Memory accounting distinguishes useful token bytes, internal page fragmentation,
and contiguous maximum-context reservation. The physical pool is allocated up
front; unused pool capacity remains reserved on the GPU. Do not equate fewer live
pages with lower process memory unless the configured pool is also smaller.
A maximum-concurrency claim additionally requires an empirical capacity sweep
with fixed model, GPU budget, scratch allowance, prompt distribution, and output
length. This repository does not infer that claim from an allocation ratio.

## Correctness coverage

Tests include randomized allocator ownership histories, reference-count/free-list
conservation, block boundary behavior, multi-branch COW, atomic OOM, context limits,
actual K/V prefix preservation, ragged greedy token parity, EOS and failure cleanup,
checkpoint roundtrip, and independent Hugging Face full-forward logits. Numerical
agreement is checked with dtype-appropriate tolerances; token parity is a stronger
workload-specific check, not a guarantee for all floating-point executions.

## Earlier multi-warp attention measurements

The earlier `a85b408` iteration added opt-in `warps=4` and `warps=8` variants. Each warp
processes a disjoint subset of a context partition, then shared-memory reduction
combines their softmax states. `warps=1` was the default baseline in that iteration.
All three variants compiled and passed the expanded GPU correctness suite:
53 tests passed, none skipped. At batch 32 / context 2048, four warps was
fastest (1.313 ms), followed by eight (1.746 ms), then one (2.208 ms).
The four-warp partition-size sweep covered 64, 128, 256, 512, and 1024 tokens;
256 had the lowest median latency. The 128 and 512 results were close
(1.331 and 1.340 ms), so this is a measured configuration, not a universal optimum.

Those measurements cover the online algorithm. The current cache default is
the tiled kernel described below. These historical attention-only measurements
do not establish an end-to-end generation speedup.

Compare candidates on the same GPU after the expanded correctness suite passes:

```bash
python -m pytest -q
python -m benchmarks.attention --warps 1 --repeats 5 --output results/warps1.json
python -m benchmarks.attention --warps 4 --repeats 5 --output results/warps4.json
python -m benchmarks.attention --warps 8 --repeats 5 --output results/warps8.json
```

The benchmark now reports median latency and individual timing samples, alternating
custom/flash timing order between repeats. Raw samples, including slower
candidates, are retained in the validation directory. No A100 or full-checkpoint
performance is implied by these A40 attention measurements.

## Tiled attention

The low-level function preserves its online default for explicit baseline use.
Select the new kernel with:

```python
paged_attention(q, k, v, tables, lengths, backend="cuda",
                algorithm="tiled", warps=4, split_size=256)
```

Tiled attention requires four warps and a partition size from 1 to 1024.
It supports the same FP32/FP16/BF16 types and head dimensions (32 through 256,
in multiples of 32). `PagedCache` defaults to `attention_algorithm="tiled"`,
`attention_warps=4`, and `split_size=256`. To reproduce the original model path,
set `attention_algorithm="online", attention_warps=1` when constructing it.

```sh
python -m benchmarks.attention --algorithm tiled --warps 4 --repeats 9 --iterations 500
```

The final A40 validation includes three longer confirmation runs, the online
baseline, and a 12-shape batch/context sweep. Shorter runs varied more: at the
main shape a five-by-100 sweep reported 91.2%, while the longer confirmations
reported 83.3–84.1%. Use the latter for the headline result. Performance varies
by workload; the sweep includes a 74.4% ratio at batch 8 / context 2048. Tests
also cover all supported head sizes, non-power-of-two pages, partition tails,
invalid padded page entries that must remain unread, extreme logits, and
small-model greedy parity for the online and tiled paths.

## Full-checkpoint validation commands

```sh
# Independent SDPA baseline: records metrics and exits nonzero on greedy mismatch.
python -m benchmarks.generation checkpoints/Meta-Llama-3-8B --budget-gib 38
# Storage-layout control: identical attention arithmetic, separately allocated caches.
python -m benchmarks.generation checkpoints/Meta-Llama-3-8B --budget-gib 38 --dense-attention cuda
python -m benchmarks.generation checkpoints/Meta-Llama-3-8B --budget-gib 38 --dense-attention cuda --text-file benchmarks/data/project-readme.txt
python -m benchmarks.capacity checkpoints/Meta-Llama-3-8B --budget-gib 38
python -m benchmarks.capacity_generation --help
```

The default contiguous baseline remains independent PyTorch SDPA. `--dense-attention cuda`
uses the same tiled kernel over contiguous maximum-context storage, and is explicitly
reported as a storage-layout control. Capacity uses fixed lengths
128,257,512,769,1024,1537,2048,3073 and four decode writes, without shared prefixes.
A 38 GiB allocator limit constrains allocations but does not emulate 40GB GPU hardware.
