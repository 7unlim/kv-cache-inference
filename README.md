# Paged KV Cache Inference Engine

Inference-only Llama 3 implementation with a C++17 block allocator, copy-on-write
prefix sharing, and a split-K CUDA decode-attention kernel. Python/PyTorch handles
model orchestration and matrix multiplications; it does not delegate generation
to Transformers. Transformers is used for tokenization and as an independent test oracle.

## Validation status

CPU correctness is tested, including logits against Hugging Face's independent
Llama implementation on a small random model. Local validation on Apple Silicon:
16 Python tests passed, 13 CUDA tests skipped; 10,000 randomized native allocator
operations passed both normally and with UBSan. The combined ASan/UBSan executable
stalled locally and was terminated; ASan is not locally validated. Linux CI is
configured to run both sanitizers. CUDA source and GPU benchmarks are
provided, but **CUDA compilation, Llama 3 8B checkpoint execution, and A100 performance
have not yet been validated**. No A100 throughput, memory reduction, or concurrency
claim is an established result of this repository. In particular, 1,100 tokens/s,
75% of flash SDPA, and 4x concurrent sequences are unverified targets.

This is a static-batch inference engine, not an HTTP server. It currently uses
token-at-a-time prefill, host-side scheduling, and fresh attention scratch buffers
per call. Those are explicit performance limitations. It supports standard unscaled
Llama RoPE (including Llama 3 8B), not Llama 3.1's scaled RoPE or arbitrary architectures.

## Install and test

Requires Python 3.10+, a C++17 compiler, and PyTorch 2.5+. A CUDA toolkit with
`nvcc` and a CUDA-enabled PyTorch build are additionally required for GPU execution.
No CUDA compiler is needed for CPU tests. The CUDA extension compiles lazily on
first use, using [PyTorch's extension loader](https://docs.pytorch.org/docs/stable/cpp_extension.html).

```bash
python3 -m venv .venv
source .venv/bin/activate
# On a GPU host, install the appropriate CUDA build of torch first.
pip install -e '.[test,model]'
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
- `kv_cache/csrc/decode.cu`: one warp per sequence/query-head/context partition;
  GQA maps query heads to KV heads. Warp shuffles reduce QK dots, online softmax
  accumulates weighted V in FP32, and a second kernel merges partition outputs
  using their maxima and exponential sums. Empty partitions have zero mass.
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
run and ignored by Git; no fabricated benchmark result is checked in.

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
