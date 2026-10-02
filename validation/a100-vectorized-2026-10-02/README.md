# A100 vectorized attention experiment — 2026-10-02 UTC

This iteration keeps the validated tiled default and adds an explicit `vectorized`
option for FP16/BF16 head dimension 128. Other supported shapes use tiled attention.
It packs adjacent K/V elements into aligned loads; accumulation and partition merge
remain FP32. It requires four warps, as does the original tiled implementation.

On the same A100 80GB PCIe / PyTorch 2.8.0+cu128 session, batch32/context2048:

- Original tiled, split256: 0.490062 ms, 53.3% of flash throughput.
- Vectorized, split256: 0.417692 ms, 62.2% of flash throughput.
- This is a 1.17x speedup over the same-session original kernel.

These are medians of five 100-call samples, not the longer nine-by-500 confirmation
used for earlier headline measurements. The 75% A100 target is still unmet.
No full-checkpoint generation throughput or parity is claimed for the new option.
The earlier default-path checkpoint results remain in `a100-8b-2026-10-02`.

## Validation and rejected candidates

`tests.xml` records 115 passing GPU tests for the initial vectorized candidate.
`grouped-tests.xml` records 133 passing tests, including six extra vectorized
non-power-of-two page/tail cases and twelve tests of a rejected shared-KV candidate.
`reduction-tests.xml` records 136 passing tests for another experimental revision.
The current source omits the rejected shared-KV and modified-reduction paths.
The saved final-source manifest identifies the cleaned implementation, not every
intermediate experiment. CPU-only validation is 25 passed / 96 CUDA skips.

The shared-KV candidate was slower (best saved result: 54.1% of flash). Thread-count
and reduction experiments also failed to meet 75%. The saved JSON files include
all successfully retrieved measurements, including slower ones. A subsequent SSH
transfer stalled; some later records were not retrieved before stopping the pod.
No successful tensor-core build or measurement was obtained, and that candidate is
not shipped. The session exceeded its intended one-hour cap; the pod was then stopped
and the console verified compute and storage at $0/hour.

## Reproduction

```sh
python -m pytest -q
python -m benchmarks.attention --algorithm tiled --warps 4 --split-size 256 --repeats 5 --iterations 100 --output results/tiled.json
python -m benchmarks.attention --algorithm vectorized --warps 4 --split-size 256 --repeats 5 --iterations 100 --output results/vectorized.json
# Full-model candidate validation is still needed; these commands are provided to run it.
python -m benchmarks.generation checkpoints/Meta-Llama-3-8B --budget-gib 38 --attention-algorithm vectorized
python -m benchmarks.generation checkpoints/Meta-Llama-3-8B --budget-gib 38 --attention-algorithm vectorized --dense-attention cuda
```

## Capacity constraint

The unchanged 38 GiB workload needs 176 sequences to achieve 4x the measured dense
capacity of 44. The pinned model's BF16 weights plus the necessary paged BF16 KV
storage for those sequences occupy **40.395 GiB before scratch space**. Four times
capacity cannot fit that budget by reducing allocator overhead alone. Quantization,
lossless compression, or genuine prefix sharing would be separate changes requiring
new accuracy and workload validation. Neither shorter prompts nor a slower/larger
baseline was substituted to manufacture the ratio. A100 40GB hardware remains untested.
