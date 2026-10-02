# Tiled attention validation — October 2, 2026 UTC

Source: implementation in this commit, based on `061fc90`. `source-sha256.json`
identifies the exact 20 implementation, benchmark, build, and test files copied
to the GPU. All hashes were verified remotely before shutdown.
Hardware: NVIDIA A40. PyTorch 2.8.0+cu128, CUDA 12.8, Python 3.12.

## Correctness and integration

93 tests passed in 57.75 seconds; none skipped. See `tests.log` and `tests.xml`.
Coverage includes online 1/4/8 warps and tiled 4 warps; FP32/FP16/BF16;
all head dimensions divisible by 32 from 32 to 256; noncontiguous pages;
page sizes 3, 8, 16; empty partitions; partial tails; extreme softmax scores;
and small-model greedy token parity against an independent contiguous cache.
`PagedCache` selects the tiled kernel by default; the low-level attention API
retains the online default so the old kernel is still easy to reproduce.

## Performance

Canonical shape: batch 32, context 2048, BF16, 32 query heads, 8 KV heads,
head dimension 128, page size 16, partition size 256, four warps.
Three independent processes each measured nine samples of 500 calls, with ten
warmup calls before each sample and alternating custom/flash timing order.

- `final1.json`: tiled 0.602609 ms, flash 0.506667 ms, ratio 84.079%.
- `final2.json`: tiled 0.605430 ms, flash 0.505581 ms, ratio 83.508%.
- `final3.json`: tiled 0.608589 ms, flash 0.506890 ms, ratio 83.289%.
- `online4.json`: previous four-warp online kernel, 1.337917 ms, ratio 37.665%.

The median confirmation run is 2.210x faster than the same-host online baseline,
with 54.75% lower latency. The canonical confirmations have maximum absolute
error 0.0009765625 against forced flash SDPA. The 75% attention target is met
for this A40 workload; this does not validate A100 performance.

`shape-*.json` retain all 12 batch/context combinations (batches 1, 8, 32;
contexts 128, 512, 2048, 4096), each five samples of 100 calls. Ratios vary by
shape and timing duration. In particular, batch 8 / context 2048 is 74.4%,
and the shorter batch 32 / context 2048 run is 91.2%. The more conservative
83.3–84.1% longer-run range is the headline, not the best short-run number.
Individual samples and corresponding flash timings are retained in every JSON.

Attention timing includes custom scratch allocation and excludes page-table
construction and dense gathering. It is not generation tokens/s. There was no
8B checkpoint execution, no A100 test, and no measured concurrency sweep.

## Reproduce

With CUDA PyTorch and nvcc installed, activate a virtual environment and install
`.[test,model,cuda]`, including ninja. Run from the repository root:

```sh
python -m pytest -q -x --junitxml=results/tests.xml
for run in 1 2 3; do
  python -m benchmarks.attention --algorithm tiled --warps 4 --repeats 9 --iterations 500 --output "results/final$run.json"
done
python -m benchmarks.attention --warps 4 --repeats 9 --iterations 500 --output results/online4.json
for b in 1 8 32; do
  for n in 128 512 2048 4096; do
    python -m benchmarks.attention --algorithm tiled --warps 4 --batch "$b" --context "$n" --output "results/shape-b$b-n$n.json"
  done
done
```

Results were copied locally before stopping the pod. Early tuning trials used
a different pod and are not the final evidence: preliminary short runs reached
76.4%, but longer checks of that earlier kernel fell to 71.9–72.3%. Subsequent
16-lane grouping and power-of-two page lookup produced the final implementation
verified here on a replacement A40. The old pod stopped before its raw exploratory
files were retrieved, so only the new complete final run is checked in here.
