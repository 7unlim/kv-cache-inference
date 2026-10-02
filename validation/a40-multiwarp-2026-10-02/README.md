# A40 multi-warp iteration, 2026-10-02 UTC

Measured source commit: `a85b4082046897fece18b3f392244d7425158eeb`.
Run date: October 2 UTC / October 1, 2026 Pacific.
NVIDIA A40; PyTorch 2.8.0+cu128; CUDA 12.8; Python 3.12.
A replacement A40 host was used because the previous allocation was unavailable.

## Correctness

All 53 tests passed in 49.81 seconds, with zero failures or skips.
CUDA attention tests cover 1/4/8 warps, FP32/FP16/BF16, head dimensions
32/64/128/256, ragged contexts, permuted physical pages, and empty partitions.
Small-model greedy parity tests use the default one-warp path.

## Measured performance

BF16, batch 32, context 2048, 32 query heads, 8 KV heads, head dimension 128,
page size 16. Each result is the median of five samples of 100 calls, with
10 warmup calls per sample. Custom/flash timing order alternates by repeat.
The benchmark checks output agreement before timing. Maximum absolute error
against forced flash SDPA was 0.0009765625 for all seven configurations.

- One warp, split 256: 2.208274 ms; flash 0.490813 ms; 22.226% of flash throughput.
- Four warps, split 256: 1.312800 ms; flash 0.503781 ms; 38.375% of flash throughput.
- Eight warps, split 256: 1.746500 ms; flash 0.512799 ms; 29.362% of flash throughput.
- Four warps, split 64: 1.393953 ms; 36.082% of flash throughput.
- Four warps, split 128: 1.330885 ms; 37.846% of flash throughput.
- Four warps, split 512: 1.339760 ms; 37.487% of flash throughput.
- Four warps, split 1024: 1.404214 ms; 35.636% of flash throughput.

Four warps / split 256 is 1.682x the same-session one-warp throughput,
with 40.55% lower latency. It is the best median among configurations tested,
not a statistically established global optimum. Separate candidate processes
ran sequentially; flash timings varied somewhat between processes. Individual
samples and each corresponding flash baseline are retained in the JSON files.

The previous session's 18.0% result is preserved separately; comparing it directly
with 38.4% would mix host/session effects with code changes. Use this session's
22.2% one-warp baseline for the iteration comparison.

These are attention-only measurements including custom scratch allocation but
excluding cache construction and dense gathering. They do not establish Llama
3 8B tokens/s, A100 performance, or maximum concurrent sequences. No checkpoint
was downloaded. The default model path remains one warp; four warps is opt-in.

## Reproduce

Activate a virtual environment using the host CUDA PyTorch installation and
install `.[test,model,cuda]` (including ninja), then run from the repository root:

```sh
python -m pytest -q -x --junitxml=results/tests.xml
for w in 1 4 8; do
  python -m benchmarks.attention --warps "$w" --repeats 5 --output "results/warps$w.json"
done
for s in 64 128 512 1024; do
  python -m benchmarks.attention --warps 4 --split-size "$s" --repeats 5 --output "results/warps4-split$s.json"
done
```

The pod was stopped after all result files were copied locally.
