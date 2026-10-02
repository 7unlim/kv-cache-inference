# A100 80GB real-checkpoint validation — 2026-10-02 UTC

Model: meta-llama/Meta-Llama-3-8B, revision
`8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`, BF16, 8,030,261,248 parameters.
Hardware: NVIDIA A100 80GB PCIe; PyTorch 2.8.0+cu128 / CUDA 12.8.
The 38 GiB allocation cap is not an A100 40GB hardware substitute.

## Recorded iterations

- `generation-initial`: tokenwise prefill, before RoPE reuse; 530 paged tokens/s.
- `generation-prefill`: batched prefill; 982 paged tokens/s.
- `generation-rope`: also reuse per-pass RoPE factors; 1,377 paged tokens/s.
- `generation-final1/2`: repeated independent SDPA comparisons; 1,389 and 1,364 paged tokens/s, 409 mismatches each.
- `generation-text`: independent SDPA text comparison; 1,411 paged tokens/s, 1,451 mismatches.
- `generation-controlled-random/text`: same tiled kernel for both storage layouts;
  1,378 / 1,405 paged tokens/s, zero mismatches in 4,096 output tokens each.
- `attention`: 54.97% of forced flash SDPA throughput, nine 500-call repeats.
- `capacity`: 154 paged versus 44 contiguous, 3.5x, uncensored failure boundaries;
  99.6156% less unused reserved cache at matched batch 44.
- `capacity-generation`: actual prefill and five generated tokens pass at both capacities.
- `parity`: identical prefill followed by 32 teacher-forced steps, seven argmax
  mismatches among 1,024 decisions. Worst absolute logit difference is 2.4375.
- `checkpoint-oracle`: two real-model prefill prompts compared with Hugging Face;
  both argmaxes agree, maximum absolute logit error 0.146484375. Not bitwise equivalence.
- `controlled-tests`: final source suite, 103 passed, none skipped.

JSON files preserve parameters, hardware and measured results. Generation uses batch32,
128 prompt tokens, 128 output tokens, three iterations except initial (one), EOS disabled.
Wall time includes prefill and scheduling, excluding loading and warmup.
The fixed text corpus is versioned under benchmarks/data. These are benchmark workloads,
not production traffic. The final source manifest does not identify the earlier interim
prefill/initial implementations; those records retain their historical stage labels.

No claim of independent SDPA exact parity, 4x capacity, A100 75% attention throughput,
or actual A100 40GB performance is established by this session.
