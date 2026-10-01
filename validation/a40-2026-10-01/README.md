# A40 validation, 2026-10-01

Source commit: 6010b597bd0bdb428bd654f4e010abc421660360.
NVIDIA A40; PyTorch 2.8.0+cu128; CUDA 12.8; Python 3.12.

Commands (activated virtual environment with ninja installed):

```sh
python -m pytest -q -x --junitxml=results/tests.xml
python -m benchmarks.attention --batch 32 --context 2048 --output results/attention.json
```

29 tests passed. The JSON contains the initial attention-only timing;
it is not a full Llama 3 8B generation benchmark. No checkpoint was downloaded.
