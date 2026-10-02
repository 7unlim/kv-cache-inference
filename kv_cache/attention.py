from pathlib import Path
from functools import lru_cache
import torch
import torch.nn.functional as F

@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / "csrc"
    return load(name="paged_decode_cuda", sources=[str(root / "decode.cpp"),
                str(root / "decode.cu")], extra_cuda_cflags=["-O3"], verbose=False)

def paged_attention(q, k, v, tables, lengths, *, backend="reference", split_size=256, warps=1, algorithm="online"):
    """q [B,H,D]; pages [P,S,Hkv,D]; one query at each sequence's last token."""
    if algorithm not in ("online", "tiled", "vectorized"):
        raise ValueError("algorithm must be online, tiled, or vectorized")
    if backend == "cuda":
        if not q.is_cuda:
            raise ValueError("CUDA backend requires CUDA tensors")
        return extension().decode(q, k, v, tables, lengths, split_size, warps, {"online":0,"tiled":1,"vectorized":2}[algorithm])
    if backend != "reference":
        raise ValueError(f"unknown attention backend: {backend}")
    outputs = []
    for i, length in enumerate(lengths.tolist()):
        if length <= 0:
            raise ValueError("attention requires nonempty sequences")
        blocks = tables[i, : (length + k.shape[1]-1)//k.shape[1]].long()
        keys = k[blocks].flatten(0, 1)[:length].transpose(0, 1)
        values = v[blocks].flatten(0, 1)[:length].transpose(0, 1)
        repeats = q.shape[1] // k.shape[2]
        keys = keys.repeat_interleave(repeats, dim=0)
        values = values.repeat_interleave(repeats, dim=0)
        outputs.append(F.scaled_dot_product_attention(q[i, :, None, :], keys, values).squeeze(1))
    return torch.stack(outputs)
