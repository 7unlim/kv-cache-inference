"""Shared model loading for measured checkpoints and explicitly synthetic experiments."""
from dataclasses import asdict
from pathlib import Path
import torch
from kv_cache.model import Config, Llama


def load_model(checkpoint=None, *, synthetic=False, device="cuda", dtype=torch.bfloat16,
               config=None):
    if bool(checkpoint) == synthetic:
        raise ValueError("choose exactly one of checkpoint or synthetic weights")
    if checkpoint:
        return Llama.from_checkpoint(checkpoint, device=device, dtype=dtype)
    # Allocate directly on the target device; avoid a second 16 GB CPU copy.
    with torch.device("meta"):
        model = Llama(config or Config())
    model.to(dtype=dtype)
    model.to_empty(device=device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.02)
    return model.eval()


def model_record(model, checkpoint=None):
    return {
        "weights": "checkpoint" if checkpoint else "synthetic_random_not_pretrained",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "config": asdict(model.config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
        "checkpoint_loaded": bool(checkpoint),
        "source_revision": ((Path(checkpoint)/"source-revision.txt").read_text().strip()
                            if checkpoint and (Path(checkpoint)/"source-revision.txt").exists() else None),
    }
