import pytest
import torch
from benchmarks.workload import load_model, model_record
from benchmarks.capacity import find_limit
from kv_cache.model import Config


def test_synthetic_model_is_finite_and_explicitly_labeled():
    config=Config(vocab_size=31,hidden_size=64,intermediate_size=96,
                  num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    model=load_model(synthetic=True,device="cpu",dtype=torch.float32,config=config)
    report=model_record(model)
    assert report["weights"]=="synthetic_random_not_pretrained"
    assert not report["checkpoint_loaded"]
    assert report["parameter_count"]==sum(p.numel() for p in model.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert torch.equal(model.model.norm.weight,torch.ones(64))
    with pytest.raises(ValueError): load_model(synthetic=False)
    with pytest.raises(ValueError): load_model("checkpoint",synthetic=True)


@pytest.mark.parametrize("limit",[0,1,7,31,32,100])
def test_capacity_search_records_failed_boundary_or_censoring(limit):
    maximum,censored,trials=find_limit(lambda b:{"success":b<=limit},32)
    assert maximum==min(limit,32)
    assert censored==(limit>=32)
    if not censored:
        assert not trials[maximum+1]["success"]
    if maximum: assert trials[maximum]["success"]
