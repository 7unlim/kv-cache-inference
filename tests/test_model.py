import json
import torch
import pytest
from kv_cache.model import Config,Llama
from kv_cache.cache import PagedCache,ContiguousCache
from kv_cache.engine import Engine

@pytest.fixture
def model():
    torch.manual_seed(5)
    return Llama(Config(vocab_size=71,hidden_size=64,intermediate_size=96,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=64)).eval()

def caches(model):
    c = model.config
    common = dict(layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=c.head_dim)
    return (PagedCache(**common,blocks=32,block_size=4,max_context=64),
            ContiguousCache(**common,batch=3,max_context=64))

def test_step_logits_and_greedy_tokens(model):
    paged,dense = caches(model)
    prompts = [[1,8,4,3,5],[2,9],[3,1,2,7,3,2,8]]
    assert Engine(model,paged).generate(prompts,12) == Engine(model,dense).generate(prompts,12)
    assert paged.pool.free_blocks == 32
    for cache in [paged,dense]:
        cache.create(100); cache.create(101)
    for _ in range(15):
        tokens = torch.randint(0,71,(2,))
        a = model.step(tokens,[100,101],paged)
        b = model.step(tokens,[100,101],dense)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

def test_cow_values_and_model_output(model):
    paged,dense = caches(model)
    paged.create(0); dense.create(0); dense.create(1)
    for token in [1,2,3]:
        model.step(torch.tensor([token]),[0],paged)
        model.step(torch.tensor([token,token]),[0,1],dense)
    paged.fork(0,1)
    old = paged.pool.table(0)[0]
    prefix = paged.k[:,old,:3].clone()
    a = model.step(torch.tensor([4,5]),[0,1],paged)
    b = model.step(torch.tensor([4,5]),[0,1],dense)
    torch.testing.assert_close(a,b)
    for seq in [0,1]:
        torch.testing.assert_close(paged.k[:,paged.pool.table(seq)[0],:3],prefix)
    assert paged.pool.table(0) != paged.pool.table(1)

def test_oom_cleanup_and_eos(model):
    c=model.config
    cache=PagedCache(c.num_hidden_layers,1,4,c.num_key_value_heads,c.head_dim,64)
    engine=Engine(model,cache)
    with pytest.raises(RuntimeError): engine.generate([[1],[2]],2)
    assert cache.pool.free_blocks == 1
    # Any token ends the request, so no decode forward or second block needed.
    assert len(engine.generate([[1,2,3,4]],20,range(c.vocab_size))[0]) == 1
    assert cache.pool.free_blocks == 1
    assert engine.generate([[1]],0) == [[]]

def test_checkpoint_roundtrip(model,tmp_path):
    from safetensors.torch import save_file
    from dataclasses import asdict
    (tmp_path/"config.json").write_text(json.dumps({"model_type":"llama",**asdict(model.config)}))
    save_file(model.state_dict(),tmp_path/"model.safetensors")
    loaded=Llama.from_checkpoint(tmp_path,device="cpu",dtype=torch.float32)
    a,_=caches(model); b,_=caches(loaded)
    assert Engine(model,a).generate([[1,3]],5)==Engine(loaded,b).generate([[1,3]],5)

def test_huggingface_full_forward_oracle(model):
    # Independent implementation verifies RoPE, normalization, GQA and causal positions.
    from transformers import LlamaConfig,LlamaForCausalLM
    from dataclasses import asdict
    hf=LlamaForCausalLM(LlamaConfig(**asdict(model.config))).eval()
    hf.load_state_dict(model.state_dict(),strict=True)
    paged,_=caches(model)
    ids=[0,1]
    for seq in ids: paged.create(seq)
    tokens=torch.tensor([[1,4,7,9,2,8],[3,1,6,4,2,9]])
    with torch.inference_mode():
        expected=hf(tokens).logits
        for pos in range(tokens.shape[1]):
            actual=model.step(tokens[:,pos],ids,paged)
            torch.testing.assert_close(actual,expected[:,pos],rtol=2e-5,atol=2e-6)

def test_engine_shares_prefix_with_constrained_pool(model):
    c=model.config
    # A full common page and one private tail each needs only three pages.
    paged=PagedCache(c.num_hidden_layers,3,4,c.num_key_value_heads,c.head_dim,64)
    _,dense=caches(model)
    prompts=[[1,2,3,4,5],[1,2,3,4,6]]
    assert Engine(model,paged).generate(prompts,3)==Engine(model,dense).generate(prompts,3)
    assert paged.pool.free_blocks==3
    with pytest.raises(RuntimeError):
        Engine(model,paged).generate(prompts,3,share_prefix=False)
    assert paged.pool.free_blocks==3

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA device required")
def test_cuda_model_greedy_matches_dense():
    torch.manual_seed(5)
    model=Llama(Config(vocab_size=71,hidden_size=128,intermediate_size=192,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128)).cuda().to(torch.bfloat16).eval()
    common=dict(layers=2,kv_heads=2,head_dim=32,max_context=128,device="cuda",dtype=torch.bfloat16)
    paged=PagedCache(**common,blocks=24,block_size=16,backend="cuda")
    dense=ContiguousCache(**common,batch=3)
    prompts=[[1,2,3,4,5],[1,2,3,9],[1,2,3,4,5,7]]
    assert Engine(model,paged).generate(prompts,24)==Engine(model,dense).generate(prompts,24)
