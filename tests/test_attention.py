import pytest
import torch
import torch.nn.functional as F
from kv_cache.attention import paged_attention

@pytest.mark.parametrize("warps,algorithm",[(1,"online"),(4,"online"),(8,"online"),(4,"tiled")])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("head_dim",[32,64,128,256])
def test_cuda_ragged_permuted_pages(dtype,head_dim,warps,algorithm):
    if not torch.cuda.is_available(): pytest.skip("CUDA device required")
    torch.manual_seed(17)
    lengths=torch.tensor([1,17,257,513],device="cuda",dtype=torch.int32)
    pages=4*33
    k=torch.randn(pages,16,2,head_dim,device="cuda",dtype=dtype)
    v=torch.randn_like(k)
    tables=torch.randperm(pages,device="cuda").reshape(4,33).to(torch.int32)
    q=torch.randn(4,8,head_dim,device="cuda",dtype=dtype)
    expected=paged_attention(q,k,v,tables,lengths)
    actual=paged_attention(q,k,v,tables,lengths,backend="cuda",split_size=128,warps=warps,algorithm=algorithm)
    tol={torch.float32:2e-5,torch.float16:3e-3,torch.bfloat16:3e-2}[dtype]
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)

def test_reference_matches_explicit_softmax():
    torch.manual_seed(11)
    q=torch.randn(2,4,32)
    k=torch.randn(6,4,2,32); v=torch.randn_like(k)
    tables=torch.tensor([[4,1,3],[5,2,0]],dtype=torch.int32)
    lengths=torch.tensor([5,11],dtype=torch.int32)
    expected=[]
    for row,n in enumerate(lengths):
        keys=k[tables[row].long()].flatten(0,1)[:n].transpose(0,1).repeat_interleave(2,0)
        values=v[tables[row].long()].flatten(0,1)[:n].transpose(0,1).repeat_interleave(2,0)
        weights=torch.softmax((q[row,:,None]*keys).sum(-1)/(32**0.5),dim=-1)
        expected.append((weights[...,None]*values).sum(1))
    torch.testing.assert_close(paged_attention(q,k,v,tables,lengths),torch.stack(expected))

@pytest.mark.parametrize("head_dim",[96,160,192,224])
@pytest.mark.parametrize("split_size",[1,17,1024])
@pytest.mark.parametrize("page_size",[3,8])
def test_tiled_tail_dimensions_and_partitions(head_dim,split_size,page_size):
    if not torch.cuda.is_available(): pytest.skip("CUDA device required")
    torch.manual_seed(21)
    q=torch.randn(2,4,head_dim,device="cuda")
    k=torch.randn(10,page_size,2,head_dim,device="cuda"); v=torch.randn_like(k)
    # Padding is deliberately invalid: empty partitions must not read it.
    tables=torch.tensor([[7,-1,-1,-1,-1],[9,1,8,2,6]],device="cuda",dtype=torch.int32)
    lengths=torch.tensor([1,4*page_size+1],device="cuda",dtype=torch.int32)
    expected=paged_attention(q,k,v,tables,lengths)
    actual=paged_attention(q,k,v,tables,lengths,backend="cuda",algorithm="tiled",
                           warps=4,split_size=split_size)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)

@pytest.mark.parametrize("scale",[0.0,50.0])
def test_tiled_softmax_extreme_scores(scale):
    if not torch.cuda.is_available(): pytest.skip("CUDA device required")
    torch.manual_seed(72)
    q=torch.randn(2,4,128,device="cuda")*scale
    k=torch.randn(66,16,2,128,device="cuda")*scale
    v=torch.randn_like(k)
    tables=torch.randperm(66,device="cuda").reshape(2,33).to(torch.int32)
    lengths=torch.tensor([1,521],device="cuda",dtype=torch.int32)
    expected=paged_attention(q,k,v,tables,lengths)
    actual=paged_attention(q,k,v,tables,lengths,backend="cuda",algorithm="tiled",warps=4)
    torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
