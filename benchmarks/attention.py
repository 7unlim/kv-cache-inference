"""python -m benchmarks.attention --output results/attention.json"""
import argparse
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from kv_cache.attention import paged_attention

def timed(fn,iterations):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end)/iterations

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--context",type=int,default=2048)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--split-size",type=int,default=256)
    p.add_argument("--warps",type=int,choices=[1,4,8],default=1)
    p.add_argument("--algorithm",choices=["online","tiled","vectorized"],default="online")
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--output",default="results/attention.json")
    args=p.parse_args()
    if min(args.batch,args.context,args.iterations,args.split_size,args.repeats)<=0: p.error("arguments must be positive")
    if not torch.cuda.is_available(): p.error("CUDA GPU required")
    torch.manual_seed(42)
    b,n=args.batch,args.context
    h,hkv,d,s=32,8,128,16
    width=(n+s-1)//s
    k=torch.randn(b*width,s,hkv,d,device="cuda",dtype=torch.bfloat16)
    v=torch.randn_like(k)
    tables=torch.randperm(b*width,device="cuda").view(b,width).to(torch.int32)
    lengths=torch.full((b,),n,device="cuda",dtype=torch.int32)
    q=torch.randn(b,h,d,device="cuda",dtype=torch.bfloat16)
    dense_k=k[tables.long()].flatten(1,2)[:,:n].transpose(1,2).contiguous()
    dense_v=v[tables.long()].flatten(1,2)[:,:n].transpose(1,2).contiguous()
    def custom():
        return paged_attention(q,k,v,tables,lengths,backend="cuda",split_size=args.split_size,warps=args.warps,algorithm=args.algorithm)
    def flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q[:,:,None],dense_k,dense_v,enable_gqa=True).squeeze(2)
    actual,expected=custom(),flash()
    torch.testing.assert_close(actual,expected,atol=0.03,rtol=0.03)
    paged_samples,flash_samples=[],[]
    for repeat in range(args.repeats):
        # Alternate timing order to reduce a consistent warmup/clock-order bias.
        order=[(custom,paged_samples),(flash,flash_samples)]
        if repeat%2: order.reverse()
        for fn,samples in order: samples.append(timed(fn,args.iterations))
    paged_ms,flash_ms=statistics.median(paged_samples),statistics.median(flash_samples)
    report={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "batch":b,"context":n,"heads":h,"kv_heads":hkv,"head_dim":d,"dtype":"bfloat16",
        "block_size":s,"split_size":args.split_size,"iterations":args.iterations,
        "algorithm":args.algorithm,"warps":args.warps,"repeats":args.repeats,
        "paged_samples_ms":paged_samples,"flash_samples_ms":flash_samples,
        "paged_ms":paged_ms,"flash_sdpa_ms":flash_ms,"throughput_ratio":flash_ms/paged_ms,
        "paged_queries_per_second":b*1000/paged_ms,"max_abs_error":(actual-expected).abs().max().item(),
        "scope":"attention only; includes custom scratch allocation; excludes page construction and dense gathering"}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
