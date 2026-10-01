"""Deterministic allocator accounting, not an empirical GPU capacity benchmark."""
import argparse
import json
from pathlib import Path
from kv_cache import BlockPool

def measure(lengths,max_context=4096,block_size=16,layers=32,kv_heads=8,head_dim=128):
    if not lengths or min(lengths)<=0 or max(lengths)>max_context:
        raise ValueError("lengths must be in [1,max_context]")
    blocks=sum((n+block_size-1)//block_size for n in lengths)
    pool=BlockPool(blocks,block_size,max_context)
    for i in range(len(lengths)): pool.create(i)
    for pos in range(max(lengths)):
        pool.append([i for i,n in enumerate(lengths) if pos<n])
    token_bytes=2*layers*kv_heads*head_dim*2
    dense=len(lengths)*max_context*token_bytes
    paged=(blocks-pool.free_blocks)*block_size*token_bytes
    useful=sum(lengths)*token_bytes
    dense_waste=dense-useful; paged_waste=paged-useful
    return {"lengths":lengths,"max_context":max_context,"block_size":block_size,
        "bytes_per_token":token_bytes,"dense_reserved_bytes":dense,"paged_live_block_bytes":paged,
        "dense_waste_bytes":dense_waste,"paged_waste_bytes":paged_waste,
        "waste_reduction_fraction":1-paged_waste/dense_waste if dense_waste else None,
        "live_allocation_ratio":dense/paged,
        "scope":"allocator accounting; excludes weights, scratch, metadata, and unused physical pool capacity; not measured GPU concurrency"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",default="128,257,512,769,1024,1537,2048,3073")
    p.add_argument("--max-context",type=int,default=4096)
    p.add_argument("--block-size",type=int,default=16)
    p.add_argument("--output",default="results/memory.json")
    args=p.parse_args()
    report=measure([int(x) for x in args.lengths.split(",")],args.max_context,args.block_size)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
