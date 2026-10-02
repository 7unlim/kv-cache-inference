"""GPU-resident synthetic steady-state decode capacity, not end-to-end serving capacity.

Both caches hold the same model and context-length distribution. Each trial
allocates and touches the cache, constructs live metadata, and runs full-model
steps. No inference is made from allocation arithmetic alone.
"""
import argparse
import gc
import json
from pathlib import Path
import torch
from benchmarks.workload import load_model, model_record
from kv_cache.cache import PagedCache, ContiguousCache


def find_limit(probe, maximum):
    """Find largest successful batch assuming capacity is monotone; report censoring."""
    records = {}
    def test(n):
        if n not in records:
            records[n] = probe(n)
        return records[n]["success"]
    low, high = 0, 1
    while test(high):
        low = high
        if high == maximum:
            return low, True, records
        high = min(maximum, high*2)
    while high-low > 1:
        mid = (low+high)//2
        if test(mid): low = mid
        else: high = mid
    return low, False, records


@torch.inference_mode()
def probe(model, backend, batch, lengths, max_context, steps, budget):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    c = model.config
    common = dict(layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
                  head_dim=c.head_dim, max_context=max_context,
                  device="cuda", dtype=torch.bfloat16)
    # Each larger batch extends the same deterministic length sequence.
    live_lengths = [lengths[i % len(lengths)] for i in range(batch)]
    live_tokens = sum(live_lengths)
    blocks = sum((n+steps+15)//16 for n in live_lengths)
    cache = None
    result = {"batch":batch,"success":False,"initial_lengths":live_lengths}
    try:
        if backend == "paged":
            cache = PagedCache(**common, blocks=blocks, block_size=16, backend="cuda")
            for i in range(batch): cache.create(i)
            for pos in range(max(live_lengths)):
                cache.pool.append([i for i,n in enumerate(live_lengths) if n>pos])
        else:
            cache = ContiguousCache(**common, batch=batch)
            for i,n in enumerate(live_lengths):
                cache.create(i); cache.lengths[i] = n
        # Touch physical allocations; content is deliberately synthetic, not a prompt prefill.
        cache.k.zero_(); cache.v.zero_()
        reserved = 2*cache.k.numel()*cache.k.element_size()
        tokens = torch.arange(batch, device="cuda") % c.vocab_size
        ids = list(range(batch))
        for _ in range(steps):
            logits = model.step(tokens, ids, cache)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("non-finite decode logits")
            tokens = logits.argmax(-1)
            del logits
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        token_bytes = 2*c.num_hidden_layers*c.num_key_value_heads*c.head_dim*2
        result.update(success=peak<=budget, peak_allocated_bytes=peak,
                      cache_reserved_bytes=reserved,
                      useful_kv_bytes=(live_tokens+batch*steps)*token_bytes,
                      unused_reserved_kv_bytes=reserved-(live_tokens+batch*steps)*token_bytes)
        if peak>budget: result["failure"]="allocation_budget_exceeded"
    except torch.OutOfMemoryError:
        # Catch only CUDA OOM. Correctness/runtime failures must abort the benchmark.
        result["failure"]="cuda_out_of_memory"
    finally:
        del cache
        gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",nargs="?")
    p.add_argument("--synthetic-weights",action="store_true")
    p.add_argument("--lengths",default="128,257,512,769,1024,1537,2048,3073")
    p.add_argument("--max-context",type=int,default=4096)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--max-batch",type=int,default=256)
    p.add_argument("--budget-gib",type=float,default=0,help="0 uses device VRAM minus 2 GiB; allocator budget, not hardware emulation")
    p.add_argument("--output",default="results/capacity.json")
    args=p.parse_args()
    lengths=[int(n) for n in args.lengths.split(",")]
    if min(args.steps,args.max_batch,args.max_context,*lengths)<=0 or max(lengths)+args.steps>args.max_context:
        p.error("positive dimensions and lengths + steps <= max context required")
    if bool(args.checkpoint)==args.synthetic_weights: p.error("choose checkpoint or --synthetic-weights")
    if not torch.cuda.is_available(): p.error("CUDA GPU required")
    total=torch.cuda.get_device_properties(0).total_memory
    budget=int(args.budget_gib*2**30) if args.budget_gib else total-2*2**30
    if not 0<budget<total: p.error("budget must be positive and smaller than device VRAM")
    torch.cuda.set_per_process_memory_fraction(budget/total)
    torch.manual_seed(42)
    model=load_model(args.checkpoint,synthetic=args.synthetic_weights)
    report={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
            "device_vram_bytes":total,"allocator_budget_bytes":budget,"parameters":vars(args),
            "model":model_record(model,args.checkpoint),
            "scope":"synthetic prepopulated KV; full-model steady-state decode; no prompt prefill; no serving scheduler; allocator budget is not hardware emulation"}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    for backend in ("paged","contiguous"):
        maximum,censored,trials=find_limit(lambda b:probe(model,backend,b,lengths,args.max_context,args.steps,budget),args.max_batch)
        report[backend]={"max_successful_batch":maximum,"search_censored":censored,
                         "trials":[trials[n] for n in sorted(trials)]}
        path.write_text(json.dumps(report,indent=2)+"\n")
    a,b=report["paged"],report["contiguous"]
    report["capacity_ratio"]=a["max_successful_batch"]/b["max_successful_batch"] if b["max_successful_batch"] else None
    common_batch=min(a["max_successful_batch"],b["max_successful_batch"])
    if common_batch:
        matched={backend:probe(model,backend,common_batch,lengths,args.max_context,args.steps,budget)
                 for backend in ("paged","contiguous")}
        report["matched_batch_memory"]=matched
        if all(r["success"] for r in matched.values()):
            dense_waste=matched["contiguous"]["unused_reserved_kv_bytes"]
            report["wasted_kv_reduction_fraction"]=(1-matched["paged"]["unused_reserved_kv_bytes"]/dense_waste
                                                       if dense_waste else None)
    report["ratio_is_lower_bound"]=a["search_censored"] and not b["search_censored"]
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("paged","contiguous")},indent=2))
    print("Maximum successful batches:",a["max_successful_batch"],b["max_successful_batch"])

if __name__=="__main__": main()
