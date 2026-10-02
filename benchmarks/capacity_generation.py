"""Validate capacity candidates by actually prefilling and generating, not seeding KV."""
import argparse,gc,json,time,hashlib
from pathlib import Path
import torch
from benchmarks.workload import load_model
from kv_cache.cache import PagedCache,ContiguousCache
from kv_cache.engine import Engine

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('capacity_report')
    p.add_argument('--output',default='results/capacity-generation.json');args=p.parse_args()
    source=json.loads(Path(args.capacity_report).read_text());budget=source['allocator_budget_bytes']
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    model=load_model(args.checkpoint);c=model.config
    config=source['parameters'];lengths=[int(n) for n in config['lengths'].split(',')];steps=config['steps']
    report={'gpu':torch.cuda.get_device_name(),'allocator_budget_bytes':budget,
            'scope':'full prefill and greedy generation at steady-state capacity candidates; random token prompts; EOS disabled; these candidates are not a new exhaustive capacity search'}
    common=dict(layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=c.head_dim,
                max_context=config['max_context'],device='cuda',dtype=torch.bfloat16)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for backend in ('paged','contiguous'):
        gc.collect();torch.cuda.empty_cache();torch.manual_seed(42)
        batch=source[backend]['max_successful_batch'];live=[lengths[i%len(lengths)] for i in range(batch)]
        prompts=[torch.randint(0,c.vocab_size,(n,)).tolist() for n in live]
        cache=engine=None
        result={'batch':batch,'success':False,'prompt_lengths':live}
        try:
            if backend=='paged':
                cache=PagedCache(**common,blocks=sum((n+steps+15)//16 for n in live),block_size=16,backend='cuda')
            else:cache=ContiguousCache(**common,batch=batch)
            engine=Engine(model,cache)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            output=engine.generate(prompts,steps+1)
            torch.cuda.synchronize()
            result.update(success=all(len(row)==steps+1 for row in output),seconds=time.perf_counter()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          output_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest())
        except torch.OutOfMemoryError:
            result['failure']='cuda_out_of_memory'
        finally:
            del engine,cache
            gc.collect();torch.cuda.empty_cache()
        report[backend]=result;path.write_text(json.dumps(report,indent=2)+'\n')
        print(backend,result['batch'],result['success'],flush=True)

if __name__=='__main__':main()
