"""Checkpoint generation comparison; end-to-end includes prefill and host scheduling."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from kv_cache.model import Llama
from kv_cache.cache import PagedCache,ContiguousCache
from kv_cache.engine import Engine

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--prompt-length",type=int,default=128)
    p.add_argument("--new-tokens",type=int,default=128)
    p.add_argument("--max-context",type=int,default=4096)
    p.add_argument("--blocks",type=int,default=0,help="0 sizes pool to the benchmark workload")
    p.add_argument("--iterations",type=int,default=3)
    p.add_argument("--output",default="results/generation.json")
    args=p.parse_args()
    if min(args.batch,args.prompt_length,args.new_tokens,args.max_context,args.iterations)<=0: p.error("dimensions must be positive")
    if args.prompt_length+args.new_tokens-1>args.max_context: p.error("context too small")
    if not torch.cuda.is_available(): p.error("CUDA GPU required")
    torch.manual_seed(42)
    model=Llama.from_checkpoint(args.checkpoint)
    c=model.config
    prompts=torch.randint(0,c.vocab_size,(args.batch,args.prompt_length)).tolist()
    common=dict(layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=c.head_dim,
                max_context=args.max_context,device="cuda",dtype=torch.bfloat16)
    report={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "checkpoint":str(Path(args.checkpoint).resolve()),"parameters":vars(args),
        "scope":"static batch; synthetic token prompts; EOS disabled; end-to-end includes tokenwise prefill; dense baseline uses SDPA auto dispatch"}
    outputs={}
    for backend in ["paged","contiguous"]:
        if backend=="paged":
            blocks=args.blocks or args.batch*((args.prompt_length+args.new_tokens-1+15)//16)
            cache=PagedCache(**common,blocks=blocks,block_size=16,backend="cuda")
        else:
            cache=ContiguousCache(**common,batch=args.batch)
        engine=Engine(model,cache)
        engine.generate(prompts,2)  # compile and warm up
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(args.iterations): output=engine.generate(prompts,args.new_tokens)
        torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)/args.iterations
        outputs[backend]=output
        report[backend]={"seconds":elapsed,"output_tokens_per_second":args.batch*args.new_tokens/elapsed,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "cache_reserved_bytes":2*cache.k.numel()*cache.k.element_size()}
        del engine,cache
        gc.collect(); torch.cuda.empty_cache()
    report["token_for_token_match"]=outputs["paged"]==outputs["contiguous"]
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
    if not report["token_for_token_match"]: raise SystemExit("greedy output mismatch")

if __name__=="__main__": main()
