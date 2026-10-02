"""Checkpoint generation comparison; end-to-end includes prefill and host scheduling."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from benchmarks.workload import load_model,model_record
from kv_cache.cache import PagedCache,ContiguousCache
from kv_cache.engine import Engine

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",nargs="?")
    p.add_argument("--synthetic-weights",action="store_true",help="random Llama 3 8B-shaped weights; NOT pretrained model validation")
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--dense-attention",choices=["sdpa","cuda"],default="sdpa",help="sdpa is independent; cuda isolates storage layout using the same selected attention")
    p.add_argument("--attention-algorithm",choices=["tiled","vectorized"],default="tiled")
    p.add_argument("--split-size",type=int,default=256)
    p.add_argument("--text-file",help="optional fixed text corpus; deterministic token windows with BOS")
    p.add_argument("--prompt-length",type=int,default=128)
    p.add_argument("--new-tokens",type=int,default=128)
    p.add_argument("--max-context",type=int,default=4096)
    p.add_argument("--blocks",type=int,default=0,help="0 sizes pool to the benchmark workload")
    p.add_argument("--tokenwise-prefill",action="store_true",help="retain original prefill for comparison")
    p.add_argument("--budget-gib",type=float,default=0,help="optional PyTorch allocator budget; not hardware emulation")
    p.add_argument("--iterations",type=int,default=3)
    p.add_argument("--output",default="results/generation.json")
    args=p.parse_args()
    if min(args.batch,args.prompt_length,args.new_tokens,args.max_context,args.iterations)<=0: p.error("dimensions must be positive")
    if args.prompt_length+args.new_tokens-1>args.max_context: p.error("context too small")
    if bool(args.checkpoint)==args.synthetic_weights: p.error("choose checkpoint or --synthetic-weights")
    if not torch.cuda.is_available(): p.error("CUDA GPU required")
    if args.budget_gib:
        total=torch.cuda.get_device_properties(0).total_memory
        if not 0<args.budget_gib*2**30<total: p.error("budget must be smaller than physical VRAM")
        torch.cuda.set_per_process_memory_fraction(args.budget_gib*2**30/total)
    torch.manual_seed(42)
    model=load_model(args.checkpoint,synthetic=args.synthetic_weights)
    c=model.config
    if args.text_file:
        if not args.checkpoint: p.error("text prompts require checkpoint tokenizer")
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
        text=Path(args.text_file).read_text()
        corpus=tokenizer.encode(text,add_special_tokens=False)
        width=args.prompt_length-1
        if width<=0 or len(corpus)<width: p.error("corpus/prompt length too short")
        starts=[i*(len(corpus)-width)//max(1,args.batch-1) for i in range(args.batch)]
        prompts=[[tokenizer.bos_token_id]+corpus[start:start+width] for start in starts]
        prompt_source={"kind":"fixed_text_windows","corpus_sha256":hashlib.sha256(text.encode()).hexdigest()}
    else:
        prompts=torch.randint(0,c.vocab_size,(args.batch,args.prompt_length)).tolist()
        prompt_source={"kind":"uniform_random_token_ids","seed":42}
    prompt_source["token_ids_sha256"]=hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    common=dict(layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=c.head_dim,
                max_context=args.max_context,device="cuda",dtype=torch.bfloat16)
    report={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "model":model_record(model,args.checkpoint),"parameters":vars(args),"prompt_source":prompt_source,
        "scope":"static batch; benchmark prompts; EOS disabled; end-to-end includes prefill",
        "dense_attention_comparison":"independent SDPA auto dispatch" if args.dense_attention=="sdpa" else "same selected CUDA arithmetic; contiguous maximum-context reservation; storage-layout control"}
    attention=dict(attention_algorithm=args.attention_algorithm,attention_warps=4,split_size=args.split_size)
    outputs={}
    for backend in ["paged","contiguous"]:
        if backend=="paged":
            blocks=args.blocks or args.batch*((args.prompt_length+args.new_tokens-1+15)//16)
            cache=PagedCache(**common,blocks=blocks,block_size=16,backend="cuda",**attention)
        else:
            cache=ContiguousCache(**common,batch=args.batch,backend=args.dense_attention,**attention)
        engine=Engine(model,cache)
        engine.generate(prompts,min(2,args.new_tokens),batched_prefill=not args.tokenwise_prefill)  # compile and warm up
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(args.iterations): output=engine.generate(prompts,args.new_tokens,batched_prefill=not args.tokenwise_prefill)
        torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)/args.iterations
        outputs[backend]=output
        report[backend]={"seconds":elapsed,"output_tokens_per_second":args.batch*args.new_tokens/elapsed,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "cache_reserved_bytes":2*cache.k.numel()*cache.k.element_size()}
        del engine,cache
        gc.collect(); torch.cuda.empty_cache()
    report["token_for_token_match"]=outputs["paged"]==outputs["contiguous"]
    mismatches=[{"sequence":i,"position":j,"paged":a,"contiguous":b}
                for i,(pa,co) in enumerate(zip(outputs["paged"],outputs["contiguous"]))
                for j,(a,b) in enumerate(zip(pa,co)) if a!=b]
    report["mismatched_token_count"]=len(mismatches)
    report["first_mismatches"]=mismatches[:10]
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
    if not report["token_for_token_match"]: raise SystemExit("greedy output mismatch")

if __name__=="__main__": main()
