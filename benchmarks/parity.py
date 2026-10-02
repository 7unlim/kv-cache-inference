"""Teacher-forced, same-token comparison; avoids conflating rollout divergence with error."""
import argparse,json
from pathlib import Path
import torch
from benchmarks.workload import load_model,model_record
from kv_cache.cache import PagedCache,ContiguousCache

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint')
    p.add_argument('--batch',type=int,default=32);p.add_argument('--prompt-length',type=int,default=128)
    p.add_argument('--steps',type=int,default=32);p.add_argument('--output',default='results/parity.json')
    args=p.parse_args();torch.manual_seed(42)
    model=load_model(args.checkpoint);c=model.config
    common=dict(layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=c.head_dim,
                max_context=args.prompt_length+args.steps,device='cuda',dtype=torch.bfloat16)
    paged=PagedCache(**common,blocks=args.batch*((common['max_context']+15)//16),block_size=16,backend='cuda')
    dense=ContiguousCache(**common,batch=args.batch)
    ids=list(range(args.batch))
    for cache in (paged,dense):
        for i in ids:cache.create(i)
    tokens=torch.randint(0,c.vocab_size,(args.batch,args.prompt_length),device='cuda')
    a=model.prefill(tokens,ids,paged);b=model.prefill(tokens,ids,dense)
    report={'model':model_record(model,args.checkpoint),'gpu':torch.cuda.get_device_name(),
            'scope':'identical teacher-forced tokens from dense greedy path; BF16; not independent model correctness oracle',
            'prefill_max_abs_logit_error':(a-b).abs().max().item(),'steps':[]}
    for step in range(args.steps):
        chosen=b.argmax(-1)
        a=model.step(chosen,ids,paged);b=model.step(chosen,ids,dense)
        top=b.float().topk(2,dim=-1)
        mismatch=a.argmax(-1)!=b.argmax(-1)
        report['steps'].append({'step':step,'max_abs_logit_error':(a-b).abs().max().item(),
            'mean_abs_logit_error':(a-b).abs().float().mean().item(),
            'argmax_mismatches':int(mismatch.sum().item()),
            'dense_top_two_margins_at_mismatches':(top.values[:,0]-top.values[:,1])[mismatch].tolist()})
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
