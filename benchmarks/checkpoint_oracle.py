"""Check real-checkpoint prefill logits against Hugging Face's independent Llama."""
import argparse,json
from pathlib import Path
import torch
from transformers import LlamaForCausalLM
from benchmarks.workload import load_model,model_record
from kv_cache.cache import PagedCache

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',default='results/checkpoint-oracle.json')
    args=p.parse_args();torch.manual_seed(42)
    model=load_model(args.checkpoint);c=model.config
    cache=PagedCache(c.num_hidden_layers,16,16,c.num_key_value_heads,c.head_dim,128,
                     device='cuda',dtype=torch.bfloat16,backend='cuda')
    cache.create(0);cache.create(1)
    tokens=torch.randint(0,c.vocab_size,(2,128),device='cuda')
    actual=model.prefill(tokens,[0,1],cache).float()
    hf=LlamaForCausalLM.from_pretrained(args.checkpoint,torch_dtype=torch.bfloat16,
                                     attn_implementation='sdpa').cuda().eval()
    expected=hf(tokens).logits[:,-1].float()
    report={'model':model_record(model,args.checkpoint),'gpu':torch.cuda.get_device_name(),
            'scope':'two random-token prompts, length 128, pretrained BF16 prefill logits versus Hugging Face SDPA',
            'max_abs_logit_error':(actual-expected).abs().max().item(),
            'mean_abs_logit_error':(actual-expected).abs().mean().item(),
            'argmax_match':actual.argmax(-1).eq(expected.argmax(-1)).tolist()}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
