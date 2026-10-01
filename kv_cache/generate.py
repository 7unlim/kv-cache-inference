import argparse
import json
from pathlib import Path
import torch
from .model import Llama
from .cache import PagedCache
from .engine import Engine

def main():
    p = argparse.ArgumentParser(description="Greedy generation from a local Llama 3 HF safetensors checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--blocks", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=["cuda","reference"], default="cuda")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Llama.from_checkpoint(args.checkpoint,device=args.device,dtype=dtype)
    c = model.config
    cache = PagedCache(c.num_hidden_layers,args.blocks,args.block_size,c.num_key_value_heads,c.head_dim,
                       args.max_context,device=args.device,dtype=dtype,backend=args.backend)
    raw = json.loads((Path(args.checkpoint)/"config.json").read_text())
    eos = raw.get("eos_token_id",tokenizer.eos_token_id)
    eos = eos if isinstance(eos,list) else ([] if eos is None else [eos])
    prompts = [tokenizer.encode(s,add_special_tokens=True) for s in args.prompt]
    output = Engine(model,cache).generate(prompts,args.max_new_tokens,eos)
    for prompt,tokens in zip(args.prompt,output):
        print(json.dumps({"prompt":prompt,"token_ids":tokens,"completion":tokenizer.decode(tokens)}))

if __name__ == "__main__":
    main()
