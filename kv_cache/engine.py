"""Single-owner static-batch greedy generation, with ragged prompts and EOS release."""
import torch

class Engine:
    def __init__(self, model, cache):
        self.model, self.cache = model, cache
        self.next_id = 0

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens=32, eos_token_ids=(), *, share_prefix=True, batched_prefill=True):
        if max_new_tokens < 0 or not prompts or any(not p for p in prompts):
            raise ValueError("nonempty prompts and nonnegative generation length required")
        if any(len(p)+max(0,max_new_tokens-1) > self.model.config.max_position_embeddings for p in prompts):
            raise ValueError("request exceeds model context")
        if max_new_tokens == 0:
            return [[] for _ in prompts]
        device = next(self.model.parameters()).device
        ids = list(range(self.next_id,self.next_id+len(prompts)))
        self.next_id += len(prompts)
        live = set()
        logits = {}
        outputs = [[] for _ in prompts]
        try:
            shared = 0
            if share_prefix and hasattr(self.cache, "fork") and len(prompts) > 1:
                for pos in range(min(map(len, prompts))):
                    if any(p[pos] != prompts[0][pos] for p in prompts[1:]):
                        break
                    shared += 1
            # A shared partial page is copied on divergence, so it saves no KV
            # storage once equal-length prompts branch. Prefer batched prefill.
            if batched_prefill and len({len(p) for p in prompts})==1 and shared<getattr(self.cache,"block_size",1):
                shared=0
            self.cache.create(ids[0]); live.add(ids[0])
            for pos in range(shared):
                result = self.model.step(torch.tensor([prompts[0][pos]], device=device), [ids[0]], self.cache)
            if shared:
                for i in range(len(prompts)): logits[i] = result[0]
            for seq in ids[1:]:
                if shared: self.cache.fork(ids[0], seq)
                else: self.cache.create(seq)
                live.add(seq)
            if batched_prefill and shared==0 and len({len(p) for p in prompts})==1:
                result=self.model.prefill(torch.tensor(prompts,device=device),ids,self.cache)
                for i in range(len(prompts)): logits[i]=result[i]
            else:
                # Token-at-a-time prefill intentionally reuses the same tested paged path.
                for pos in range(shared, max(map(len,prompts))):
                    active = [i for i,p in enumerate(prompts) if pos < len(p)]
                    batch = torch.tensor([prompts[i][pos] for i in active], device=device)
                    result = self.model.step(batch,[ids[i] for i in active],self.cache)
                    for row,i in enumerate(active):
                        logits[i] = result[row]
            active = list(range(len(prompts)))
            for step in range(max_new_tokens):
                predicted = torch.stack([logits[i] for i in active]).argmax(-1).tolist()
                keep, tokens = [], []
                for i, token in zip(active, predicted):
                    outputs[i].append(token)
                    if token in eos_token_ids or step == max_new_tokens-1:
                        self.cache.release(ids[i]); live.remove(ids[i])
                    else:
                        keep.append(i); tokens.append(token)
                if not keep:
                    break
                result = self.model.step(torch.tensor(tokens,device=device),[ids[i] for i in keep],self.cache)
                for row,i in enumerate(keep):
                    logits[i] = result[row]
                active = keep
            return outputs
        finally:
            for seq in live:
                self.cache.release(seq)
