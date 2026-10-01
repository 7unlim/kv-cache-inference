"""Inference-only Llama 3 architecture with explicit weights and cache ownership."""
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if min(self.vocab_size, self.hidden_size, self.intermediate_size,
               self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads,
               self.max_position_embeddings) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("invalid attention head configuration")
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head dimension")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y*torch.rsqrt(y.square().mean(-1, keepdim=True)+self.eps)).to(x.dtype)*self.weight

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=False)

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def rope(self, x, positions):
        # HF Llama uses split-half rotation, not adjacent even/odd pairs.
        d = self.config.head_dim
        freq = 1.0/(self.config.rope_theta**(torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d))
        angle = positions.float()[:, None]*freq[None, :]
        angle = torch.cat((angle, angle), -1)[:, None, :]
        a, b = x.chunk(2, dim=-1)
        return x*angle.cos().to(x.dtype)+torch.cat((-b,a), -1)*angle.sin().to(x.dtype)

    @torch.inference_mode()
    def step(self, tokens, ids, cache):
        if tokens.ndim != 1 or tokens.shape[0] != len(ids) or not ids:
            raise ValueError("one input token per nonempty sequence required")
        c = self.config
        lengths = [cache.pool.length(i) if hasattr(cache, "pool") else cache.lengths[i] for i in ids]
        if max(lengths) >= c.max_position_embeddings:
            raise ValueError("model context limit exceeded")
        positions = torch.tensor(lengths, device=tokens.device)
        x = self.model.embed_tokens(tokens)
        state = cache.begin(ids)
        # If device execution fails after begin(), discard these sequences.
        for index, layer in enumerate(self.model.layers):
            norm = layer.input_layernorm(x)
            attn = layer.self_attn
            q = self.rope(attn.q_proj(norm).view(-1,c.num_attention_heads,c.head_dim), positions)
            k = self.rope(attn.k_proj(norm).view(-1,c.num_key_value_heads,c.head_dim), positions)
            v = attn.v_proj(norm).view(-1,c.num_key_value_heads,c.head_dim)
            x = x+attn.o_proj(cache.attend(index,q,k,v,state).flatten(1))
            x = x+layer.mlp(layer.post_attention_layernorm(x))
        return self.lm_head(self.model.norm(x))

    @classmethod
    def from_checkpoint(cls, directory, *, device="cuda", dtype=torch.bfloat16):
        from safetensors import safe_open
        root = Path(directory)
        raw = json.loads((root/"config.json").read_text())
        if raw.get("model_type") != "llama" or raw.get("rope_scaling"):
            raise ValueError("only standard Llama with unscaled RoPE is supported (Llama 3, not 3.1)")
        if raw.get("attention_bias") or raw.get("mlp_bias") or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("unsupported bias or activation configuration")
        c = Config(**{k:v for k,v in raw.items() if k in Config.__dataclass_fields__})
        if raw.get("head_dim", c.head_dim) != c.head_dim:
            raise ValueError("custom head dimension unsupported")
        with torch.device("meta"):
            model = cls(c)
        model.to_empty(device=device)
        model.to(dtype=dtype)
        expected = model.state_dict()
        loaded = set()
        index = root/"model.safetensors.index.json"
        paths = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        with torch.no_grad():
            for filename in paths:
                with safe_open(root/filename, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        if name not in expected:
                            if name.endswith("rotary_emb.inv_freq"):
                                continue
                            raise ValueError(f"unexpected checkpoint tensor: {name}")
                        tensor = f.get_tensor(name)
                        if tensor.shape != expected[name].shape or name in loaded:
                            raise ValueError(f"invalid/duplicate checkpoint tensor: {name}")
                        expected[name].copy_(tensor)
                        loaded.add(name)
        missing = set(expected)-loaded
        if c.tie_word_embeddings and "model.embed_tokens.weight" in loaded:
            model.lm_head.weight = model.model.embed_tokens.weight
            missing.discard("lm_head.weight")
        if missing:
            raise ValueError(f"missing checkpoint weights: {sorted(missing)}")
        return model.eval()
