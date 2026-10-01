import torch
from . import BlockPool
from .attention import paged_attention

class PagedCache:
    def __init__(self, layers, blocks, block_size, kv_heads, head_dim, max_context,
                 *, device="cpu", dtype=torch.float32, backend="reference"):
        self.pool = BlockPool(blocks, block_size, max_context)
        self.block_size = block_size
        self.backend = backend
        self.k = torch.empty(layers, blocks, block_size, kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self._ids = set()

    def create(self, seq):
        self.pool.create(seq)
        self._ids.add(seq)

    def fork(self, source, target):
        self.pool.fork(source, target)
        self._ids.add(target)

    def release(self, seq):
        self.pool.release(seq)
        self._ids.remove(seq)

    def begin(self, ids):
        reservation = self.pool.append(ids)
        for source, target in reservation.copies:
            self.k[:, target].copy_(self.k[:, source])
            self.v[:, target].copy_(self.v[:, source])
        device = self.k.device
        slots = torch.tensor(reservation.slots, device=device, dtype=torch.long)
        lengths = torch.tensor([self.pool.length(i) for i in ids], device=device, dtype=torch.int32)
        tables = [self.pool.table(i) for i in ids]
        width = max(map(len, tables))
        tables = torch.tensor([t + [-1]*(width-len(t)) for t in tables], device=device, dtype=torch.int32)
        return slots, tables, lengths

    def attend(self, layer, q, k, v, state):
        slots, tables, lengths = state
        self.k[layer].flatten(0, 1)[slots] = k
        self.v[layer].flatten(0, 1)[slots] = v
        return paged_attention(q.contiguous(), self.k[layer], self.v[layer], tables, lengths,
                               backend=self.backend)

    @property
    def allocated_bytes(self):
        return 2*self.k.numel()*self.k.element_size()

class ContiguousCache:
    """Independent dense pre-allocation baseline, one slot per sequence."""
    def __init__(self, layers, batch, max_context, kv_heads, head_dim,
                 *, device="cpu", dtype=torch.float32):
        self.k = torch.empty(layers, batch, max_context, kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.max_context = max_context
        self.lengths = {}
        self.rows = {}
        self.free = list(reversed(range(batch)))

    def create(self, seq):
        if seq in self.rows:
            raise ValueError("sequence already exists")
        if not self.free:
            raise RuntimeError("dense cache full")
        self.rows[seq] = self.free.pop()
        self.lengths[seq] = 0

    def release(self, seq):
        self.free.append(self.rows.pop(seq))
        del self.lengths[seq]

    def begin(self, ids):
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sequence")
        for i in ids:
            if self.lengths[i] >= self.max_context:
                raise ValueError("context limit exceeded")
        state = [(self.rows[i], self.lengths[i]) for i in ids]
        for i in ids:
            self.lengths[i] += 1
        return state

    def attend(self, layer, q, k, v, state):
        import torch.nn.functional as F
        # Batched dense baseline for equal-length requests (the throughput workload).
        if len({pos for _, pos in state}) == 1:
            rows = torch.tensor([row for row, _ in state], device=q.device)
            pos = state[0][1]
            self.k[layer, rows, pos] = k
            self.v[layer, rows, pos] = v
            keys = self.k[layer, rows, :pos+1].transpose(1, 2)
            values = self.v[layer, rows, :pos+1].transpose(1, 2)
            return F.scaled_dot_product_attention(q[:, :, None], keys, values,
                                                  enable_gqa=True).squeeze(2)
        outputs = []
        for i, (row, pos) in enumerate(state):
            self.k[layer, row, pos] = k[i]
            self.v[layer, row, pos] = v[i]
            keys = self.k[layer, row, :pos+1].transpose(0, 1)
            values = self.v[layer, row, :pos+1].transpose(0, 1)
            repeats = q.shape[1] // keys.shape[0]
            outputs.append(F.scaled_dot_product_attention(q[i, :, None],
                keys.repeat_interleave(repeats, 0), values.repeat_interleave(repeats, 0)).squeeze(1))
        return torch.stack(outputs)
