#include <torch/extension.h>
torch::Tensor decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor tables, torch::Tensor lengths, int64_t split);
torch::Tensor decode(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                     torch::Tensor tables, torch::Tensor lengths, int64_t split) {
  TORCH_CHECK(q.is_cuda(), "queries must be CUDA tensors");
  for (const auto& t : {q,k,v,tables,lengths}) {
    TORCH_CHECK(t.device()==q.device() && t.is_contiguous(), "same-device contiguous tensors required");
  }
  TORCH_CHECK(q.dim()==3 && k.dim()==4 && v.sizes()==k.sizes(), "invalid Q/K/V shape");
  TORCH_CHECK(tables.dim()==2 && lengths.dim()==1 && tables.size(0)==q.size(0)
      && lengths.size(0)==q.size(0), "invalid metadata shape");
  TORCH_CHECK(q.size(0)>0 && q.size(1)>0 && k.size(0)>0 && k.size(1)>0
      && k.size(2)>0 && tables.size(1)>0, "empty dimensions unsupported");
  TORCH_CHECK(q.size(2)==k.size(3) && q.size(2)<=256 && q.size(2)%32==0,
      "head dimension must be a multiple of 32, at most 256");
  TORCH_CHECK(q.size(1)%k.size(2)==0, "query heads must divide into KV groups");
  TORCH_CHECK(q.scalar_type()==k.scalar_type() && k.scalar_type()==v.scalar_type(), "dtype mismatch");
  TORCH_CHECK(q.scalar_type()==torch::kFloat16 || q.scalar_type()==torch::kBFloat16
      || q.scalar_type()==torch::kFloat32, "unsupported dtype");
  TORCH_CHECK(tables.scalar_type()==torch::kInt32 && lengths.scalar_type()==torch::kInt32,
      "metadata must be int32");
  TORCH_CHECK(split>0 && split<=65536, "invalid split size");
  // Metadata values are trusted allocator output; shape checks do not synchronize the GPU.
  return decode_cuda(q,k,v,tables,lengths,split);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("decode", &decode); }
