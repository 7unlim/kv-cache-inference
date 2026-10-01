#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float warp_sum(float x) {
  for (int offset=16; offset; offset/=2) x += __shfl_down_sync(0xffffffff,x,offset);
  return __shfl_sync(0xffffffff,x,0);
}
// Several warps process disjoint tokens of one context partition. Each warp
// computes a stable online softmax; a shared-memory merge combines their states.
// WARPS=1 retains the original implementation and remains the default baseline.
template<typename T, int WARPS>
__global__ void partial(const T* q,const T* k,const T* v,const int* tables,
                        const int* lengths,float* out,float* maxima,float* sums,
                        int H,int HK,int D,int S,int width,int splits,int split) {
  int b=blockIdx.x,h=blockIdx.y,p=blockIdx.z;
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  int idx=(b*H+h)*splits+p;
  float acc[8]={0}, query[8];
  for(int j=0;j<D/32;++j) query[j]=float(q[(b*H+h)*D+j*32+lane]);
  float m=-INFINITY, sum=0;
  int end=min(lengths[b],(p+1)*split);
  for(int t=p*split+warp;t<end;t+=WARPS) {
    int physical=tables[b*width+t/S];
    int64_t base=((int64_t(physical)*S+t%S)*HK+h/(H/HK))*D;
    float dot=0;
    for(int j=0;j<D/32;++j) dot+=query[j]*float(k[base+j*32+lane]);
    dot=warp_sum(dot)*rsqrtf(float(D));
    float next=fmaxf(m,dot), alpha=expf(m-next), beta=expf(dot-next);
    for(int j=0;j<D/32;++j) acc[j]=acc[j]*alpha+beta*float(v[base+j*32+lane]);
    sum=sum*alpha+beta; m=next;
  }
  if constexpr (WARPS > 1) {
    __shared__ float warp_max[WARPS], warp_sum_exp[WARPS];
    __shared__ float warp_values[WARPS][256];
    if(lane==0) { warp_max[warp]=m; warp_sum_exp[warp]=sum; }
    for(int j=0;j<D/32;++j) warp_values[warp][j*32+lane]=acc[j];
    __syncthreads();
    if(warp!=0) return;
    m=-INFINITY; sum=0;
    for(int w=0;w<WARPS;++w) m=fmaxf(m,warp_max[w]);
    for(int j=0;j<D/32;++j) acc[j]=0;
    for(int w=0;w<WARPS;++w) {
      // Empty warps/partitions must not evaluate -infinity - -infinity.
      if(warp_sum_exp[w]==0) continue;
      float weight=expf(warp_max[w]-m);
      sum+=warp_sum_exp[w]*weight;
      for(int j=0;j<D/32;++j) acc[j]+=warp_values[w][j*32+lane]*weight;
    }
  }
  for(int j=0;j<D/32;++j) out[int64_t(idx)*D+j*32+lane]=acc[j];
  if(lane==0) { maxima[idx]=m; sums[idx]=sum; }
}
template<typename T>
__global__ void merge(const float* partials,const float* maxima,const float* sums,
                      T* out,int H,int D,int splits) {
  int row=blockIdx.x,lane=threadIdx.x;
  float m=-INFINITY;
  for(int p=0;p<splits;++p) m=fmaxf(m,maxima[row*splits+p]);
  float total=0,acc[8]={0};
  for(int p=0;p<splits;++p) {
    int idx=row*splits+p;
    if(sums[idx]==0) continue;
    float weight=expf(maxima[idx]-m);
    total+=sums[idx]*weight;
    for(int j=0;j<D/32;++j) acc[j]+=partials[int64_t(idx)*D+j*32+lane]*weight;
  }
  for(int j=0;j<D/32;++j) out[int64_t(row)*D+j*32+lane]=T(acc[j]/total);
}
torch::Tensor decode_cuda(torch::Tensor q,torch::Tensor k,torch::Tensor v,
                          torch::Tensor tables,torch::Tensor lengths,int64_t split,int64_t warps) {
  c10::cuda::CUDAGuard guard(q.device());
  int B=q.size(0),H=q.size(1),D=q.size(2),S=k.size(1),width=tables.size(1);
  int splits=(width*S+split-1)/split;
  auto opts=q.options().dtype(torch::kFloat32);
  auto parts=torch::empty({B,H,splits,D},opts);
  auto maxima=torch::empty({B,H,splits},opts), sums=torch::empty_like(maxima);
  auto result=torch::empty_like(q);
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,q.scalar_type(),"paged_decode",[&] {
    if (warps==1) {
    partial<scalar_t,1><<<dim3(B,H,splits),32,0,stream>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),tables.data_ptr<int>(),lengths.data_ptr<int>(),parts.data_ptr<float>(),
      maxima.data_ptr<float>(),sums.data_ptr<float>(),H,k.size(2),D,S,width,splits,split);
    }
    else if (warps==4) {
    partial<scalar_t,4><<<dim3(B,H,splits),128,0,stream>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),tables.data_ptr<int>(),lengths.data_ptr<int>(),parts.data_ptr<float>(),
      maxima.data_ptr<float>(),sums.data_ptr<float>(),H,k.size(2),D,S,width,splits,split);
    }
    else if (warps==8) {
    partial<scalar_t,8><<<dim3(B,H,splits),256,0,stream>>>(q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),tables.data_ptr<int>(),lengths.data_ptr<int>(),parts.data_ptr<float>(),
      maxima.data_ptr<float>(),sums.data_ptr<float>(),H,k.size(2),D,S,width,splits,split);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge<scalar_t><<<B*H,32,0,stream>>>(parts.data_ptr<float>(),maxima.data_ptr<float>(),
      sums.data_ptr<float>(),result.data_ptr<scalar_t>(),H,D,splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return result;
}
