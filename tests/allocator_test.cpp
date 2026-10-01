#include "block_pool.h"
#include <iostream>
#include <random>
#include <map>
void require(bool value) { if (!value) throw std::runtime_error("invariant failed"); }
int main() {
  paged::BlockPool pool(128, 8, 128);
  std::map<int,int> lengths;
  std::mt19937 rng(19);
  int next=0;
  for(int step=0;step<10000;++step) {
    unsigned op=rng()%4;
    if(lengths.empty() || op==0) { pool.create(next); lengths[next++]=0; }
    else {
      auto it=lengths.begin(); std::advance(it,rng()%lengths.size());
      int id=it->first;
      if(op==1) { pool.fork(id,next); lengths[next++]=it->second; }
      else if(op==2) { pool.release(id); lengths.erase(id); }
      else {
        try { pool.append({id}); ++it->second; }
        catch(const std::runtime_error&) { require(pool.sequence(id).length==it->second); }
        catch(const std::length_error&) { require(pool.sequence(id).length==it->second); }
      }
    }
    std::vector<int> refs(128);
    for(auto [id,length]: lengths) {
      auto& seq=pool.sequence(id);
      require(seq.length==length);
      require(seq.blocks.size()==static_cast<size_t>((length+7)/8));
      for(int block:seq.blocks) ++refs.at(block);
    }
    int free=0;
    for(int block=0;block<128;++block) {
      require(refs[block]==pool.refcount(block));
      free+=refs[block]==0;
    }
    require(free==pool.free_blocks());
  }
  for(auto [id,length]:lengths) pool.release(id);
  require(pool.free_blocks()==128);
  std::cout << "10000 randomized allocator operations passed\n";
}
