#pragma once
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace paged {
struct Sequence { int length = 0; std::vector<int> blocks; };
struct Append { std::vector<int> slots; std::vector<std::pair<int,int>> copies; };
// Host-side metadata. Caller serializes operations and completes block copies
// before writing returned slots. No device pointers are owned here.
class BlockPool {
 public:
  BlockPool(int count, int block_size, int max_context)
      : size_(block_size), max_(max_context) {
    if (count <= 0 || size_ <= 0 || max_ <= 0)
      throw std::invalid_argument("pool dimensions must be positive");
    refs_.resize(count);
    for (int i = count - 1; i >= 0; --i) free_.push_back(i);
  }
  void create(int id) {
    if (!seqs_.emplace(id, Sequence{}).second)
      throw std::invalid_argument("sequence already exists");
  }
  void fork(int source, int target) {
    const auto copy = seqs_.at(source);
    if (!seqs_.emplace(target, copy).second)
      throw std::invalid_argument("target already exists");
    for (int block : copy.blocks) ++refs_[block];
  }
  void release(int id) {
    const auto &s = seqs_.at(id);
    for (int block : s.blocks) if (--refs_[block] == 0) free_.push_back(block);
    seqs_.erase(id);
  }
  // Preflight the entire batch: OOM/context errors leave all sequences intact.
  Append append(const std::vector<int>& ids) {
    std::unordered_set<int> seen;
    int needed = 0;
    std::unordered_map<int, int> remaining_refs;
    for (int id : ids) {
      if (!seen.insert(id).second) throw std::invalid_argument("duplicate sequence");
      const auto &s = seqs_.at(id);
      if (s.length == max_) throw std::length_error("context limit exceeded");
      if (s.length % size_ == 0) ++needed;
      else {
        int block = s.blocks.back();
        auto entry = remaining_refs.emplace(block, refs_[block]);
        if (entry.first->second > 1) { ++needed; --entry.first->second; }
      }
    }
    if (needed > static_cast<int>(free_.size()))
      throw std::runtime_error("KV block pool exhausted");
    Append out;
    out.slots.reserve(ids.size()); out.copies.reserve(ids.size());
    // Reserve host metadata before changing any references.
    for (int id : ids) {
      auto &s = seqs_.at(id);
      if (s.length % size_ == 0) s.blocks.reserve(s.blocks.size()+1);
    }
    for (int id : ids) {
      auto &s = seqs_.at(id);
      const int offset = s.length % size_;
      if (offset == 0) s.blocks.push_back(allocate());
      else if (refs_[s.blocks.back()] > 1) {
        const int old = s.blocks.back(), fresh = allocate();
        --refs_[old]; s.blocks.back() = fresh;
        out.copies.emplace_back(old, fresh);
      }
      out.slots.push_back(s.blocks.back()*size_ + offset);
      ++s.length;
    }
    return out;
  }
  const Sequence& sequence(int id) const { return seqs_.at(id); }
  int free_blocks() const { return static_cast<int>(free_.size()); }
  int refcount(int block) const { return refs_.at(block); }
 private:
  int allocate() {
    const int block = free_.back(); free_.pop_back(); refs_[block] = 1; return block;
  }
  int size_, max_;
  std::vector<int> free_, refs_;
  std::unordered_map<int, Sequence> seqs_;
};
}
