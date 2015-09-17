#pragma once

#include <memory>
#include <stdint.h>
#include <boost/noncopyable.hpp>
#include <glog/logging.h>

namespace petuum {

// Fixed size sparse vector

    //固定大小的稀疏向量
    // 单例对象
class SparseVector : boost::noncopyable {
public:
  SparseVector(size_t capacity, size_t value_size):
      capacity_(capacity),
      value_size_(value_size),
      size_(0),
      data_(new uint8_t[(value_size + sizeof(int32_t))*capacity]) { }

  ~SparseVector() { }

  size_t get_capacity() const;
  size_t get_size() const;

  uint8_t* GetValPtr(int32_t key);
  const uint8_t* GetValPtrConst(int32_t key) const;

  uint8_t* GetByIdx(int32_t index, int32_t *key);
  const uint8_t* GetByIdxConst(int32_t index, int32_t *key) const;

  void Copy(const SparseVector &src);

  void Reset() { size_ = 0; }

  void Compact(int32_t index);

  uint8_t* get_data_ptr() { return data_.get(); }


  // 核心的数据结构都在这里
private:
  // 得到key的值
  int32_t &GetKeyByIdx(int32_t index);

  // 得到value的指针
  uint8_t *GetValPtrByIdx(int32_t index);

  // 得到指针
  uint8_t *GetPtrByIdx(int32_t index);

  // 加入了const限制
  const int32_t &GetKeyByIdxConst(int32_t index) const;
  const uint8_t *GetValPtrByIdxConst(int32_t index) const;

  bool Search(int32_t key, int32_t st, int32_t end, int32_t *found_idx) const;
  // assume 1) size_ < capacity_
  // 2) index < size_
  void InsertAtIndex(int32_t index, int32_t key);

  const size_t capacity_;
  const size_t value_size_;
  size_t size_;
  // 数据是不能转移所有权的内存数据结构
  std::unique_ptr<uint8_t[]> data_;
};

}
