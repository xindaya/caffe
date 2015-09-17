#include <petuum_ps_common/oplog/inc_append_only_buffer.hpp>
#include <string.h>
//#
// 具体实现了inc append buffer
// 这的操作不是计算 w= delta w+w
// 而是将这些数据存储起来，这个容器就是append_only_buffer#
namespace petuum {
//#
// 判断容量是否够
// #
bool IncAppendOnlyBuffer::Inc(int32_t row_id, int32_t col_id, const void *delta) {
  if (size_ + sizeof(int32_t) + sizeof(int32_t) + update_size_
      > capacity_)
    return false;

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = row_id;
  size_ += sizeof(int32_t);

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = col_id;
  size_ += sizeof(int32_t);

  memcpy(buff_.get() + size_, delta, update_size_);
  size_ += update_size_;

  return true;
}

bool IncAppendOnlyBuffer::BatchInc(int32_t row_id, const int32_t *col_ids,
                                   const void *deltas, int32_t num_updates) {
  if (size_ + (sizeof(int32_t) + sizeof(int32_t) + update_size_)*num_updates
      > capacity_)
    return false;
//#
// 原来这个buffer的结构决定了，就算是batch的操作，也要做相应的tuple结构来存储
// #
  for (int i = 0; i < num_updates; ++i) {
    *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = row_id;
    size_ += sizeof(int32_t);

    *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = col_ids[i];
    size_ += sizeof(int32_t);

    memcpy(buff_.get() + size_,
           reinterpret_cast<const uint8_t*>(deltas) + update_size_*i, update_size_);
    size_ += update_size_;
  }
  return true;
}

bool IncAppendOnlyBuffer::DenseBatchInc(int32_t row_id, const void *deltas,
                                        int32_t index_st,
                                        int32_t num_updates) {
  if (size_ + (sizeof(int32_t) + sizeof(int32_t) + update_size_)*num_updates
      > capacity_)
    return false;

  for (int i = 0; i < num_updates; ++i) {
    *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = row_id;
    size_ += sizeof(int32_t);

    *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = index_st + i;
    size_ += sizeof(int32_t);

    memcpy(buff_.get() + size_, reinterpret_cast<const uint8_t*>(deltas) + update_size_*i, update_size_);
    size_ += update_size_;
  }
  return true;
}


void IncAppendOnlyBuffer::InitRead() {
  read_ptr_ = buff_.get();
}

const void *IncAppendOnlyBuffer::Next(
    int32_t *row_id, int32_t const **col_ids, int32_t *num_updates) {
  if (read_ptr_ >= buff_.get() + size_)
    return 0;

  *row_id = *(reinterpret_cast<int32_t*>(read_ptr_));
  read_ptr_ += sizeof(int32_t);

  *col_ids = reinterpret_cast<int32_t*>(read_ptr_);
  read_ptr_ += sizeof(int32_t);

  *num_updates = 1;

  void *update_ptr = read_ptr_;

  read_ptr_ += update_size_;

  return update_ptr;
}

}
