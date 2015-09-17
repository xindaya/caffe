#include <petuum_ps_common/oplog/batch_inc_append_only_buffer.hpp>
#include <string.h>

namespace petuum {
//#
// batch_inc_append_only_buffer 注释
// 这个文件主要是为了批量batch 更新设计的append only buffer
// #

bool BatchIncAppendOnlyBuffer::Inc(int32_t row_id, int32_t col_id, const void *delta) {
  if (size_ + sizeof(int32_t) + sizeof(int32_t) + update_size_
      > capacity_)
    return false;

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = row_id;
  size_ += sizeof(int32_t);

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = 1;
  size_ += sizeof(int32_t);

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = col_id;
  size_ += sizeof(int32_t);

  memcpy(buff_.get() + size_, delta, update_size_);
  size_ += update_size_;

  return true;
}

bool BatchIncAppendOnlyBuffer::BatchInc(int32_t row_id, const int32_t *col_ids,
                                        const void *deltas, int32_t num_updates) {
  //#首先做判断，如果超出了容量限制，那么数据就写不成功，如果caller使用的是无限重试机制的话，那么在逻辑上
  // 阻塞写入线程，直到写入成功为止
  // #
  if (size_ + sizeof(int32_t) + sizeof(int32_t) +
      (sizeof(int32_t) + update_size_)*num_updates > capacity_)
    return false;
//#写入row_id#
  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = row_id;
  size_ += sizeof(int32_t);
//#写入更新的数据规模#
  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = num_updates;
  size_ += sizeof(int32_t);
//#需要更新的列的id全部#
  memcpy(buff_.get() + size_, col_ids, sizeof(int32_t)*num_updates);
  size_ += sizeof(int32_t)*num_updates;
//#将更新的delta写入buffer中#
  memcpy(buff_.get() + size_, deltas, num_updates);
  size_ += update_size_*num_updates;

  return true;
}

//#
// 这个没有给定具体的列的id列表，而是从一个偏移量开始，更新多个数据
//
// 从具体的实现上来看，也是讲这个过程转化为传统的上面的工作
//
// #

bool BatchIncAppendOnlyBuffer::DenseBatchInc(int32_t row_id, const void *deltas,
                                             int32_t index_st,
                                             int32_t num_updates) {
  if (size_ + sizeof(int32_t) + sizeof(int32_t) +
      (sizeof(int32_t) + update_size_)*num_updates > capacity_)
    return false;

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = row_id;
  size_ += sizeof(int32_t);

  *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = num_updates;
  size_ += sizeof(int32_t);
//#记录id 的list#
  for (int32_t i = 0; i < num_updates; ++i) {
    *(reinterpret_cast<int32_t*>(buff_.get() + size_)) = index_st + i;
    size_ += sizeof(int32_t);
  }
  //拷贝更新参数
  memcpy(buff_.get() + size_, deltas, num_updates);
  size_ += update_size_*num_updates;

  return true;
}


void BatchIncAppendOnlyBuffer::InitRead() {
  read_ptr_ = buff_.get();
}

//#返回的是一个void类型的指针#
const void *BatchIncAppendOnlyBuffer::Next(
    int32_t *row_id, int32_t const **col_ids, int32_t *num_updates) {
  if (read_ptr_ >= buff_.get() + size_)
    return 0;
    // #解析row_id，然后指针后移一位#
  *row_id = *(reinterpret_cast<int32_t*>(read_ptr_));
  read_ptr_ += sizeof(int32_t);
//#解析更新的delta数目，指针后移一位#
  *num_updates = *(reinterpret_cast<int32_t*>(read_ptr_));
  read_ptr_ += sizeof(int32_t);
//#解析col_ids，指针后移若干位#
  *col_ids = reinterpret_cast<int32_t*>(read_ptr_);
  read_ptr_ += sizeof(int32_t)*(*num_updates);

  void *update_ptr = read_ptr_;

  read_ptr_ += update_size_*(*num_updates);

  return update_ptr;
}

}
