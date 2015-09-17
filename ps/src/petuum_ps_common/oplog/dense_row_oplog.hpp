// Author: jinliang
#pragma once

#include <stdint.h>

#include <functional>
#include <boost/noncopyable.hpp>
#include <memory>
#include <string.h>

#include <petuum_ps_common/oplog/abstract_row_oplog.hpp>
#include <glog/logging.h>

namespace petuum {
class DenseRowOpLog : public virtual AbstractRowOpLog {
public:
  //#传入了两个函数作为操作
  // InitUpdate
  // CheckZeroUpdate
  // #
  //-----------
  //#row_size_难道表示的是一行之内多少列？
  // 如果是这样，那么问题解决了，都好理解了
  // 但是我们有必要这么用吗？
  //
  // ---
  // update_size表示的是一个row_id,col_id 组成的tuple定义的一个cell的使用字节的数目
  //
  // #
  DenseRowOpLog(InitUpdateFunc InitUpdate,
                CheckZeroUpdateFunc CheckZeroUpdate,
                size_t update_size,
                size_t row_size):
      AbstractRowOpLog(update_size),
      row_size_(row_size),
      oplogs_(new uint8_t[update_size*row_size]),
      InitUpdate_(InitUpdate),
      CheckZeroUpdate_(CheckZeroUpdate) {
    CHECK(row_size > 0);
    Reset();
  }

  virtual ~DenseRowOpLog() { }

//#
// .get() 获取的是指针地址
// 用cpp编程果然是一个难度很大的工作
// 不过适应之后就没有什么难度了吧
// #
// #
  //
  void Reset() {
    memset(oplogs_.get(), 0, update_size_*row_size_);
  }

//#
// 不理解作者的意图
// col_id vs row_size_
//
// #
  void* Find(int32_t col_id) {
    return oplogs_.get() + col_id*update_size_;
  }

  const void* FindConst(int32_t col_id) const {
    return oplogs_.get() + col_id*update_size_;
  }

  void* FindCreate(int32_t col_id) {
    return Find(col_id);
  }

  // Guaranteed ordered traversal
  void* BeginIterate(int32_t *column_id) {
    iter_col_id_ = 0;
    return Next(column_id);
  }

// #reinterpret_cast?
// 算是显式的类型转换吧#
  void* Next(int32_t *column_id) {
    uint8_t *update = reinterpret_cast<uint8_t*>(Find(iter_col_id_));
    // #如果是零的话，要处理
    //
    // #
    while (CheckZeroUpdate_(update) && iter_col_id_ < row_size_) {
      update += update_size_;
      ++iter_col_id_;
    }
    if (iter_col_id_ >= row_size_)
      return 0;
    *column_id = iter_col_id_;
    ++iter_col_id_;
    return update;
  }

  // Guaranteed ordered traversal, in ascending order of column_id
  const void* BeginIterateConst(int32_t *column_id) const {
    iter_col_id_ = 0;
    return NextConst(column_id);
  }

  const void* NextConst(int32_t *column_id) const {
    const uint8_t *update = reinterpret_cast<const uint8_t*>(
        FindConst(iter_col_id_));
    while (CheckZeroUpdate_(update) && iter_col_id_ < row_size_) {
      update += update_size_;
      ++iter_col_id_;
    }
    if (iter_col_id_ >= row_size_)
      return 0;
    *column_id = iter_col_id_;
    ++iter_col_id_;
    return update;
  }

  size_t GetSize() const {
    return row_size_;
  }

  size_t ClearZerosAndGetNoneZeroSize() {
    size_t num_nonzeros = 0;
    int32_t col_id = 0;
    for (col_id = 0; col_id < row_size_; ++col_id) {
      uint8_t *update = oplogs_.get() + update_size_*col_id;
      if (!CheckZeroUpdate_(update))
        ++num_nonzeros;
    }
    num_nonzeros_ = num_nonzeros;
    return num_nonzeros;
  }

  size_t GetSparseSerializedSize() {
    return sizeof(int32_t) + sizeof(int32_t)*num_nonzeros_
        + update_size_*num_nonzeros_;
  }

  size_t GetDenseSerializedSize() {
    return row_size_*update_size_;
  }

// #
// 算是压缩存储，将原来不用的数据删除掉了
// #
  size_t SerializeSparse(void *mem) {
    int32_t col_id = 0;
    int32_t *mem_num_updates = reinterpret_cast<int32_t*>(mem);
    *mem_num_updates = num_nonzeros_;

    uint8_t *mem_index = reinterpret_cast<uint8_t*>(mem) + sizeof(int32_t);
    uint8_t *mem_oplogs = mem_index + num_nonzeros_*sizeof(int32_t);

    for (col_id = 0; col_id < row_size_; ++col_id) {
      if (CheckZeroUpdate_(oplogs_.get() + col_id*update_size_))
        continue;

      *(reinterpret_cast<int32_t*>(mem_index)) = col_id;
      memcpy(mem_oplogs, oplogs_.get() + col_id*update_size_, update_size_);
      mem_index += sizeof(int32_t);
      mem_oplogs += update_size_;
    }

    return GetSparseSerializedSize();
  }

  size_t SerializeDense(void *mem) {
    memcpy(mem, oplogs_.get(), AbstractRowOpLog::update_size_*row_size_);
    return GetDenseSerializedSize();
  }

  const void *ParseDenseSerializedOpLog(const void *mem,
                                        int32_t *num_updates,
                                        size_t *serialized_size) const {
    *num_updates = row_size_;
    *serialized_size = row_size_*AbstractRowOpLog::update_size_;
    return mem;
  }

  void OverwriteWithDenseUpdate(const void *updates, int32_t index_st,
                                int32_t num_updates) {
    size_t offset = index_st*AbstractRowOpLog::update_size_;
    uint8_t *updates_dest = oplogs_.get() + offset;
    memcpy(updates_dest, updates, num_updates*AbstractRowOpLog::update_size_);
  }

protected:
  const size_t row_size_; // capacity
  size_t num_nonzeros_; //非零的数据行数

  //----
  std::unique_ptr<uint8_t[]> oplogs_;
  //为什么用这个智能指针呢#
  // #uint8这个是什么单位？
  // 表示用8bit来表示int？
  // 好像是这样干的#
  //---
  const InitUpdateFunc InitUpdate_;
  const CheckZeroUpdateFunc CheckZeroUpdate_;
  mutable int32_t iter_col_id_; // mutable是cpp的一个新的关键字？
};
}
