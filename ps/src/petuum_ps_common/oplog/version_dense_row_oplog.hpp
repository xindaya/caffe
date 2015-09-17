// Author: jinliang
#pragma once

#include <stdint.h>

#include <functional>
#include <boost/noncopyable.hpp>
#include <memory>
#include <string.h>

#include <petuum_ps_common/oplog/abstract_row_oplog.hpp>
#include <glog/logging.h>

// 和原来的，没有加version的数据相比，我们的现在的版本，就是就是增加verison这个变量
// 好神奇的变量
// #

namespace petuum {
class VersionDenseRowOpLog : public virtual AbstractRowOpLog {
public:
  VersionDenseRowOpLog(InitUpdateFunc InitUpdate,
                CheckZeroUpdateFunc CheckZeroUpdate,
                size_t update_size,
                size_t row_size):
      AbstractRowOpLog(update_size),
      row_size_(row_size),
      oplogs_(new uint8_t[update_size*row_size]),
      InitUpdate_(InitUpdate),
      CheckZeroUpdate_(CheckZeroUpdate),
      version_(0),
      end_of_version_(false) {
    CHECK(row_size > 0);
    //#
    // 上来什么不做，直接就重置
    // #
    Reset();
  }

  virtual ~VersionDenseRowOpLog() { }

  void Reset() {
    memset(oplogs_.get(), 0, update_size_*row_size_);
  }

  void* Find(int32_t col_id) {
    return oplogs_.get() + col_id*update_size_;
  }

  const void* FindConst(int32_t col_id) const {
    return oplogs_.get() + col_id*update_size_;
  }

  void* FindCreate(int32_t col_id) {
    return Find(col_id);
  }

//#
// 上面的和其他的也没有什么区别，为何不直接继承一下呢
//
// 具体情况，需要用beyond_compare来分析一下吧
//
// version是用64位存的
// cmu的同学，为什么要在这么底层的层次上做处理
// #
// #
  void SetVersion(uint64_t version) {
    version_ = version;
  }

  void SetEndOfVersion(bool end_of_version) {
    end_of_version_ = end_of_version;
  }

  //#
  // 疯了，写的代码这么晦涩啊，太服了
  // #
  uint64_t GetVersion(const uint8_t *bytes, size_t num_updates) const {
    return *(reinterpret_cast<const uint64_t*>(bytes + num_updates*update_size_));
  }

  bool GetEndOfVersion(const uint8_t *bytes, size_t num_updates) const {
    return *(reinterpret_cast<const bool*>(bytes + num_updates*update_size_
                                     + sizeof(uint64_t)));
  }

  // Guaranteed ordered traversal
  void* BeginIterate(int32_t *column_id) {
    iter_col_id_ = 0;
    return Next(column_id);
  }

  void* Next(int32_t *column_id) {
    uint8_t *update = reinterpret_cast<uint8_t*>(Find(iter_col_id_));
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
        + update_size_*num_nonzeros_ + sizeof(uint64_t)
        + sizeof(bool);
  }

  size_t GetDenseSerializedSize() {
    return row_size_*update_size_ + sizeof(uint64_t)
        + sizeof(bool);
  }

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

    uint8_t *mem_uint8 = reinterpret_cast<uint8_t*>(mem);

    *reinterpret_cast<uint64_t*>(mem_uint8 + num_nonzeros_*update_size_)
        = version_;
    *reinterpret_cast<bool*>(mem_uint8 + num_nonzeros_*update_size_
                                 + sizeof(uint64_t)) = end_of_version_;

    return GetSparseSerializedSize();
  }

  size_t SerializeDense(void *mem) {
    memcpy(mem, oplogs_.get(), AbstractRowOpLog::update_size_*row_size_);
    uint8_t *mem_uint8 = reinterpret_cast<uint8_t*>(mem);

    *reinterpret_cast<uint64_t*>(mem_uint8 + row_size_*update_size_)
        = version_;
    *reinterpret_cast<bool*>(mem_uint8 + row_size_*update_size_
                             + sizeof(uint64_t)) = end_of_version_;

    return GetDenseSerializedSize();
  }

  const void *ParseDenseSerializedOpLog(const void *mem,
                                        int32_t *num_updates,
                                        size_t *serialized_size) const {
    *num_updates = row_size_;
    *serialized_size = row_size_*AbstractRowOpLog::update_size_
                       + sizeof(uint64_t) + sizeof(bool);
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
  size_t num_nonzeros_;
  std::unique_ptr<uint8_t[]> oplogs_;
  const InitUpdateFunc InitUpdate_;
  const CheckZeroUpdateFunc CheckZeroUpdate_;
  mutable int32_t iter_col_id_;
  uint64_t version_;
  bool end_of_version_; // denote that this is the last row oplog
  // for this row in this version
  // all future oplogs of this row will have a higher version number
};
}
