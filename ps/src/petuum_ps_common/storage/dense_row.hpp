#pragma once

#include <petuum_ps_common/storage/numeric_store_row.hpp>
#include <petuum_ps_common/storage/vector_store.hpp>
#include <petuum_ps_common/storage/ns_count_calc.hpp>

namespace petuum {
// using 别名，DenseRowCore就是等式后面的别名
// DenseRowCore<mytype>的意思就是：NumericStoreRow<VectorStore, mytype >
template<typename V>
using DenseRowCore  = NumericStoreRow<VectorStore, V >;

//template<typename V>
//using DenseRowCore  = NumericStoreRow<VectorStore, V, NSCountImplCalc>;
//定义Row数据
template<typename V>
class DenseRow : public DenseRowCore<V> {
public:
  DenseRow() { }
  ~DenseRow() { }
//定义操作符[]，作用为提取第col_id列的数据
  V operator [](int32_t col_id) const {
    std::unique_lock<std::mutex> lock(DenseRowCore<V>::mtx_);
    return DenseRowCore<V>::store_.Get(col_id);
  }
//把to中的数据拷贝到vector中
  // Bulk read. Thread-safe.
  void CopyToVector(std::vector<V> *to) const {
    std::unique_lock<std::mutex> lock(DenseRowCore<V>::mtx_);
    DenseRowCore<V>::store_.CopyToVector(to);
  }
//将to中的数据拷贝到mem中
  void CopyToMem(void *to) const {
    std::unique_lock<std::mutex> lock(DenseRowCore<V>::mtx_);
    DenseRowCore<V>::store_.CopyToMem(to);
  }
//获取数据指针
  // not thread-safe
  const void *GetDataPtr() const {
    return DenseRowCore<V>::store_.GetDataPtr();
  }
};

}
