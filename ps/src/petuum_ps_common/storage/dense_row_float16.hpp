#pragma once

#include <petuum_ps_common/storage/numeric_store_row.hpp>
#include <petuum_ps_common/storage/vector_store_float16.hpp>
#include <petuum_ps_common/storage/ns_count_calc.hpp>

namespace petuum {

template<typename V>
using DenseRowFloat16Core  = NumericStoreRow<VectorStoreFloat16, V>;
//定义Float16数据
template<typename V>
class DenseRowFloat16 : public DenseRowFloat16Core<V> {
public:
  DenseRowFloat16() { }
  ~DenseRowFloat16() { }
//定义操作符[],作用为获取第col_id列的数据
  V operator [](int32_t col_id) const {
    std::unique_lock<std::mutex> lock(DenseRowFloat16Core<V>::mtx_);
    return DenseRowFloat16Core<V>::store_.Get(col_id);
  }
//将to中的数据拷贝到vector中
  // Bulk read. Thread-safe.
  void CopyToVector(std::vector<V> *to) const {
    std::unique_lock<std::mutex> lock(DenseRowFloat16Core<V>::mtx_);
    DenseRowFloat16Core<V>::store_.Copy(to);
  }

    //获取数据指针
  // not thread-safe
  const void *GetDataPtr() const {
    return DenseRowFloat16Core<V>::store_.GetDataPtr();
  }
};

}
