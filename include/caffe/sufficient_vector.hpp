#ifndef CAFFE_SUFFICIENT_VECTOR_HPP_
#define CAFFE_SUFFICIENT_VECTOR_HPP_

#include <cstdlib>
#include <cstdio>

#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "syncedmem.hpp"

namespace caffe {
// 根据这个类的原始意义来看,这个类的是一个高效的vector实现
// 看里面的api,好像是模仿的blob实现

/**
 * @brief 
 */
class SufficientVector {
 public:
  explicit SufficientVector(const size_t a_size, const size_t b_size,
      const int layer_id);
  SufficientVector() : a_size_(0), b_size_(0), layer_id_(-1) {}
  ~SufficientVector();

  void Reshape(const size_t a_size, const size_t b_size);

  inline int layer_id() { return layer_id_; } 
  const void* cpu_a() const;
  const void* gpu_a() const;
  const void* cpu_b() const;
  const void* gpu_b() const;
  inline size_t a_size() const { return a_size_; }
  inline size_t b_size() const { return b_size_; }
  void* mutable_cpu_a();
  void* mutable_gpu_a();
  void* mutable_cpu_b();
  void* mutable_gpu_b();
  
  template <typename Dtype>
  void FromProto(const SVProto& proto);
  template <typename Dtype>
  void ToProto(SVProto* proto) const;

 private:
  // M = a_ x b_^{T}
    // 不存储矩阵,用来存储两个向量
// 倒还是使用的底层的实现,怎么感觉a和b其实是向量
  shared_ptr<SyncedMemory> a_;
  shared_ptr<SyncedMemory> b_;
  size_t a_size_;
  size_t b_size_;
  int layer_id_;

};  // class SufficientVector 

}  // namespace caffe

#endif  // CAFFE_SUFFICIENT_VECTOR_HPP_
