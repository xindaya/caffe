#include <boost/thread.hpp>
#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

// 重要的事情是生成
// datatransformer
template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
	  data_transformer_(transform_param_){ }
//按照bosen下定义的LayerSetUp输入参数形式重新定义LayerSetUp()函数的输入。
template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const bool init_ps, int* num_tables,
     map<string, vector<int> >* layer_name_to_blob_global_idx) {
 //bosen新加内容
 // 获取phase

 data_transformer_.set_phase(Caffe::phase(this->thread_id_));

  // 根据是否有label来处理,有label,将output_labels 设置为true
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // 这个工具类,用来执行的操作是对datum做处理
  // 根据transform的参数来初始化
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  //bosen下的DataLayerSet()函数与原生caffe的不同，增加了init_ps参数
  DataLayerSetUp(bottom, top, init_ps);
}

// 执行的是数据预取, 与主线程不在一个路径上
template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      // 注意prefetch_free_
      // 注意prefetch_full_
      prefetch_free_(), prefetch_full_() {
    // 一次性读PREFETCH_COUNT个.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
      // 这个是将prefetch代表的指针入队列
      // 相当于将存放free(没有具体内容的指针)填满
    prefetch_free_.push(&prefetch_[i]);
  }
}

// 数据预取是一个layer了
// 这样的设计非常棒.

// 简单的说
// 1. 申请内存空间
// 2. 启动预取线程
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
	bool init_ps, int* num_tables,
    map<string, vector<int> >* layer_name_to_blob_global_idx) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top, init_ps, num_tables, 
      layer_name_to_blob_global_idx);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.


  // 避免并发做内存操作
  // 并发做操作,会有什么问题??
  // 内存不够?
  // 还是什么问题
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
// 这一步的目的是为了开辟内存空间
    prefetch_[i].data_.mutable_cpu_data();
    // 有么有label
    if (this->output_labels_) {
// 如果有label, 也开辟相应的内存空间
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
// 如果使用gpu mode, 那么GPU也要申请空间
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
// 申请label的gpu空间
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
// 启动线程
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
 }


template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  // cudaStream_t 的结构体,在这里生成的
    // 实现内存到线程的异步拷贝
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
      // while循环
      // 不到must-stop,坚决不停止
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      // 这里就是加载数据的核心代码
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
          // 如果使用gpu,异步将数据发送到gpu显存
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      // 取出来的batch放到了full这个blockingqueue里
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  // batch 是datum类型的,所以内部存储了data和label
  // 如果有label,需要把label的数据也放到blob里
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
