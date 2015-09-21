#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
//按照bosen下定义的LayerSetUp输入参数形式重新定义LayerSetUp()函数的输入
//整个并未涉及bosen对该文件做的修改，修改的地方只是基于原生caffe的升级
namespace caffe {

template <typename Dtype>
void CuDNNReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const bool init_ps, int* num_tables,
    map<string, vector<int> >* layer_name_to_blob_global_idx) {
  ReLULayer<Dtype>::LayerSetUp(bottom, top,init_ps, num_tables,
      layer_name_to_blob_global_idx);
  // initialize cuDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReLULayer<Dtype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
}

template <typename Dtype>
CuDNNReLULayer<Dtype>::~CuDNNReLULayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroy(this->handle_);
}

INSTANTIATE_CLASS(CuDNNReLULayer);

}  // namespace caffe
#endif
