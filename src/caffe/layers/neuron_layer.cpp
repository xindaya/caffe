#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
//整个并未涉及bosen对该文件做的修改，修改的地方只是基于原生caffe的升级
namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
