#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
//整个并未涉及bosen对该文件做的修改，修改的地方只是基于原生caffe的升级
namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_data());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe
