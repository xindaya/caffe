#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
//按照bosen下定义的LayerSetUp输入参数形式重新定义LayerSetUp()函数的输入
template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const bool init_ps, int* num_tables,
    map<string, vector<int> >* layer_name_to_blob_global_idx) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  //bosen新加内容定义了layer的name
  string name = this->layer_param_.name();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
      if (init_ps) {//bosen新加内容，定义layer_name_to_blob_global_idx变量
        this->blob_global_id_.resize(2);
        (*layer_name_to_blob_global_idx)[name] = vector<int>(2);
      } 
    } else {
      this->blobs_.resize(1);
      if (init_ps) {//bosen新加内容，定义layer_name_to_blob_global_idx变量
        this->blob_global_id_.resize(1);
        (*layer_name_to_blob_global_idx)[name] = vector<int>(1);
      }  
    }
    // Intialize the weight
	//bosen新加内容，定义weight_table的id
	int weight_table_id = (init_ps ? *num_tables : -1);
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
	//按照bosen下的reset函数定义reset的输入
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_, BlobProto_BlobMode_GLOBAL, weight_table_id));
   //bosen下的weight初始化在solver中的InitTrainNet中，因此此处注释之
    // fill the weights
   // shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
       // this->layer_param_.inner_product_param().weight_filler()));
    //weight_filler->Fill(this->blobs_[0].get());
    //bosen新加内容，生成layer_name_to_blob_global_idx ，创建weight-table
	if (init_ps) {
      // Create the weight table
      this->blobs_[0]->CreatePSTable();
      this->blob_global_id_[0] = weight_table_id;
      (*layer_name_to_blob_global_idx)[name][0] = weight_table_id;
      ++(*num_tables);
    }
	
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      //bosen新加内容，定义bias-table的id
	   int bias_table_id = (init_ps ? *num_tables : -1);
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_, BlobProto_BlobMode_GLOBAL, bias_table_id));
      //bosen下的bias初始化在solver中的InitTrainNet中，因此此处注释之
	  //shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          //this->layer_param_.inner_product_param().bias_filler()));
      //bias_filler->Fill(this->blobs_[1].get());
	  //bosen新加内容，生成layer_name_to_blob_global_idx ，创建bias-table
      if (init_ps) {
        // Create the bias table
        this->blobs_[1]->CreatePSTable();
        this->blob_global_id_[1] = bias_table_id;
        (*layer_name_to_blob_global_idx)[name][1] = bias_table_id;
        ++(*num_tables);
      }
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}
//bosen新加函数，语义理解，从SV_cpu中计算梯度
template <typename Dtype>
void InnerProductLayer<Dtype>::ComputeGradientFromSV_cpu(
    const SufficientVector* v) {
  // Gradient with respect to weight
  const Dtype* top_diff = (const Dtype*)v->cpu_a();
  const Dtype* bottom_data = (const Dtype*)v->cpu_b();
  CHECK_EQ(v->a_size() / sizeof(Dtype), N_ * M_);
  CHECK_EQ(v->b_size() / sizeof(Dtype), K_ * M_);
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
}
#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
//bosen新加函数
STUB_GPU_SV(InnerProductLayer, ComputeGradientFromSV);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
