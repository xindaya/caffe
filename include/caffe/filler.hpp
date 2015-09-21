// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.
//目前未实现MSRAFiller和BilinearFiller功能，待开发

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>
//bosen新加
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
//bosen新加
#include "caffe/context.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
  //bosen新加
  virtual void FillPSTable(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
//bosen新加内容FillPSTable
//virtual void FillPSTable(Blob<Dtype>* blob) {
  //  const int count = blob->count();
  //  const Dtype value = this->filler_param_.value();
  //  CHECK(count);
  //  petuum::UpdateBatch<Dtype> update_batch(count);
  //  for (int i = 0; i < count; ++i) {
  //    update_batch.UpdateSet(i, i, value);
  //  }
  //  blob->table()->BatchInc(1, update_batch);
  //  CHECK_EQ(this->filler_param_.sparse(), -1)
  //       << "Sparsity not supported by this Filler.";
  //}
  virtual void FillPSTable(Blob<Dtype>* blob) {
    const int count = blob->count();
    const int global_table_row_capacity = blob->global_table_row_capacity();
    const Dtype value = this->filler_param_.value();
    int update_idx = 0;
    for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
      petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity);
      for (int i = 0; i < global_table_row_capacity; ++i) {
        update_batch.UpdateSet(i, i, value);
        ++update_idx;
        if (update_idx >= count) { break; }
      }
      blob->table()->BatchInc(r, update_batch);
      if (update_idx >= count) { break; }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
//bosen新加内容
 //virtual void FillPSTable(Blob<Dtype>* blob) {
  //  const int count = blob->count();
  //  CHECK(count);
  //  Dtype* rn = new Dtype[count];
  //  caffe_rng_uniform<Dtype>(count, Dtype(this->filler_param_.min()),
  //      Dtype(this->filler_param_.max()), rn);
  //  petuum::UpdateBatch<Dtype> update_batch(count);
  //  for (int i = 0; i < count; ++i) {
  //    update_batch.UpdateSet(i, i, rn[i]);
  //  }
  //  blob->table()->BatchInc(1, update_batch);
  //  delete rn;

  //  CHECK_EQ(this->filler_param_.sparse(), -1)
  //       << "Sparsity not supported by this Filler.";
  //}
  virtual void FillPSTable(Blob<Dtype>* blob) {
    const int count = blob->count();
    const int global_table_row_capacity = blob->global_table_row_capacity();
    Dtype* rn = new Dtype[count];
    caffe_rng_uniform<Dtype>(count, Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), rn);

    int update_idx = 0;
    for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
      petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity);
      for (int i = 0; i < global_table_row_capacity; ++i) {
        update_batch.UpdateSet(i, i, rn[update_idx]);
        ++update_idx;
        if (update_idx >= count) { break; }
      }
      blob->table()->BatchInc(r, update_batch);
      if (update_idx >= count) { break; }
    }

    delete rn;
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
      GenerateSparseGaussianRN(blob, blob->mutable_cpu_data());//bosen将生成稀疏高斯RN的功能集成为函数
  }
  //bosen新加内容
  //virtual void FillPSTable(Blob<Dtype>* blob) {
  //  const int count = blob->count();
  //  CHECK(count);
  //  Dtype* rn = new Dtype[count];
  //  GenerateSparseGaussianRN(blob, rn);
  //  petuum::UpdateBatch<Dtype> update_batch(count);
  //  for (int i = 0; i < count; ++i) {
  //    update_batch.UpdateSet(i, i, rn[i]);
  //  }
  //  blob->table()->BatchInc(1, update_batch);
  //  delete rn;
  //}
  virtual void FillPSTable(Blob<Dtype>* blob) {
    const int count = blob->count();
    const int global_table_row_capacity = blob->global_table_row_capacity();
    Dtype* rn = new Dtype[count];
    GenerateSparseGaussianRN(blob, rn);

    int update_idx = 0;
    for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
      petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity);
      for (int i = 0; i < global_table_row_capacity; ++i) {
        update_batch.UpdateSet(i, i, rn[update_idx]);
        ++update_idx;
        if (update_idx >= count) { break; }
      }
      blob->table()->BatchInc(r, update_batch);
      if (update_idx >= count) { break; }
    }

    delete rn;
  }
    //原生caffe里的东西，在此处注释之
	//Dtype* data = blob->mutable_cpu_data();
protected:
//bosen将功能用函数GenerateSparseGaussianRN集成
  void GenerateSparseGaussianRN(Blob<Dtype>* blob, Dtype* rn) {
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), rn);
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        rn[i] *= mask[i];
      }
    }
  }

 //protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    GeneratePositiveUnitballRN(blob, blob->mutable_cpu_data());////bosen将生成RN的功能集成为函数

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }

//bosen新加内容
//virtual void FillPSTable(Blob<Dtype>* blob) {
  //  const int count = blob->count();
  //  DCHECK(count);
  //  Dtype* rn = new Dtype[count];
  //  GeneratePositiveUnitballRN(blob, rn);
  //  petuum::UpdateBatch<Dtype> update_batch(count);
  //  for (int i = 0; i < count; ++i) {
  //    update_batch.UpdateSet(i, i, rn[i]);
  //  }
  //  blob->table()->BatchInc(1, update_batch);
  //  delete rn;

  //  CHECK_EQ(this->filler_param_.sparse(), -1)
  //       << "Sparsity not supported by this Filler.";
  //}
  virtual void FillPSTable(Blob<Dtype>* blob) {
    const int count = blob->count();
    const int global_table_row_capacity = blob->global_table_row_capacity();
    Dtype* rn = new Dtype[count];
    GeneratePositiveUnitballRN(blob, rn);

    int update_idx = 0;
    for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
      petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity);
      for (int i = 0; i < global_table_row_capacity; ++i) {
        update_batch.UpdateSet(i, i, rn[update_idx]);
        ++update_idx;
        if (update_idx >= count) { break; }
      }
      blob->table()->BatchInc(r, update_batch);
      if (update_idx >= count) { break; }
    }

    delete rn;
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
    //原生caffe里的东西，在此处注释之
	//Dtype* data = blob->mutable_cpu_data();
	protected:
  void GeneratePositiveUnitballRN(Blob<Dtype>* blob, Dtype* rn) {
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, rn);
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += rn[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        rn[i * dim + j] /= sum;
      }
    }
    //bosen将该判断语句放在了最前面
	//CHECK_EQ(this->filler_param_.sparse(), -1)
        // << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  
  //bosen新加内容
  //virtual void FillPSTable(Blob<Dtype>* blob) {
  //  const int count = blob->count();
  //  CHECK(count);
  //  int fan_in = blob->count() / blob->num();
  //  Dtype scale = sqrt(Dtype(3) / fan_in);
  //  Dtype* rn = new Dtype[count];
  //  caffe_rng_uniform<Dtype>(blob->count(), -scale, scale, rn);
  //  petuum::UpdateBatch<Dtype> update_batch(count);
  //  for (int i = 0; i < count; ++i) {
  //    update_batch.UpdateSet(i, i, rn[i]);
  //  }
  //  blob->table()->BatchInc(1, update_batch);
  //  delete rn;

  //  CHECK_EQ(this->filler_param_.sparse(), -1)
  //       << "Sparsity not supported by this Filler.";
  //}
  virtual void FillPSTable(Blob<Dtype>* blob) {
    const int count = blob->count();
    const int global_table_row_capacity = blob->global_table_row_capacity();
    int fan_in = blob->count() / blob->num();
	int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    Dtype* rn = new Dtype[count];
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale, rn);

    int update_idx = 0;
    for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
      petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity);
      for (int i = 0; i < global_table_row_capacity; ++i) {
        update_batch.UpdateSet(i, i, rn[update_idx]);
        ++update_idx;
        if (update_idx >= count) { break; }
      }
      //LOG(INFO) << "test get " << r;
      //petuum::RowAccessor row_acc;
      //blob->table()->template Get<petuum::DenseRow<Dtype> >(
      //  r, &row_acc, 0);
      //LOG(INFO) << "batch inc " << r;
      blob->table()->BatchInc(r, update_batch);
      //LOG(INFO) << "batch inc done " << r;
      if (update_idx >= count) { break; }
    }
    //
    //for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
    //  petuum::DenseUpdateBatch<Dtype> update_batch(0, global_table_row_capacity);
    //  for (int i = 0; i < global_table_row_capacity; ++i) {
    //    update_batch[i] = rn[update_idx];
    //    ++update_idx;
    //    if (update_idx >= count) { break; }
    //  }
    //  blob->table()->DenseBatchInc(r, update_batch);
    //  if (update_idx >= count) { break; }
    //}

    delete rn;
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
template <typename Dtype>
class MSRAFiller : public Filler<Dtype> {
 public:
  explicit MSRAFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode
 */
template <typename Dtype>
class BilinearFiller : public Filler<Dtype> {
 public:
  explicit BilinearFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
 //暂不实现MSRAFiller和BilinearFiller
 // } else if (type == "msra") {
 //   return new MSRAFiller<Dtype>(param);
 // } else if (type == "bilinear") {
 //   return new BilinearFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
