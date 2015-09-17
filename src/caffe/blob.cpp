#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
// -------------------------
// modification part
#include "caffe/context.hpp"
#include "../../include/caffe/blob.hpp"
#include "../../include/caffe/syncedmem.hpp"
#include "../../include/caffe/context.hpp"

#include <petuum_ps_common/include/petuum_ps.hpp>
#include <petuum_ps_common/include/table_gflags_declare.hpp>
#include <petuum_ps_common/include/init_table_config.hpp>
// -------------------------

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
// -------------------------
// modification part
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(height, 0);
  CHECK_GE(width, 0);
// -------------------------
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
// -------------------------
// modification part
  //CHECK(data_) << " count " << count_ << " "<< num_ << " " << channels_ 
  //             << " " << height_ << " " << width_ << " capacity " << capacity_;
  if(blob_mode_ == BlobProto_BlobMode_GLOBAL) {
    const int num_rows_per_table = util::Context::num_rows_per_table();
    global_table_row_capacity_ = (count_ + num_rows_per_table - 1) / num_rows_per_table;
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}
// ---------------------------------------------------------------------------
// modification part
template <typename Dtype>
void Blob<Dtype>::ReshapeWithoutAllocation(const int num, const int channels, 
    const int height, const int width) {
  // call Reshap() directly since SyncedMemory allocates memory lazily
  Reshape(num, channels, height, width);    
}
// ---------------------------------------------------------------------------
template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}
// ---------------------------------------------------------------------------
// modification part
template <typename Dtype>
void Blob<Dtype>::ReshapeWithoutAllocationLike(const Blob<Dtype>& other) {
  ReshapeWithoutAllocation(other.num(), other.channels(), other.height(), 
      other.width());
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// modification part
template <typename Dtype>
void Blob<Dtype>::CreatePSTable() {
  CHECK_GE(global_id_, 0);
  CHECK_GE(count_, 0);
  
  util::Context& context = util::Context::get_instance();
  int param_table_staleness = context.get_int32("table_staleness");
  int num_rows_per_table = context.num_rows_per_table();

  // Creating PS tables 
  petuum::ClientTableConfig table_config;
  petuum::InitTableConfig(&table_config);

  table_config.table_info.row_type = caffe::kDenseRowDtypeID;
  table_config.table_info.table_staleness = param_table_staleness;
  table_config.process_cache_capacity = num_rows_per_table * 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  table_config.thread_cache_capacity = 1;
  global_table_row_capacity_
      = (count_ + num_rows_per_table - 1) / num_rows_per_table;
  table_config.table_info.row_capacity = global_table_row_capacity_;
  table_config.table_info.dense_row_oplog_capacity
      = global_table_row_capacity_;
  table_config.no_oplog_replay = true;

  petuum::PSTableGroup::CreateTable(global_id_, table_config);
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width, const BlobProto_BlobMode blob_mode, const int global_id)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0), blob_mode_(blob_mode), global_id_(global_id) {
  if(blob_mode_ == BlobProto_BlobMode_GLOBAL) {
    ReshapeWithoutAllocation(num, channels, height, width);
  } else {
    Reshape(num, channels, height, width);
  }
}
// ---------------------------------------------------------------------------

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}
// ---------------------------------------------------------------------------
// modification part
template <typename Dtype>
void Blob<Dtype>::UpdatePSTable() {
  CHECK_EQ(blob_mode_, BlobProto_BlobMode_GLOBAL);

  // flush diff_
  const Dtype* update = static_cast<const Dtype*>(diff_->cpu_data());
  int update_idx = 0;  
  for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
    petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity_);
    for (int i = 0; i < global_table_row_capacity_; ++i) {
      update_batch.UpdateSet(i, i, Dtype(-1) * update[update_idx]);
      ++update_idx;
      if (update_idx >= count_) { break; }
    }
    global_table_ptr_->BatchInc(r, update_batch);
    if (update_idx >= count_) { break; }
  }
}

template <typename Dtype>
void Blob<Dtype>::UpdatePSTable(const Dtype* update) {
  CHECK_EQ(blob_mode_, BlobProto_BlobMode_GLOBAL);

  int update_idx = 0;
  for (int r = 0; r < util::Context::num_rows_per_table(); ++r) {
    petuum::UpdateBatch<Dtype> update_batch(global_table_row_capacity_);
    for (int i = 0; i < global_table_row_capacity_; ++i) {
      update_batch.UpdateSet(i, i, Dtype(-1) * update[update_idx]);
      ++update_idx;
      if (update_idx >= count_) { break; }
    }
    global_table_ptr_->BatchInc(r, update_batch);
    if (update_idx >= count_) { break; }
  }
}

template <typename Dtype>
void Blob<Dtype>::SyncWithPSTable(const int clock) {
  CHECK_EQ(blob_mode_, BlobProto_BlobMode_GLOBAL);
  Dtype* data_temp = ReadPSTable(clock);
  data_->set_cpu_ps_data(data_temp);
}

template <typename Dtype>
Dtype* Blob<Dtype>::ReadPSTable(const int clock) const {
  CHECK(global_table_ptr_);
  
  void* data_temp;
  CaffeMallocHost(&data_temp, capacity_ * sizeof(Dtype));
  Dtype* data = (Dtype*)data_temp;

  vector<vector<Dtype> > row_caches(util::Context::num_rows_per_table());
  for (int r_idx = 0; r_idx < util::Context::num_rows_per_table(); ++r_idx) {
    row_caches[r_idx].resize(global_table_row_capacity_);
    petuum::RowAccessor row_acc;
    //LOG(INFO) << "get clock " << clock << " count " << count_ << " height " << height_ << " width " << width_ << " channel " << channels_ << " num " << num_;
    const auto& r = global_table_ptr_->template Get<petuum::DenseRow<Dtype> >(
        r_idx, &row_acc, clock);
    r.CopyToVector(&row_caches[r_idx]);
    //LOG(INFO) << "get done";
  }

  int data_idx = 0;
  for (int r_idx = 0; r_idx < util::Context::num_rows_per_table(); ++r_idx) {
    for (int i = 0; i < global_table_row_capacity_; ++i) {
      data[data_idx] = row_caches[r_idx][i];
      ++data_idx;
      if (data_idx >= count_) { break; }
    }
    if (data_idx >= count_) { break; }
  } 

  // release memory
  for (int r_idx = 0; r_idx < util::Context::num_rows_per_table(); ++r_idx) {
    vector<Dtype>().swap(row_caches[r_idx]);
  }
  vector<vector<Dtype> >().swap(row_caches);

  return data;
}
// ---------------------------------------------------------------------------


template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
// ---------------------------------------------------------------------------
// modification part
  if (blob_mode_ == BlobProto_BlobMode_GLOBAL) {
    if (!copy_diff) {
      LOG(FATAL) << "Currently Petuum Caffe does not support "
                 << "copying data to blobs with GLOBAL mode";
	}
  }
// ---------------------------------------------------------------------------
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape, const bool init_ps_table) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
// ---------------------------------------------------------------------------
// modification part
  if (blob_mode_ == BlobProto_BlobMode_GLOBAL) {
    if (init_ps_table) { // initialize ps table
      // update values in ps table
      Dtype* data_vec = ReadPSTable(0);
      for (int i = 0; i < count_; ++i) {
        data_vec[i] = data_vec[i] - proto.data(i);
      }
      diff_->set_cpu_data(data_vec);
      UpdatePSTable();
    }
  } else { 
    // copy data
    Dtype* data_vec = mutable_cpu_data();
    if (proto.double_data_size() > 0) {
      CHECK_EQ(count_, proto.double_data_size());
      for (int i = 0; i < count_; ++i) {
        data_vec[i] = proto.double_data(i);
      }
    } else {
      CHECK_EQ(count_, proto.data_size());
      for (int i = 0; i < count_; ++i) {
        data_vec[i] = proto.data(i);
      }
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}
// ---------------------------------------------------------------------------
// modification part
// add "typename Dtype" "double, float" ???=>??? "Dtype"
template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
// ---------------------------------------------------------------------------
// modification part  
  proto->set_blob_mode(blob_mode_);
  proto->set_global_id(global_id_);
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
// ---------------------------------------------------------------------------
// modification part  
  proto->set_blob_mode(blob_mode_);
  proto->set_global_id(global_id_);  
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

