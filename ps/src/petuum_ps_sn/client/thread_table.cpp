#include <petuum_ps_sn/client/thread_table.hpp>
#include <petuum_ps_sn/thread/context.hpp>
#include <petuum_ps_common/include/row_access.hpp>
#include <petuum_ps_common/storage/process_storage.hpp>

#include <glog/logging.h>
#include <functional>

namespace petuum {

ThreadTableSN::ThreadTableSN(const AbstractRow *sample_row) :
    sample_row_(sample_row){ }

ThreadTableSN::~ThreadTableSN() {
  for (auto iter = row_storage_.begin(); iter != row_storage_.end(); iter++) {
    if (iter->second != 0)
      delete iter->second;
  }

  for (auto iter = oplog_map_.begin(); iter != oplog_map_.end(); iter++) {
    if (iter->second != 0)
      delete iter->second;
  }
}

AbstractRow *ThreadTableSN::GetRow(int32_t row_id) {
  boost::unordered_map<int32_t, AbstractRow* >::iterator row_iter
      = row_storage_.find(row_id);
  if (row_iter == row_storage_.end()) {
    return 0;
  }
  return row_iter->second;
}

void ThreadTableSN::InsertRow(int32_t row_id, const AbstractRow *to_insert) {
  AbstractRow *row = to_insert->Clone();
  boost::unordered_map<int32_t, AbstractRow* >::iterator row_iter
      = row_storage_.find(row_id);
  if (row_iter != row_storage_.end()) {
    delete row_iter->second;
    row_iter->second = row;
  } else {
    row_storage_[row_id] = row;
  }

  boost::unordered_map<int32_t, RowOpLog* >::iterator oplog_iter
      = oplog_map_.find(row_id);
  if (oplog_iter != oplog_map_.end()) {
    int32_t column_id;
    void *delta = oplog_iter->second->BeginIterate(&column_id);
    while (delta != 0) {
      row->ApplyInc(column_id, delta);
      delta = oplog_iter->second->Next(&column_id);
    }
  }
}

void ThreadTableSN::Inc(int32_t row_id, int32_t column_id, const void *delta) {
  boost::unordered_map<int32_t, RowOpLog* >::iterator oplog_iter
      = oplog_map_.find(row_id);

  RowOpLog *row_oplog;
  if (oplog_iter == oplog_map_.end()) {
    row_oplog = new RowOpLog(sample_row_->get_update_size(),
                             std::bind(&AbstractRow::InitUpdate,
                                       sample_row_,
                                       std::placeholders::_1,
                                       std::placeholders::_2));
    oplog_map_[row_id] = row_oplog;
  } else {
    row_oplog = oplog_iter->second;
  }

  void *oplog_delta = row_oplog->FindCreate(column_id);
  sample_row_->AddUpdates(column_id, oplog_delta, delta);

  boost::unordered_map<int32_t, AbstractRow* >::iterator row_iter
      = row_storage_.find(row_id);
  if (row_iter != row_storage_.end()) {
    row_iter->second->ApplyIncUnsafe(column_id, delta);
  }
}

void ThreadTableSN::BatchInc(int32_t row_id, const int32_t *column_ids,
                           const void *deltas, int32_t num_updates) {
  boost::unordered_map<int32_t, RowOpLog* >::iterator oplog_iter
      = oplog_map_.find(row_id);

  RowOpLog *row_oplog;

  if (oplog_iter == oplog_map_.end()) {
    row_oplog = new RowOpLog(sample_row_->get_update_size(),
                             std::bind(&AbstractRow::InitUpdate, sample_row_,
                                       std::placeholders::_1,
                                       std::placeholders::_2));
    oplog_map_[row_id] = row_oplog;
  } else {
    row_oplog = oplog_iter->second;
  }

  const uint8_t* deltas_uint8 = reinterpret_cast<const uint8_t*>(deltas);

  for (int i = 0; i < num_updates; ++i) {
    void *oplog_delta = row_oplog->FindCreate(column_ids[i]);
    sample_row_->AddUpdates(column_ids[i], oplog_delta, deltas_uint8
                            + sample_row_->get_update_size()*i);
  }

  boost::unordered_map<int32_t, AbstractRow* >::iterator row_iter
      = row_storage_.find(row_id);
  if (row_iter != row_storage_.end()) {
    row_iter->second->ApplyBatchIncUnsafe(column_ids, deltas, num_updates);
  }
}

void ThreadTableSN::FlushCache(ProcessStorage &process_storage,
			     const AbstractRow *sample_row) {
  for (auto oplog_iter = oplog_map_.begin(); oplog_iter != oplog_map_.end();
       oplog_iter++) {
    int32_t row_id = oplog_iter->first;

    RowAccessor row_accessor;
    bool found = process_storage.Find(row_id, &row_accessor);

    int32_t column_id;
    void *delta = oplog_iter->second->BeginIterate(&column_id);

    if (found) {
      while (delta != 0) {
	row_accessor.GetRowData()->ApplyInc(column_id, delta);
	delta = oplog_iter->second->Next(&column_id);
      }
    }
    delete oplog_iter->second;
  }
  oplog_map_.clear();

  for (auto iter = row_storage_.begin(); iter != row_storage_.end(); iter++) {
    if (iter->second != 0)
      delete iter->second;
  }
  row_storage_.clear();
}

}
