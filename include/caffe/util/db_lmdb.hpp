#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string>

#include "lmdb.h"

#include "caffe/util/db.hpp"


// 好吧,我得承认,我确实只知道leveldb,不知道lmdb
// leveldb是bigtable的单机实现,在学习bigtable的时候,顺便学习了一把leveldb
// 但是lmdb从来没有接触过
// 下面开始照葫芦画瓢过程
//http://rayz0620.github.io/2015/05/25/lmdb_in_caffe/
// 上面这个链接讲的非常好
namespace caffe { namespace db {

// 这个函数封装的不错
inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor)
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
  }
  virtual void SeekToFirst() { Seek(MDB_FIRST); }
  virtual void Next() { Seek(MDB_NEXT); }
// 好别扭的转换,习惯就好,习惯就好,习惯就好
  virtual string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

 private:

    // 这个方法藏得够深,不过还好,统一了db的接口,
    // 对后续的工作就提供了一个好的开始
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  /*
   * mdb_env是整个数据库环境的句柄，mdb_dbi是环境中一个数据库的句柄;
   * mdb_key和mdb_data用来存放向数据库中输入数据的“值”;
   * mdb_txn是数据库事物操作的句柄，”txn”是”transaction”的缩写;
   *
   * */
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
};

class LMDBTransaction : public Transaction {
 public:
    // MDB_dbi 这个是lmdb数据库的句柄吧
    // MDB_txn 是事务实现的方式
  explicit LMDBTransaction(MDB_dbi* mdb_dbi, MDB_txn* mdb_txn)
    : mdb_dbi_(mdb_dbi), mdb_txn_(mdb_txn) { }

    // leveldb 还有个容器,这个lmdb的容器呢?,是放到mdb_txn_中间吗?
  virtual void Put(const string& key, const string& value);
  virtual void Commit() { MDB_CHECK(mdb_txn_commit(mdb_txn_)); }

 private:
  MDB_dbi* mdb_dbi_;
  MDB_txn* mdb_txn_;

  DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual LMDBCursor* NewCursor();
  virtual LMDBTransaction* NewTransaction();

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
