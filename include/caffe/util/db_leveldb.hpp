#ifndef CAFFE_UTIL_DB_LEVELDB_HPP
#define CAFFE_UTIL_DB_LEVELDB_HPP

#include <string>
// 引入了leveldb的api
//http://duanple.blog.163.com/blog/static/70971767201171705113636/
// 上面的链接给出了leveldb的操作详细信息
// leveldb的api非常清晰易懂
#include "leveldb/db.h"
// leveldb提供的批处理操作
#include "leveldb/write_batch.h"

#include "caffe/util/db.hpp"

// leveldb 的核心数据结构
// batch
// iteration
namespace caffe { namespace db {
// 将leveldb的iterator封装为游标cursor
class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::Iterator* iter)
    : iter_(iter) { SeekToFirst(); }
  ~LevelDBCursor() { delete iter_; }
  virtual void SeekToFirst() { iter_->SeekToFirst(); }
  virtual void Next() { iter_->Next(); }
  virtual string key() { return iter_->key().ToString(); }
  virtual string value() { return iter_->value().ToString(); }
  virtual bool valid() { return iter_->Valid(); }

 private:
  leveldb::Iterator* iter_;
};

/*
 * 不理解的是为什么要时实现不同的组件,为什么不搞一个大点的呢
 * 将 transaction cusor 都放到db这个类里面?
 * 为什么呢?
 * 看不透
 * */

// 将leveldb提供的批量写,封装为一个事务
class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) { CHECK_NOTNULL(db_); }
    // 一条一条的插入数据
  virtual void Put(const string& key, const string& value) {
    batch_.Put(key, value);
  }
    // 提交,写入到数据库中
  virtual void Commit() {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), &batch_);
    CHECK(status.ok()) << "Failed to write batch to leveldb "
                       << std::endl << status.ToString();
  }

 private:
    // 数据库藏在这里
  leveldb::DB* db_;
    // 执行事务操作的容器也藏在这里
  leveldb::WriteBatch batch_;

  DISABLE_COPY_AND_ASSIGN(LevelDBTransaction);
};


/*
 *实现的是DB
 * */
class LevelDB : public DB {
 public:
  LevelDB() : db_(NULL) { }
  virtual ~LevelDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (db_ != NULL) {
      delete db_;
      db_ = NULL;
    }
  }
    // 这种命名方式很像c style 作者是c系出身吧
  virtual LevelDBCursor* NewCursor() {
    return new LevelDBCursor(db_->NewIterator(leveldb::ReadOptions()));
  }
  virtual LevelDBTransaction* NewTransaction() {
    return new LevelDBTransaction(db_);
  }

 private:
  leveldb::DB* db_;
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LEVELDB_HPP
