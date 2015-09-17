// Author: Jinliang, Yihua Fang

#pragma once

#include <petuum_ps_common/include/row_access.hpp>
#include <petuum_ps_common/client/client_row.hpp>

#include <boost/noncopyable.hpp>

// 正如其名所示，该类是进程内部的存储

namespace petuum {
// 采用了boost::noncopyable 所以有单例对象的行为方式
class AbstractProcessStorage : boost::noncopyable {
public:
  // capacity is the upper bound of the number of rows this ProcessStorage
  // can store.

  // 容量是可以在进程中存储的行数的一个上限
  AbstractProcessStorage() { }

  virtual ~AbstractProcessStorage() { }

  // Look up for row_id. If found, return a pointer pointing to that row,
  // otherwise return 0.
  // If the storage is a evictable type, then the row_accessor is set
  // to the corresponding row, which maintains the reference count.
  // Otherwise, row_accessor is not used and can actually be a NULL pointer.

  // 查找 row_id, 如果找到了就返回相应行的一个指针
  // 如果没有找到，那么返回0
  // 如果storage是evictable 类型，那么 row_accessor 被设置为相应的列， 保留reference count
  // 否则， row_accessor 不用设置， 实际上是一个NULL类型的指针
  virtual ClientRow *Find(int32_t row_id, RowAccessor* row_accessor) = 0;

  virtual bool Find(int32_t row_id) = 0;

  // Insert a row, and take ownership of client_row.
  // Insertion failes if the row has already existed and return false; otherwise
  // return true.

  // 插入一行， 同时获取client_row 的所有权
  // 如果该行已经存在,那么失败，否则true
  virtual bool Insert(int32_t row_id, ClientRow* client_row) = 0;
};


}  // namespace petuum
