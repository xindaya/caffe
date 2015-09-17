// Author: Jinliang

#pragma once

#include <petuum_ps_common/oplog/abstract_append_only_buffer.hpp>
#include <petuum_ps_common/oplog/dense_append_only_buffer.hpp>
#include <petuum_ps_common/oplog/inc_append_only_buffer.hpp>
#include <petuum_ps_common/oplog/batch_inc_append_only_buffer.hpp>

#include <boost/noncopyable.hpp>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace petuum {

class BufferPool : boost::noncopyable {
public:
    BufferPool(int32_t thread_id,
               size_t pool_size, //池子的大小
               AppendOnlyOpLogType append_only_oplog_type,// oplog的类型
               size_t buff_capacity,//buff的容量
               size_t update_size, // update的数目
               size_t dense_row_capacity) : // dense_row的容量
      pool_size_(pool_size),
      num_buffs_in_pool_(pool_size),
      begin_(0),// pool头指针
      end_(0),
      pool_(pool_size) //我说pool_是什么，原来是这个啊，名字不一致啊，有没有！！
    // pool_到底是什么类型的数据，vector？ 在哪里定义的# -> 环状Buffer，具体见getbuffer
    //#
    //
    // 初始化的第一件事，就是先生成要用的
    // buff_ptr就是得到的pool的地址
    //
    // #
    {
    for (auto &buff_ptr : pool_) {
      buff_ptr = CreateAppendOnlyBuffer(thread_id,
                                        append_only_oplog_type,
                                        buff_capacity,
                                        update_size,
                                        dense_row_capacity);
    }
  }

  ~BufferPool() {
    for (auto &buff_ptr : pool_) {
      delete buff_ptr;
      buff_ptr = 0;
    }
  }

  AbstractAppendOnlyBuffer *GetBuff() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (num_buffs_in_pool_ == 0) cv_.wait(lock);

    num_buffs_in_pool_--;
    AbstractAppendOnlyBuffer *buff = pool_[begin_];
    pool_[begin_] = 0;
    begin_ = (begin_ + 1) % pool_size_;
    return buff;
  }

//#
// PutBuff这个动作，需要上锁
// 如果不上锁会怎么样？
// #
// #
  void PutBuff(AbstractAppendOnlyBuffer *buff) {
    std::unique_lock<std::mutex> lock(mtx_);
    CHECK(num_buffs_in_pool_ < pool_size_) << "buffer pool is full!";

    pool_[end_] = buff;
        //#
        // 使用取模的方式来做处理
        // #
    end_ = (end_ + 1) % pool_size_;
    num_buffs_in_pool_++;

    if (num_buffs_in_pool_ == 1)
      cv_.notify_all();
  }

private:
    //#
    // 这个函数是static 还是私有的
    // #
  static AbstractAppendOnlyBuffer *CreateAppendOnlyBuffer(
      int32_t thread_id,
      AppendOnlyOpLogType append_only_oplog_type,
      size_t buff_capacity, size_t update_size, size_t dense_row_capacity) {
    AbstractAppendOnlyBuffer *buff = 0;
    switch(append_only_oplog_type) {
        // #
        // c++ 这里的事情很奇怪
        // 不知道这里的对象机制是如何实现的
        // 就是使用抽象的类，来作为指针
        // 指向的是具体的实现
        // @
        // #
      case Inc:
        buff = static_cast<AbstractAppendOnlyBuffer*>(
            new IncAppendOnlyBuffer(thread_id, buff_capacity, update_size));
        break;
      case BatchInc:
        buff = static_cast<AbstractAppendOnlyBuffer*>(
            new BatchIncAppendOnlyBuffer(thread_id, buff_capacity, update_size));
        break;
      case DenseBatchInc:
        buff = static_cast<AbstractAppendOnlyBuffer*>(
            new DenseAppendOnlyBuffer(thread_id, buff_capacity, update_size,
                                      dense_row_capacity));
      break;
      default:
        LOG(FATAL) << "Unknown type = " << append_only_oplog_type;
    }
    return buff;
  }

  const size_t pool_size_;

  size_t num_buffs_in_pool_;
  int begin_, end_;
  std::vector<AbstractAppendOnlyBuffer*> pool_;
    //pool_的定义原来在这里，果然是vector类型的，
    // #不然怎么使用范围for这个工具#

    //生成全局的排他锁，但是为什么需要用锁呢
  std::mutex mtx_;

  std::condition_variable cv_;
};

class OpLogBufferManager {
public:
    OpLogBufferManager(
            size_t num_table_threads,
            int32_t head_thread_id) :

      head_thread_id_(head_thread_id),
            //#
            // 通过这个申明函数，这个buffer是按照thread的数目多少来设置的
            // 这里有什么意义呢
            //
            // #
            // 使用vec 作为后缀，表示我们这个类型是向量的
            // 下面用的是vector的构造方法，占位方式
            // #
      buffer_pool_vec_(num_table_threads) { }

  ~OpLogBufferManager() {
    for (auto &pool_ptr : buffer_pool_vec_) {
      delete pool_ptr;
      pool_ptr = 0;
    }
  }

  void CreateBufferPool(
          int32_t thread_id,
          size_t pool_size,
          AppendOnlyOpLogType append_only_oplog_type,
          size_t buff_capacity,
          size_t update_size,
          size_t dense_row_capacity) {
//#
// idx 表示索引
// thread_id -head_thread_id_
// 难道这个的意思是说，线程的id是连续的？
// 要不然为何使用这种方式#
//
// #
// #
      int32_t idx = thread_id - head_thread_id_;
    buffer_pool_vec_[idx] = new BufferPool(
            thread_id,
            pool_size,
            append_only_oplog_type,
            buff_capacity,
            update_size,
        dense_row_capacity);
  }

//#
// 这个返回的是一个引用吧
// 这里使用的是通过thread_id 来获取相应的索引位置
// #
  AbstractAppendOnlyBuffer *GetBuff(int32_t thread_id) {
    int32_t idx = thread_id - head_thread_id_;
    return buffer_pool_vec_[idx]->GetBuff();
  }
// 每个线程一个bufferpool，bufferpool为环状存储
  void PutBuff(int32_t thread_id, AbstractAppendOnlyBuffer *buff) {
    int32_t idx = thread_id - head_thread_id_;
    buffer_pool_vec_[idx]->PutBuff(buff);
  }

private:
  const int32_t head_thread_id_;
  std::vector<BufferPool*> buffer_pool_vec_;
};

}
