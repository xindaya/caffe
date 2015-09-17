// Author: Aurick Qiao (aqiao@cs.cmu,edu), Jinliang
// Date: 2014.10.06

#pragma once

#include <cassert>
#include <cstring>
#include <condition_variable>
#include <mutex>

#include <memory>
#include <boost/noncopyable.hpp>

namespace petuum {

// MPMCQueue is a multi-producer-multi-consumer bounded buffer.

    // MPMCQueue 是多生产者多消费者
    // 1. 多生产者
    // 2. 多消费者
    // 3. 受限


// 这个类竟然是单例呢
template<typename T>
class MPMCQueue : boost::noncopyable {
public:
  MPMCQueue(size_t capacity)
  // 容量，容量就是容量啊
  : capacity_(capacity),
    size_(0),
          // buffer 原来是new 出来的，那么就是说它的数据结构式array
    buffer_(new T[capacity]) {
    // begin 就是head 指针啊，不是我讲的那个环形数据结构吗
    // 初始化，得到指针
    begin_ = buffer_.get();
    end_ = begin_;
  }

  ~MPMCQueue() { }

  size_t get_size() const {
    return size_;
  }

  //实现的是多生产者，多消费者，那么不加锁，不是互相乱了吗
  void Push(const T& value) {

    std::unique_lock<std::mutex> lock(mtx_);
    while (size_ == capacity_) cv_.wait(lock);
    memcpy(end_, &value, sizeof(T));
    end_++;
    size_++;
    if (end_ == buffer_.get() + capacity_) end_ = buffer_.get();
  }

  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mtx_);
// 如果空，就不出队了
    if (size_ == 0) return false;
    // 如果满了，出队，意味着有了空位，把阻塞的thread叫醒
    if (size_ == capacity_) cv_.notify_all();
    *value = *begin_;
    begin_++;
    size_--;
    // 看来不是环形数据结构
    // 如果begin 到了底，那么begin重新走到第一个
    // 虽然不是环形数据结构，但是看起来很像的
    if (begin_ == buffer_.get() + capacity_) {
      begin_ = buffer_.get();
    }
    return true;
  }

private:
  size_t capacity_, size_;
  // unique_ptr，保证数据不会被共享，所有权只有一个
  std::unique_ptr<T[]> buffer_;
  // 指针
  T *begin_, *end_;
  // 锁在这里
  std::mutex mtx_;
  // 有了条件变量，多线程写代码就容易了不少
  std::condition_variable cv_;
};

}   // namespace petuum
