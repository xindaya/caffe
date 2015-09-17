// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.01.31

#pragma once

#include <queue>
#include <mutex>
// #
// MT=multi-thread
// 有MT的地方就有锁
// 因为使用锁是最简单的实现多线程的办法
// 如果不用锁，实现一个无锁的数据结构实在是太难了
// #

namespace petuum {

// Wrap around std::queue and provide thread safety (MT = multi-threaded).

    // 简单的说，就是一个加了锁的队列，无他
template<typename T>
class MTQueue {
  public:
    MTQueue() { }

    // Return 0 if it's empty, 1 if not. val is unset if returning 0. Note the
    // different semantics than std::queue::pop.
    int pop(T* val) {
      // c++ 中还真有不少的锁法宝啊
      std::lock_guard<std::mutex> lock(mutex_);
      if (!q_.empty()) {
        *val = q_.front();
        q_.pop();
        return 1;
      }
      return 0;
    }

// 因为push和pop都是对核心数据结构的操作，所以都要加锁
    void push(const T& val) {
      std::lock_guard<std::mutex> lock(mutex_);
      q_.push(val);
    }


  // 最核心的数据结构就藏在这里的
  // 一个队列
  // 一个锁
  // 换句话说就是有锁的队列
  private:
    std::queue<T> q_;
    std::mutex mutex_;
};


}   // namespace petuum
