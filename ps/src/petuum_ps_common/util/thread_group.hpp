#pragma once

#include <vector>

namespace petuum {

template <typename ThreadType>
class ThreadGroup {
public:
  // 要是一起死啊，一起死啊
  virtual ~ThreadGroup() {
    for (auto &thread : threads_) {
      delete thread;
    }
  }

// 一起来启动，大家一起来启动
  void StartAll() {
    for (auto &thread : threads_) {
      thread->Start();
    }
  }

  // 每个thread都不许跑，都要乖乖的join
  void JoinAll() {
    for (auto &thread : threads_) {
      thread->Join();
    }
  }

protected:
    ThreadGroup(size_t num_threads):
      threads_(num_threads) { }

// 一看到vector就要明白，这个是pool
// 保留了一堆的thread
  std::vector<ThreadType*> threads_;
};
}
