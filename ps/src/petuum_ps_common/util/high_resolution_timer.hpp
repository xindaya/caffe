#pragma once

#include <time.h>
// 使用的c原生的api
namespace petuum {
    // 简单的说，这个是自己实现的计时器，分辨率的意思，就是可以统计非常小的单位，微妙够不够用呢？

// This is a simpler implementation of timer to replace
// boost::high_resolution_timer. Code based on
// http://boost.2283326.n4.nabble.com/boost-shared-mutex-performance-td2659061.html
class HighResolutionTimer {
  public:
  HighResolutionTimer();

  void restart();

  // return elapsed time (including previous restart-->pause time) in seconds.
  double elapsed() const;

  // return estimated maximum value for elapsed()
  double elapsed_max() const;

  // return minimum value for elapsed()
  double elapsed_min() const;

 private:
  double total_time_;
  struct timespec start_time_;
};

} // namespace petuum
