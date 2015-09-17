#include <petuum_ps_common/util/vector_clock_mt.hpp>
#include <vector>

namespace petuum {

VectorClockMT::VectorClockMT():
  VectorClock(){}

VectorClockMT::VectorClockMT(const std::vector<int32_t>& ids) :
  VectorClock(ids) { }
// 线程安全的代价就是要在代码中加入锁
    // 同志们，用锁可是一个技巧活
    // 用不好就搞的死锁了


void VectorClockMT::AddClock(int32_t id, int32_t clock) {

  // 这里用到了写锁
  // 就是write_lock
  std::unique_lock<SharedMutex> write_lock(mutex_);
  VectorClock::AddClock(id, clock);
}

    //这个动作也是需要修改变量，所以也得加锁
    // 但是thread id 不是唯一的吗？
    // 是不是这里其实不用加锁的
int32_t VectorClockMT::Tick(int32_t id) {
  std::unique_lock<SharedMutex> write_lock(mutex_);
  return VectorClock::Tick(id);
}

// 又有锁了
int32_t VectorClockMT::TickUntil(int32_t id, int32_t clock) {
  std::unique_lock<SharedMutex> write_lock(mutex_);
  return VectorClock::TickUntil(id, clock);
}

// read_lock 读锁
    // 读锁的开销是很小的
int32_t VectorClockMT::get_clock(int32_t id) const {
  std::unique_lock<SharedMutex> read_lock(mutex_);
  return VectorClock::get_clock(id);
}

// 又见读锁
    // 读锁可以共享
int32_t VectorClockMT::get_min_clock() const {
  std::unique_lock<SharedMutex> read_lock(mutex_);
  return VectorClock::get_min_clock();
}

}  // namespace petuum
