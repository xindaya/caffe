#include <petuum_ps_common/util/vector_clock.hpp>

namespace petuum {
// 来分析一下不是线程安全的版本实现
// 没有加锁哦
    // 使用一个无序的map做内部的数据结构
    // 第一个参数是thread id
    // 第二个参数是 clock
    //boost::unordered_map<int32_t, int32_t> vec_clock_; 这段代码是从hpp文件中拷贝的
    // 我觉得hpp文件中不应该包含具体的变量定义
    // 对于读者来说这是非常麻烦的事情
    // 没错，我就是在吐槽
VectorClock::VectorClock():
  min_clock_(-1){}

    // 初始化，谁也跑不了，都剃成光头，值为0
VectorClock::VectorClock(const std::vector<int32_t>& ids):
  min_clock_(0) {
  int size = ids.size();
  for (int i = 0; i < size; ++i) {
    int32_t id = ids[i];

    // Initialize client clock with zeros.
    vec_clock_[id] = 0;
  }
}

    // 注意有一个修改min_clock的动作
void VectorClock::AddClock(int32_t id, int32_t clock)
{
  vec_clock_[id] = clock;
  // update slowest
  if (min_clock_ == -1 || clock < min_clock_) {
    min_clock_ = clock;
  }
}

// 下面的写法怪怪的
    // 如果自己是最小的，就连同min_clock_一起更新了
int32_t VectorClock::Tick(int32_t id)
{
  if (IsUniqueMin(id)) {
    ++vec_clock_[id];
    return ++min_clock_;
  }
  ++vec_clock_[id];
  return 0;
}

// 一直更新，直到到达了预定目标
int32_t VectorClock::TickUntil(int32_t id, int32_t clock) {
  int32_t curr_clock = VectorClock::get_clock(id);
  int32_t num_ticks = clock - curr_clock;
  int32_t new_clock = 0;

  for (int32_t i = 0; i < num_ticks; ++i) {
    int32_t clock_changed = VectorClock::Tick(id);
    if (clock_changed)
      new_clock = clock_changed;
  }

  return new_clock;
}

int32_t VectorClock::get_clock(int32_t id) const
{
  auto iter = vec_clock_.find(id);
  CHECK(iter != vec_clock_.end()) << "id = " << id;
  return iter->second;
}

int32_t VectorClock::get_min_clock() const
{
  return min_clock_;
}

// =========== Private Functions ============

    // 检查唯一性的方法就是遍历一遍数据，统计和min_clock相等的个数
bool VectorClock::IsUniqueMin(int32_t id)
{
  if (vec_clock_[id] != min_clock_) {
    // definitely not the slowest.
    return false;
  }
  // check if it is also unique
  int num_min = 0;
  for (auto iter = vec_clock_.cbegin(); iter != vec_clock_.cend(); iter++) {
    if (iter->second == min_clock_) ++num_min;
    if (num_min > 1) return false;
  }
  return true;
}

}  // namespace petuum
