#pragma once

#include <glog/logging.h>
#include <boost/unordered_map.hpp>
#include <vector>

// vector clock
// 翻译为汉语的意思是向量时钟，这个可以将好几十分钟

namespace petuum {

// VectorClock manages a vector of clocks and maintains the minimum of these
// clocks. This class is single thread (ST) only.

// 先把作者的原版的内容翻译一下
// vectorClock 管理了一个clock的向量，并且维护了这些向量中的最小
// 作者为了防止我们误解他的意思，特意好心的告诉我们，这个类是单线程的
// 难道我们还有用一堆向量时钟不成


class VectorClock {
public:
  VectorClock();
  // Initialize client_ids.size() client clocks with all of them at time 0.

  // 使用一个int 类型的向量初始化
  // 我有十足的把握，这个vector中间的int类型数据是thread id， 然后统统的设置为0
  explicit VectorClock(const std::vector<int32_t>& ids);

  // Add a clock in vector clock with initial timestampe. id must be unique.
  // Return 0 on success, negatives on error (e.g., duplicated id).

  // 随时加一个新的来，是没有问题的
  virtual void AddClock(int32_t id, int32_t clock = 0);

  // Increment client's clock. Accordingly update slowest_client_clock_.
  // Return the minimum clock if client_id is the slowest thread; 0 if not;
  // negatives for error;

  virtual int32_t Tick(int32_t id);
  virtual int32_t TickUntil(int32_t id, int32_t clock);

  // Getters
  // 读写，读读读
  virtual int32_t get_clock(int32_t id) const;
  virtual int32_t get_min_clock() const;

private:

  // 看了这么多代码，我发现了一个规律，可能是一个pattern吧，就是把核心的数据结构放到private，谁也看不到


  // 是否是唯一的最小，如果不是，那么就说明还有和你一样的存在，你只管增加自己的clock就好了，让其他人负责更新min吧


  // If the tick of this client will change the slowest_client_clock_, then
  // it returns true.


  bool IsUniqueMin(int32_t id);

  // 使用一个无序的map做内部的数据结构
  // 第一个参数是thread id
  // 第二个参数是 clock
  boost::unordered_map<int32_t, int32_t> vec_clock_;

  // Slowest client clock
  int32_t min_clock_;
};

}  // namespace petuum
