// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.10.16

#pragma once

#include <type_traits>
#include <cstdint>

//#这个is_pod是新的标准，不是很秦楚到底怎么用
//cpp也是越来越复杂了
// 不对
// cpp本来就是复杂的#
namespace petuum {

// An Entry is an ID (int32_t) value (V) pair.
//
// Comment(wdai): We cannot use std::pair since we use memcpy and memmove and
// std::pair isn't POD.
template<typename V>
struct Entry {
  static_assert(std::is_pod<V>::value, "V must be POD");
  int32_t first;
  V second;
};

}  // namespace petuum
