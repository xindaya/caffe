#pragma once

#include <boost/noncopyable.hpp>
#include <stdint.h>

//#
// 这里使用了boost的noncopyable 类型，说明该类是一个单例对象
//
//  这里应该要实现的东西，是对所有的存储的类型的一个操作的抽象
//  get_key
//  is_end
// 定义了抽象迭代类，所谓Iterator就是不断更新迭代delta w
// #
namespace petuum {
template<typename V>
class AbstractIterator : boost::noncopyable {
public:
  AbstractIterator() { }
  virtual ~AbstractIterator() { }

  virtual int32_t get_key() = 0;
  virtual V & operator *() = 0;
  virtual void operator ++ () = 0;
  virtual bool is_end() = 0;
};

template<typename V>
class AbstractConstIterator : boost::noncopyable {
public:
  AbstractConstIterator() { }
  virtual ~AbstractConstIterator() { }

  virtual int32_t get_key() = 0;
  virtual V operator *() = 0;
  virtual void operator ++ () = 0;
  virtual bool is_end() = 0;
};

}
