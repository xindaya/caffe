// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.02.14

#pragma once

//#
// 是一个接口，可否加锁，可否mt
// #
namespace petuum {

// The Lockable concept (implemented as interface/abstract class) describes
// the characteristics of types that provide exclusive blocking semantics for
// execution agents (i.e. threads).
class Lockable {
public:
  // Blocks until a lock can be obtained for the current execution agent. If
  // an exception is thrown, no lock is obtained.

// 阻塞，除非当前的正在执行的agent可以获取锁
  virtual void lock() = 0;

  // Releases the lock held by the execution agent. Throws no exceptions.
  // requires: The current execution agent should hold the lock.

  // 正在执行的进程释放锁
  virtual void unlock() = 0;

  // Attempts to acquire the lock for the current execution agent without
  // blocking. If an exception is thrown, no lock is obtained.  Return true if
  // the lock was acquired, false otherwise
  // 尝试获取锁
  virtual bool try_lock() = 0;

  // 感觉这些api都是从哪个标准库里搬过来的
};

}   // namespace petuum
