#pragma once

#include <pthread.h>

// 难道封装上了瘾，怎么什么都封装了
// 连thread都封装了
// 仿照boost::thread 实现的？
// 还是直接把boost中的代码copy过来了
namespace petuum {
class Thread {
public:

  Thread() { }

  virtual ~Thread() { }

  // functor 仿函数，怎么要返回函数指针，如何用?
  virtual void *operator() () {
    return 0;
  }

  int Start() {
    InitWhenStart();
    // 这个调用是pthread 的api
    // 时而用boost::thread
    // 时而用pthread，作者果然不是一个人啊
    // 当然boost::thread 的内部也是使用的pthread的api 在linux上
    // 我很怀疑作者是为了炫技

    // http://blog.csdn.net/liangxanhai/article/details/7767430
    return pthread_create(&thr_, NULL, InternalStart, this);
  }

  void Join() {
    // 等待，等待，等待，重要的事情要说三遍
    pthread_join(thr_, NULL);
  }

protected:
  virtual void InitWhenStart() { }

private:
  //
  static void *InternalStart(void *thread) {
    Thread *t = reinterpret_cast<Thread*>(thread);
    return t->operator()();
  }

  pthread_t thr_;
};
}
