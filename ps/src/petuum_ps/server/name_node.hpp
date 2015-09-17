#pragma once

#include <petuum_ps/server/name_node_thread.hpp>
#include <pthread.h>
// Namenode 是将name_node_thread 做了二次封装
// 封装上了瘾，还是封装效果就是好呢？
// 为了提供更加简洁的接口吧

namespace petuum {

class NameNode {
public:
  static void Init();
  static void ShutDown();
private:
  static NameNodeThread *name_node_thread_;
  static pthread_barrier_t init_barrier_;
};

}
