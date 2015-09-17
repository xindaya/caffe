// author: jinliang
// date: Feb 2, 2013

#pragma once

#include <stdint.h>
#include <glog/logging.h>
#include <boost/noncopyable.hpp>

namespace petuum {

/*
 * A thin layer to manage a chunk of contiguous memory which may be allocated
 * outside (via MemAlloc) or inside (via Alloc) the class.
 * This class is meant to be simple (cheap) so it does not check any assumption
 * that a function assumes when that function is invoked. If the assumption is
 * not satisfied the behavior of the current and future operation on MemBlock
 * is undefined.
 */

// 一个对连续存储的大块内存的一层薄薄的抽象
// 这个连续存储可能在外面做malloc，也可以在class内部处理
//
// 这个类本来就是要简单，所以没有做任何的检查，简直太简单了，如果有做二次开发，开来这个类是一个关注的重点
// #


//单例
class MemBlock : boost::noncopyable {
public:
// 默认初始化，赋值为0
  MemBlock():
    mem_(0) { }

  ~MemBlock(){
    Reset(0);
  }

  /*
   * Reset the MemBlock object to manage a particular memory chunk of certain
   * size. If there is a previous memory chunk managed by this object, it is
   * freed. Once a memory chunk is give to MemBlock, it cannot be freed
   * externally.
   */

  // 重置memblock对象，用来管理特定的内存， 参数是一个mem的地址
  //  如果这个memblock 类型有自己管理的内存，那么首先要free掉
  // 一旦一个内存交给了memblock来管理，那么，你不能从外面free掉这个类#
  void Reset(void *mem){
    if(mem_ != 0){
      MemFree(mem_);
    }
    // 将mem类型转化为了纯粹的内存地址
    mem_ = reinterpret_cast<uint8_t*>(mem);
  }

  /*
   * Release the control over the memory chunk without destroying it.
   */

  // 释放所有权，但是没有free
  // 注意 mem mem_两个变量是不同的
  uint8_t *Release(){
    uint8_t *mem = mem_;
    mem_ = 0;
    return mem;
  }

  /*
   * Get a pointer to access to the underlying memory managed by this MemBlock.
   */
  // 返回的是一个内存的地址，memblock 所管理的内存的地址
  uint8_t *get_mem(){
    return mem_;
  }

  /*
   * Allocate a chunk of memory based on the size information. Must be invoked
   * when there's no memory managed by this MemBlock object yet.
   */

  // 只有在memblock对象没有管理mem的时候，这个才会执行，可惜的是这个没有检查，谁知道有没有被人用错呢
  void Alloc(size_t size){
    mem_ = MemAlloc(size);
  }


// 使用的是new来申请的内存，这个内存放在heap上，记得释放啊
// delete
  static inline uint8_t *MemAlloc(size_t nbytes){
    uint8_t *mem = new uint8_t[nbytes];
    return mem;
  }

// mem free
  static inline void MemFree(uint8_t *mem){
    delete[] mem;
  }
// 最核心的内存地址躲在这里
private:
  uint8_t *mem_;

};

};
