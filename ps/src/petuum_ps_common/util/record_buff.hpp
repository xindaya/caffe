#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <boost/noncopyable.hpp>
#include <glog/logging.h>

namespace petuum {

// A buffer that allows appending records to it.
class RecordBuff : boost::noncopyable {
public:
  // 看来有换了一个作者，因为可以使用explicate关键字来明确要调用的构造函数
  RecordBuff() {
    LOG(FATAL) << "Default constructor called";
  }

  // 将内存封装成了buffer
  // 需要的元数据
  // 1. 指针
  // 2. size 也就是大小
  // 3. offset 也就是偏置
  RecordBuff(void *mem, size_t size):
      mem_(reinterpret_cast<uint8_t*>(mem)),
      mem_size_(size),
      offset_(0) { }
  ~RecordBuff() { }

// 拷贝复制函数
  RecordBuff(RecordBuff && other):
      mem_(other.mem_),
      mem_size_(other.mem_size_),
      offset_(other.offset_) {
    //VLOG(0) << "mem_size_ = " << mem_size_;
  };

  // does not take ownership of the memory

  // 返回了原来的指针
  // 用新的指针覆盖了原来的指针
  void *ResetMem(void *mem, size_t size) {
    void *old_mem = mem_;
    mem_ = reinterpret_cast<uint8_t*>(mem);
    mem_size_ = size;
    offset_ = 0;
    return old_mem;
  }

  void ResetOffset() {
    offset_ = 0;
  }

  //#
  // 这个是一个关键的函数，append，将数据加到队尾
  // #
  bool Append(int32_t record_id, const void *record, size_t record_size) {
    // 首先看看能不能放得下这个新的buff
    if (offset_ + sizeof(int32_t) + record_size + sizeof(size_t) > mem_size_) {
      return false;
    }

//buffer就是一个record_id,record_size,mem <repeat>这样的内存组织方式
    // 其实如果一个项目如果在这个地方花费时间的话，确实不妥
    //精力是宝贵的
    //-------------------------------------------

    //将record_id 存下来
    *(reinterpret_cast<int32_t*>(mem_ + offset_)) = record_id;
    // 因为存了record_id ,所以偏移量指针需要后移
    offset_ += sizeof(int32_t);
    // 然后存储具体的数据大小
    *(reinterpret_cast<size_t*>(mem_ + offset_)) = record_size;
    //毫不例外，可以往后移动指针了
    offset_ += sizeof(size_t);
    // 把数据拷贝到buff中，用的最原始的方式
    memcpy(mem_ + offset_, record, record_size);
    // 修改指针
    offset_ += record_size;
    //VLOG(0) << "Append() end offset = " << offset_;

    // 是挺麻烦的，但是用计算机来做就容易多了
    return true;
  }

  size_t GetMemUsedSize() {
    return offset_;
  }

  int32_t *GetMemPtrInt32() {

    // 这里的条件还是很重要的，避免越界
    if (offset_ + sizeof(int32_t) > mem_size_) {
      return 0;
    }
    int32_t *ret_ptr = reinterpret_cast<int32_t*>(mem_ + offset_);

    // 这里的offset为什么要加这个sizeof的结果？？
    offset_ += sizeof(int32_t);

    return ret_ptr;
  }

  void PrintInfo() const {
    VLOG(0) << "mem_ = " << mem_
            << " mem_size_ = " << mem_size_
            << " offset = " << offset_;
  }

//这里存储了对buffer来说最有用的数据
      // 1. mem指针
      //2. size
      // 3. 偏置
private:
  uint8_t *mem_;
  size_t mem_size_;
  size_t offset_;
};


}  // namespace petuum
