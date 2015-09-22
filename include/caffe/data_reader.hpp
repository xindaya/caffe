#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */

// 这一步实现的是 从硬盘上读数据到内存里
// source -> queue

/*
 * 这个解释很详细
 * 将数据从source中读出来,放到queue中
 *
 * 1. 单线程读取数据, 即使有多个solver在running
 * 2. 这样可以保证数据读取的序列化
 * 3. data 对于多个solver来说是分布式的, 因为有循环的队列, 来保证读取的确定性.
 * */

class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();
/*
 * 注意这个函数的目的是获取名字叫free的队列
 * 不是free队列
 * */
  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }

/*
 * 这个目的是获取full的名字队列
 * */
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
/*
 * 从命名的方式来看, 这里用的技术是双buffer, 来避免io阻塞
 * 具体双buffer是怎么做的,大家可以google一下,这个算是计算机程序设计中的一个trick
 * */
   public:
      // 一对queue
      // 命名很形象
    explicit QueuePair(int size);
    ~QueuePair();
    // 这个blockingqueue内部存储的类型是Datum的指针
    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
// body 是thread 的子类, 看来是可以跑起来的哦
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
      // 只要实现了IneternalThreadEntry() 就可以启动
    void InternalThreadEntry();
      // 读一个datum的意思
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;

    // queuepair 本身就是blockingqueue
    // new_queue_pairs_ 又是一个blockingqueue,好复杂
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    // 我可以访问dataReader的私有变量
    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<:DataReader:Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
