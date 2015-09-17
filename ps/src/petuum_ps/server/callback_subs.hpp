#pragma once
#include <stdint.h>
#include <bitset>

// record_buff 做过注释了
// 简单的说：
// buffer就是一个record_id,record_size,mem <repeat>这样的内存组织方式
// ----


#include <petuum_ps_common/util/record_buff.hpp>

#include <petuum_ps_common/util/stats.hpp>

#include <petuum_ps/thread/context.hpp>
#include <glog/logging.h>

// 这里设置最大的client数目是什么意思呢？
// 难道petuum只支持8个client？

#ifndef PETUUM_MAX_NUM_CLIENTS
#define PETUUM_MAX_NUM_CLIENTS 8
#endif

namespace petuum {

class CallBackSubs {
public:

  //只用了默认的构造和解析
  CallBackSubs() { }
  ~CallBackSubs() { }
/*
 *
 * 居然用bitset来做注册机制
 * 所以序列号要连续？
 *
 * */


      // 订阅
  bool Subscribe(int32_t client_id) {
    bool bit_changed = false;
    if (!subscriptions_.test(client_id)) {
      bit_changed = true;
      subscriptions_.set(client_id);
    }
    return bit_changed;
  }

      // 取消订阅
  bool Unsubscribe(int32_t client_id) {
    bool bit_changed = false;
    if (subscriptions_.test(client_id)) {
      bit_changed = true;
      subscriptions_.reset(client_id);
    }
    return bit_changed;
  }


  bool AppendRowToBuffs(
      int32_t client_id_st,
      boost::unordered_map<int32_t, RecordBuff> *buffs,
      const void *row_data,
      size_t row_size,
      int32_t row_id,
      int32_t *failed_client_id,
      size_t *num_clients) {

    // Some simple tests show that iterating bitset isn't too bad.
    // For bitset size below 512, it takes 200~300 ns on an Intel i5 CPU.

    /*
     * 上面的注释的意思是说，使用bitset做遍历，不是一个坏主意
     * 小于512的bitset，只用200到300ns
     * */
    for (int32_t client_id = client_id_st;
         client_id < GlobalContext::get_num_clients(); ++client_id) {
      if (subscriptions_.test(client_id)) {
        bool suc = (*buffs)[client_id].Append(row_id, row_data, row_size);
        if (!suc) {
          *failed_client_id = client_id;
          return false;
        }
        ++(*num_clients);
      }
    }
    // 统计很重要
    // 要是可视化就更好了，哇咔咔

    STATS_SERVER_ADD_PER_CLOCK_ACCUM_DUP_ROWS_SENT(*num_clients);
    return true;
  }

      /*
       * 计算每个client序列化的size
       *
       * */
  void AccumSerializedSizePerClient(
      boost::unordered_map<int32_t, size_t> *client_size_map,
      size_t serialized_size) {
    int32_t client_id;
    for (client_id = 0;
         client_id < GlobalContext::get_num_clients(); ++client_id) {
      if (subscriptions_.test(client_id)) {
        (*client_size_map)[client_id] += serialized_size + sizeof(int32_t)
                                         + sizeof(size_t);
      }
    }
  }

  void AppendRowToBuffs(
          // https://leonax.net/p/3151/unordered_map-in-cpp0x/
          // unordered_map 原来是hash map啊
          // 我也是糊涂了，竟然忘了
      boost::unordered_map<int32_t, RecordBuff> *buffs,
      const void *row_data, size_t row_size, int32_t row_id,
      size_t *num_clients) {
    // Some simple tests show that iterating bitset isn't too bad.
    // For bitset size below 512, it takes 200~300 ns on an Intel i5 CPU.
    int32_t client_id;
    for (client_id = 0;
         client_id < GlobalContext::get_num_clients(); ++client_id) {
      if (subscriptions_.test(client_id)) {
        bool suc = (*buffs)[client_id].Append(row_id, row_data, row_size);
        if (!suc) {
          (*buffs)[client_id].PrintInfo();
          LOG(FATAL) << "should never happen";
        } else
          (*num_clients)++;
      }
    }
    STATS_SERVER_ADD_PER_CLOCK_ACCUM_DUP_ROWS_SENT(*num_clients);
  }

private:

      // 用bitset做注册？
      // 这里的意思是，8位的位集合
  std::bitset<PETUUM_MAX_NUM_CLIENTS> subscriptions_;
};

}  //namespace petuum
