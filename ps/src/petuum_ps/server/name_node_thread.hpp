// name_node_thread.hpp
// author: jinliang

#pragma once

#include <vector>
#include <pthread.h>
#include <queue>

#include <petuum_ps_common/util/thread.hpp>
#include <petuum_ps/server/server.hpp>
#include <petuum_ps/thread/ps_msgs.hpp>
#include <petuum_ps_common/comm_bus/comm_bus.hpp>

namespace petuum {

class NameNodeThread : public Thread{
public:
  NameNodeThread(pthread_barrier_t *init_barrier);
  ~NameNodeThread() { }

  virtual void *operator() ();

  virtual void ShutDown() {
    Join();
  }

private:
  // 这个结构体就是存放的存储创建table需要的2pc类似的模型需要的数据结构
  // 1. 多少个server有回应
  // 2. 多少个client有回应
  struct CreateTableInfo {
    int32_t num_clients_replied_;
    int32_t num_servers_replied_;
    // 这个队列是为了做什么呢
    // 需要告知每个client，我们创建了table吗？
    std::queue<int32_t> bgs_to_reply_;
    CreateTableInfo():
      num_clients_replied_(0),
      num_servers_replied_(0),
      bgs_to_reply_(){}

    ~CreateTableInfo(){}

    // 复制拷贝
    CreateTableInfo & operator= (const CreateTableInfo& info_obj){
      num_clients_replied_ = info_obj.num_clients_replied_;
      num_servers_replied_ = info_obj.num_servers_replied_;
      bgs_to_reply_ = info_obj.bgs_to_reply_;
      return *this;
    }

    // 判断，是否收到了所有server的回复信息
    bool ReceivedFromAllServers() const {
      return (num_servers_replied_ == GlobalContext::get_num_total_servers());
    }

    // 判断，是否收到了多有client的回复信息
    bool RepliedToAllClients() const {
      return (num_clients_replied_ == GlobalContext::get_num_clients());
    }
  };

  // communication function
  int32_t GetConnection(bool *is_client, int32_t *client_id);
  void SendToAllServers(MsgBase *msg);
  void SendToAllBgThreads(MsgBase *msg);

  // 构建nn的上下文环境
  void SetUpNameNodeContext();

  //构建一下通信
  void SetUpCommBus();

  // 启动nn
  void InitNameNode();

  // 是否创建了所有的table
  bool HaveCreatedAllTables();

  // 是否发送了消息
  void SendCreatedAllTablesMsg();

  // 自动机模型，事件驱动
  bool HandleShutDownMsg(); // returns true if the server may shut down
  void HandleCreateTable(int32_t sender_id, CreateTableMsg &create_table_msg);
  void HandleCreateTableReply(CreateTableReplyMsg &create_table_reply_msg);


  // 最重要的结构体放在这里
  // my_id_
  int32_t my_id_;
  pthread_barrier_t *init_barrier_;
  CommBus *comm_bus_;

  //维护信息都放在这里

  // 用vector来存放所有的bg_worker的id
  std::vector<int32_t> bg_worker_ids_;
  // 有一个client thread 被认为是head bg
  // one bg per client is refered to as head bg
  // 存放所有的table信息
  std::map<int32_t, CreateTableInfo> create_table_map_;
  Server server_obj_;
  int32_t num_shutdown_bgs_;
};
}
