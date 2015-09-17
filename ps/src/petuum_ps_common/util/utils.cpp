#include <petuum_ps_common/util/utils.hpp>

#include <utility>
#include <fstream>
#include <iostream>
#include <glog/logging.h>
#include <random>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace petuum {

    //从配置文件中解析出来host_info内容
    // 1. id
    // 2. ip
    // 3. port

void GetHostInfos(std::string server_file,
  std::map<int32_t, HostInfo> *host_map) {
  std::map<int32_t, HostInfo>& servers = *host_map;

  std::ifstream input(server_file.c_str());
  std::string line;
  while(std::getline(input, line)){

    size_t pos = line.find_first_of("\t ");
    std::string idstr = line.substr(0, pos);

    size_t pos_ip = line.find_first_of("\t ", pos + 1);
    std::string ip = line.substr(pos + 1, pos_ip - pos - 1);
    std::string port = line.substr(pos_ip + 1);

    int32_t id =  atoi(idstr.c_str());
    servers.insert(std::make_pair(id, petuum::HostInfo(id, ip, port)));
    //VLOG(0) << "get server: " << id << ":" << ip << ":" << port;
  }
  input.close();
}

// assuming the namenode id is 0

    // 从hostMap 中获取server的id信息
    // 让我们看看这个到底是怎么做到的

    // 参数是
    // 1. vector 的指针
    // 2. host_map 的引用
void GetServerIDsFromHostMap(std::vector<int32_t> *server_ids,
    const std::map<int32_t, HostInfo>& host_map){

    // server 为什么要减一
//     从后面的代码中，我们可以看到越过了id==0的host_info

  int32_t num_servers = host_map.size() - 1;
  server_ids->resize(num_servers);
  int32_t i = 0;

  for (auto host_info_iter = host_map.cbegin();
    host_info_iter != host_map.cend(); host_info_iter++) {
    if (host_info_iter->first == 0)
      continue;
    (*server_ids)[i] = host_info_iter->first;
    ++i;
  }
}


// 原来update 还要排序
// 是不是意味着，我们可以对delta w做过滤呢？


UpdateSortPolicy GetUpdateSortPolicy(const std::string &policy) {
  if (policy == "Random") {
    return Random;
  } else if (policy == "FIFO") {
    return FIFO;
  } else if (policy == "RelativeMagnitude") {
    return RelativeMagnitude;
  } else if (policy == "FIFO_N_RegMag") {
    return FIFO_N_ReMag;
  } else if (policy == "FixedOrder") {
    return FixedOrder;
  } else {
    LOG(FATAL) << "Unknown update sort policy: "
               << policy;
  }
  return Random;
}

// 虽然这里有三个ssp模型，但是好像也只有ssppush 可以用
//为什么其他的不能用》
ConsistencyModel GetConsistencyModel(const std::string &consistency_model) {
  if (consistency_model == "SSPPush") {
    return SSPPush;
  } else if (consistency_model == "SSP") {
    return SSP;
  } else if (consistency_model == "SSPAggr") {
    return SSPAggr;
  } else {
    LOG(FATAL) << "Unsupported ssp mode " << consistency_model;
  }
  return SSPPush;
}


// appendonly 是环形数组
// sparse
// dense
OpLogType GetOpLogType(const std::string &oplog_type) {
  if (oplog_type == "Sparse") {
    return Sparse;
  } else if (oplog_type == "AppendOnly"){
    return AppendOnly;
  } else if (oplog_type == "Dense") {
    return Dense;
  } else {
    LOG(FATAL) << "Unknown oplog type = " << oplog_type;
  }

  return Sparse;
}

AppendOnlyOpLogType GetAppendOnlyOpLogType(
    const std::string &append_only_oplog_type) {
    if (append_only_oplog_type == "Inc") {
      return Inc;
    } else if (append_only_oplog_type == "BatchInc") {
      return BatchInc;
    } else if (append_only_oplog_type == "DenseBatchInc") {
      return DenseBatchInc;
    } else {
      LOG(FATAL) << "Unknown append only oplog type = " << append_only_oplog_type;
    }
    return Inc;
}

// 进程级别的storage，这个代码我还没有看，需要看看然后再反过来处理这部分的代码

ProcessStorageType GetProcessStroageType(
    const std::string &process_storage_type) {
  if (process_storage_type == "BoundedDense") {
    return BoundedDense;
  } else if (process_storage_type == "BoundedSparse") {
    return BoundedSparse;
  } else {
    LOG(FATAL) << "Unknown process storage type " << process_storage_type;
  }
  return BoundedSparse;
}
  
float RestoreInf(float x) {
  if (isinf(x)) { 
    if (x > 0) {
      return FLT_MAX;
    } else {
      return -FLT_MAX;
    }
  } else {
    return x;
  }
}
  
float RestoreInfNaN(float x) {
  if (isinf(x)) { 
    if (x > 0) {
      return FLT_MAX;
    } else {
      return -FLT_MAX;
    }
  } else if (x != x) {
    return 0.01;
  } else {
    return x;
  }
}

}   // namespace petuum
