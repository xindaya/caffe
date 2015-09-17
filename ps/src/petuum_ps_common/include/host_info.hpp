#pragma once

#include <string>
#include <glog/logging.h>
// #HostInfo 这个结构体，就是讲localhost文件中的内容格式化，做了解析，便于分析#
namespace petuum {

struct HostInfo {
  int32_t id;
  std::string ip;
  std::string port;
  
  HostInfo() {}
  ~HostInfo() { }
  HostInfo(int32_t _id, std::string _ip, std::string _port):
    id(_id),
    ip(_ip),
    port(_port) {};

  HostInfo(const HostInfo &other):
    id(other.id),
    ip(other.ip),
    port(other.port) {}

  HostInfo & operator = (const HostInfo &other) {
    id = other.id;
    ip = other.ip;
    port = other.port;
    return *this;
  }
};

}
