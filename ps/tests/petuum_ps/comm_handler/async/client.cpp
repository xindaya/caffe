#include <boost/shared_array.hpp>
#include <boost/program_options.hpp>
namespace boost_po = boost::program_options;

#include <string>
#include <comm_handler.hpp>
#include <stdio.h>
#include<glog/logging.h>

using namespace petuum;

int main(int argc, char *argv[]){
  boost_po::options_description options("Allowed options");
  std::string sip;
  std::string sport;
  int32_t id;
  int32_t sid;

  options.add_options()
    ("id", boost_po::value<int32_t>(&id)->default_value(1), "node id")
    ("sid", boost_po::value<int32_t>(&sid)->default_value(0), "scheduler id")
    ("sip", boost_po::value<std::string>(&sip)->default_value("127.0.0.1"), "ip address")
    ("sport", boost_po::value<std::string>(&sport)->default_value("9999"), "port number");
    
  
  boost_po::variables_map options_map;
  boost_po::store(boost_po::parse_command_line(argc, argv, options), options_map);
  boost_po::notify(options_map);  
  google::InitGoogleLogging(argv[0]);
  ConfigParam config(id, false, "", "");
  
  CommHandler *comm;
  try{
    comm = new CommHandler(config);
  }catch(...){
    LOG(ERROR) << "failed to create comm";
    return -1;
  }
  zmq::context_t zmq_ctx(1);
  int suc = comm->Init(&zmq_ctx);

  if(suc == 0) LOG(INFO) << "comm_handler init succeeded";
  else{
    LOG(ERROR) << "comm_handler init failed";
    return -1;
  }

  suc = comm->ConnectTo(sip, sport, sid);
  if(suc < 0) LOG(ERROR) << "failed to connect to server";

  boost::shared_array<uint8_t> data;
  int32_t rid;

  suc = comm->Recv(rid, data);
  assert(suc > 0 && rid == sid);

  printf("Received msg : %d from %d\n", *((int32_t *) data.get()), sid);
  
  sleep(5);
  suc = comm->Send(sid, (uint8_t *) "hello", 6);
  assert(suc > 0);

  LOG(INFO) << "TEST NEARLY PASSED!! SHUTTING DOWN COMM THREAD!!";
  suc = comm->ShutDown();
  if(suc < 0) LOG(ERROR) << "failed to shut down comm handler";
  delete comm;
  LOG(INFO) << "TEST PASSED!! EXITING!!";
  return 0;
}
