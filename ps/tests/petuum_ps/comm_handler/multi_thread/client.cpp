#include <boost/shared_array.hpp>
#include <boost/program_options.hpp>
namespace boost_po = boost::program_options;

#include <string>
#include <comm_handler.hpp>
#include <stdio.h>
#include <glog/logging.h>

using namespace petuum;

struct thrinfo{
  int num_sthrs;
  int thrid;
  CommHandler *comm;
};

void *thr_recv(void *tin){
  thrinfo *info = (thrinfo *) tin;
  int i;
  boost::shared_array<uint8_t> data;
  for(i = 0; i < info->num_sthrs; ++i){
      int32_t rid;
      int suc = info->comm->Recv(rid, data);
      assert(suc > 0);
      printf("Thread %d, received msg : %s from %d\n", info->thrid, (char *) data.get(), rid);
  }
  return NULL;
}

int main(int argc, char *argv[]){
  boost_po::options_description options("Allowed options");
  std::string sip;
  std::string sport;
  int32_t id;
  int32_t sid;
  int num_thrs;
  int num_sthrs;
  options.add_options()
    ("id", boost_po::value<int32_t>(&id)->default_value(1), "node id")
    ("sid", boost_po::value<int32_t>(&sid)->default_value(0), "scheduler id")
    ("sip", boost_po::value<std::string>(&sip)->default_value("127.0.0.1"), "ip address")
    ("sport", boost_po::value<std::string>(&sport)->default_value("9999"), "port number")
    ("nthr", boost_po::value<int>(&num_thrs)->default_value(2), "number of threads to use")
    ("nsthr", boost_po::value<int>(&num_sthrs)->default_value(2), "number of server threads");
  
  boost_po::variables_map options_map;
  boost_po::store(boost_po::parse_command_line(argc, argv, options), options_map);
  boost_po::notify(options_map);  
  
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

  if(suc == 0) LOG(INFO) << "comm_handler init succeeded" << std::endl;
  else{
    LOG(ERROR) << "comm_handler init failed" << std::endl;
    return -1;
  }

  suc = comm->ConnectTo(sip, sport, sid);
  if(suc < 0) LOG(ERROR) << "failed to connect to server";
  
  thrinfo *info = new thrinfo[num_thrs];
  pthread_t *thrs = new pthread_t[num_thrs];
  int i;
  for(i = 0; i < num_thrs; ++i){
    info[i].num_sthrs = num_sthrs;
    info[i].thrid = i;
    info[i].comm = comm;
  }
  LOG(INFO) << "start threads to receive!";

  for(i = 0; i < num_thrs; ++i){
    pthread_create(&(thrs[i]), NULL, thr_recv, info + i);
  }

  for(i = 0; i < num_thrs; ++i){
    pthread_join(thrs[i], NULL);
  }

  suc = comm->Send(sid, (uint8_t *) "hello", 6);
  assert(suc > 0);

  LOG(INFO) << "TEST NEARLY PASSED!! SHUTTING DOWN COMM THREAD!!";
  suc = comm->ShutDown();
  if(suc < 0) LOG(ERROR) << "failed to shut down comm handler";
  delete comm;
  LOG(ERROR) << "TEST PASSED!! EXITING!!";
  return 0;
}
