/*
 * abstract_table_group.hpp
 * author: jinliang
 */

#pragma once

#include <map>
#include <cstdint>

#include <petuum_ps_common/include/configs.hpp>
#include <petuum_ps_common/include/table.hpp>
#include <petuum_ps_common/include/abstract_row.hpp>
#include <petuum_ps_common/util/class_register.hpp>

#include <petuum_ps_common/client/abstract_client_table.hpp>

#include <boost/noncopyable.hpp>
//����������������Ƕ�table�Ĺ���
namespace petuum {

class AbstractTableGroup {
public:
  AbstractTableGroup() { }

  virtual ~AbstractTableGroup() { }

// �������ʱ�򣬻�Ҫ�Լ���������id?
// ̫�����Ի���
// #
  virtual bool CreateTable(int32_t table_id,
                           const ClientTableConfig& table_config) = 0;

  virtual void CreateTableDone() = 0;

// table��Ҫע��thread����������ˣ�������ô��ģ�
  virtual void WaitThreadRegister() = 0;

  virtual AbstractClientTable *GetTableOrDie(int32_t table_id) = 0;

  virtual int32_t RegisterThread() = 0;

  virtual void DeregisterThread() = 0;

  virtual void Clock() = 0;

  virtual void GlobalBarrier() = 0;

  virtual void TurnOnEarlyComm() = 0;

  virtual void TurnOffEarlyComm() = 0;

};

}   // namespace petuum
