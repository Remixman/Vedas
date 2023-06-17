#ifndef EXECUTIONWORKER_H
#define EXECUTIONWORKER_H

#include <vector>
#include <future>
#include <thread>
#include <functional>
#include "vedas.h"
#include "BS_thread_pool.hpp"

class ExecutionWorker {
public:
  ExecutionWorker(std::vector<int>& gpu_ids);
  ~ExecutionWorker();

  BS::thread_pool *getPool();
  std::vector<int> getGpuIds() const;
  int getGpuId(int i) const;
  size_t size() const;
private:
  int gpu_num;
  std::vector<int> gpu_ids;
  BS::thread_pool *pool = nullptr;

  void initialize(int gpu_num);
};

#endif
