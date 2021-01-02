#ifndef EXECUTIONWORKER_H
#define EXECUTIONWORKER_H

#include <vector>
#include <thread>
#include "vedas.h"
#include "concurrentqueue.h"

class ExecutionWorker {
public:
  ExecutionWorker(size_t gpu_num);

  bool pushTask(size_t threadNo, function<int<void>> f);
private:
  size_t gpu_num;
  std::vector<std::thread> threads;
  std::vector<moodycamel::ConcurrentQueue<function<int(void)>>> work_queues;
}

#endif
