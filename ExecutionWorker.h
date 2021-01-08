#ifndef EXECUTIONWORKER_H
#define EXECUTIONWORKER_H

#include <vector>
#include <thread>
#include <functional>
#include "vedas.h"
#include "concurrentqueue.h"

class ExecutionWorker {
public:
  ExecutionWorker();
  ExecutionWorker(int gpu_num);

  bool pushTask(size_t threadNo, std::function<int(void)> f);
private:
  int gpu_num;
  std::vector<std::thread> threads;
  std::vector<moodycamel::ConcurrentQueue<std::function<int(void)>>> work_queues;

  void initialize(int gpu_num);
};

#endif
