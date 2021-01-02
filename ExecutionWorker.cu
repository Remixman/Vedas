#include <iostream>
#include <mutex>
#include "ExecutionWorker.h"
// #include "concurrentqueue.h"

ExecutionWorker::ExecutionWorker(size_t gpu_num) {
  this.gpu_num = gpu_num;

  for (int thread_no = 0; thread_no < gpu_num; thread_no++) {
    threads.push_back([]() -> bool {
      while(true) {
        return true;
      }
    });

    work_queues.push_back(new moodycamel::ConcurrentQueue<function<int(int)>>());
  }
}

bool ExecutionWorker::pushTask(size_t threadNo, function<int<void>> f) {
  work_queues[threadNo].enqueue(f);
  return true;
}
