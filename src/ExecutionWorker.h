#ifndef EXECUTIONWORKER_H
#define EXECUTIONWORKER_H

#include <vector>
#include <future>
#include <thread>
#include <functional>
#include "vedas.h"
#include "concurrentqueue.h"

class ExecutionWorker {
public:
  ExecutionWorker(const std::vector<int>& gpu_ids);
  ~ExecutionWorker();

  bool pushTask(size_t threadNo, std::function<int(void)> f);
  std::vector<int> getGpuIds() const;
  size_t size() const;

  int sendData(int from, int to, int data);
  int receiveData(int from, int to);

private:
  int gpu_num;
  std::vector<int> gpu_ids;
  std::vector<std::thread> threads;
  std::vector<std::vector<std::promise<int>>> promises;
  std::vector<std::vector<std::future<int>>> futures;
  std::vector<moodycamel::ConcurrentQueue<std::function<int(void)>>> work_queues;
  int waitAll();
  // int wait(size_t thread_no);

  void initialize(int gpu_num);
};

#endif
