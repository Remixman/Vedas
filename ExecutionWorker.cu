#include <iostream>
#include <mutex>
#include "ExecutionWorker.h"
#include "concurrentqueue.h"

ExecutionWorker::ExecutionWorker() {
    int gpu_num;
    cudaGetDeviceCount(&gpu_num);
    this->initialize(gpu_num);
}

ExecutionWorker::ExecutionWorker(int gpu_num) {
    this->initialize(gpu_num);
}

void ExecutionWorker::initialize(int gpu_num) {
    this->gpu_num = gpu_num;

    // Construct work queue
    for (int thread_no = 0; thread_no < gpu_num; thread_no++) {
        work_queues.push_back(moodycamel::ConcurrentQueue<std::function<int(void)>>());
    }

    for (int thread_no = 0; thread_no < gpu_num; thread_no++) {
        size_t device_no = thread_no;
        cudaSetDevice(device_no);

        // Check that device can directly send data to other device
        for (size_t other_id = 0; other_id < gpu_num; other_id++) {
            if (device_no != other_id) {
                int canAccessPeer = 0;
                cudaDeviceCanAccessPeer(&canAccessPeer, device_no, other_id);
                if (canAccessPeer) {
                    cudaDeviceEnablePeerAccess(other_id, 0);
                } else {
                    std::cerr << "P2P from " << device_no << " to " << other_id << "is not accessable\n";
                }
            }
        }

        // Initialize worker thread task
        threads.push_back(std::thread([this, thread_no]() {
            cudaSetDevice(thread_no);

            while(true) {
                // wait task in queue
                bool found_task = false;
                std::function<int(void)> task;
                do {
                    bool found_task = this->work_queues[thread_no].try_dequeue(task);
                } while (!found_task);

                // perform the task
                int result = task();
                if (result < 0) return; // exit thread
            }
        }));
    }
}

bool ExecutionWorker::pushTask(size_t threadNo, std::function<int(void)> f) {
    work_queues[threadNo].enqueue(f);
    return true;
}
