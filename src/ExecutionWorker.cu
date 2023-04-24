#include <iostream>
#include <mutex>
#include "ExecutionWorker.h"
#include "concurrentqueue.h"

ExecutionWorker::ExecutionWorker(const std::vector<int>& gpu_ids) {
    this->gpu_ids = gpu_ids;
    this->initialize(gpu_ids.size());
}

ExecutionWorker::~ExecutionWorker() {
    this->waitAll();
}

int ExecutionWorker::sendData(int from, int to, int data) {
    promises[from][to].set_value(data);
    return 0;
}

int ExecutionWorker::receiveData(int from, int to) {
    futures[from][to].get();
    return 0;
}

void ExecutionWorker::initialize(int gpu_num) {
    this->gpu_num = gpu_num;

    futures.resize(gpu_num + 1);
    for (int t = 0; t < gpu_num + 1; t++) {
        promises.push_back(std::vector<std::promise<int>>(gpu_num + 1));

        for (int i = 0; i < gpu_num + 1; i++) {
            futures[t].push_back(promises[t][i].get_future());
        }
    }

    // Construct work queue
    for (int thread_no = 0; thread_no < gpu_num; thread_no++) {
        work_queues.push_back(moodycamel::ConcurrentQueue<std::function<int(void)>>());
    }

    for (int thread_no = 0; thread_no < gpu_num; thread_no++) {
        size_t device_no = gpu_ids[thread_no];
        cudaSetDevice(device_no);

        // Check that device can directly send data to other device
        for (size_t other_idx = 0; other_idx < gpu_num; other_idx++) {
            int other_id = gpu_ids[other_idx];
            if (device_no != other_id) {
                int canAccessPeer = 0;
                cudaDeviceCanAccessPeer(&canAccessPeer, device_no, other_id);
                if (canAccessPeer) {
                    cudaDeviceEnablePeerAccess(other_id, device_no);
                } else {
                    std::cerr << "P2P from " << device_no << " to " << other_id << "is not accessable\n";
                }
            }
        }

        // Initialize worker thread task
        threads.push_back(std::thread([this, thread_no]() {
            size_t device_no = gpu_ids[thread_no];
            cudaSetDevice(device_no);

            while(true) {
                // wait task in queue
                bool found_task = false;
                std::function<int(void)> task;
                do {
                    found_task = this->work_queues[thread_no].try_dequeue(task);
                } while (!found_task);

                // perform the task
                std::cout << "\tThread: " << thread_no << " start task()\n";
                int result = task();
            }
        }));
    }
}

bool ExecutionWorker::pushTask(size_t threadNo, std::function<int(void)> f) {
    work_queues[threadNo].enqueue(f);
    return true;
}

std::vector<int> ExecutionWorker::getGpuIds() const {
    return this->gpu_ids;
}

size_t ExecutionWorker::size() const {
    return this->gpu_num;
}

int ExecutionWorker::waitAll() {
    for (int thread_no = 0; thread_no < gpu_num; thread_no++) {
        if (threads[thread_no].joinable()) {
            threads[thread_no].join();
            std::cout << "Thread [" << thread_no << "] join\n";
        }
    }

    return 0;
}

// int ExecutionWorker::wait(size_t thread_no) {
//     if (threads[thread_no].joinable()) {
//         threads[thread_no].join();
//         std::cout << "Thread [" << thread_no << "] join\n";
//     }

//     return 0;
// }
