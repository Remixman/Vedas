#include <iostream>
#include <mutex>
#include "ExecutionWorker.h"

ExecutionWorker::ExecutionWorker(std::vector<int>& gpu_ids) {
    this->gpu_ids = gpu_ids;
    this->initialize(gpu_ids.size());
}

ExecutionWorker::~ExecutionWorker() {
    delete this->pool;
}

void ExecutionWorker::initialize(int gpu_num) {
    this->gpu_num = gpu_num;
    this->pool = new BS::thread_pool(gpu_num);

    // Test send data between GPU
    for (int d = 0; d < gpu_num; ++d) {
        size_t device_no = gpu_ids[d];
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
    }
}

BS::thread_pool *ExecutionWorker::getPool() {
    return this->pool;
}

std::vector<int> ExecutionWorker::getGpuIds() const {
    return this->gpu_ids;
}

int ExecutionWorker::getGpuId(int i) const {
    return this->gpu_ids[i];
}

size_t ExecutionWorker::size() const {
    return this->gpu_num;
}
