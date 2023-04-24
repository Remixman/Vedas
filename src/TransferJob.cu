#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include "vedas.h"
#include "QueryExecutor.h"
#include "TransferJob.h"
#include "FullRelationIR.h"

using namespace std;

TransferJob::TransferJob(size_t src_thread_id, size_t dest_thread_id, QueryJob *job) {
    this->src_thread_id = src_thread_id;
    this->dest_thread_id = dest_thread_id;
    this->job = job;
}

TransferJob::~TransferJob() {

}

IR* TransferJob::getIR() {
    return this->intermediateResult;
}

int TransferJob::startJob() {
    FullRelationIR *ir = nullptr;
    // while (true) {
        intermediateResult = job->getIR();
        ir = dynamic_cast<FullRelationIR*>(intermediateResult);
        // if (ir != nullptr) break;
        assert(ir != nullptr);
    // }
    std::cout << "Start transfer job thread " << src_thread_id << " to thread "
        << dest_thread_id << "(d " << gpu_ids[src_thread_id] << " to "
        << " d " << gpu_ids[dest_thread_id] << ")\n";
    ir->movePeer(gpu_ids[src_thread_id], gpu_ids[dest_thread_id]);

    return 0;
}

void TransferJob::print() const {
  std::cout << "\tTRANSFER JOB\n";
}

std::string TransferJob::jobTypeName() const {
    return "Transfer Job";
}


void TransferJob::setGpuIds(const std::vector<int>& gpu_ids) {
    this->gpu_ids = gpu_ids;
}
