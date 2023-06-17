#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include "vedas.h"
#include "QueryExecutor.h"
#include "TransferJob.h"
#include "FullRelationIR.h"

using namespace std;

TransferJob::TransferJob(size_t src_thread_id, size_t dest_thread_id, int src_gpu_id, int dest_gpu_id, QueryJob *job) {
    this->src_thread_id = src_thread_id;
    this->dest_thread_id = dest_thread_id;
    this->src_gpu_id = src_gpu_id;
    this->dest_gpu_id = dest_gpu_id;
    this->job = job;
}

TransferJob::~TransferJob() {

}

IR* TransferJob::getIR() {
    return this->intermediateResult;
}

int TransferJob::startJob(int gpuId) {
    this->gpuId = gpuId;
    FullRelationIR *ir = dynamic_cast<FullRelationIR*>(job->getIR());
    assert(ir != nullptr);
    // std::cout << "Start transfer job thread " << src_thread_id << " to thread "
       // << dest_thread_id << "(d " << src_gpu_id << " to " << " d " << dest_gpu_id << ")\n";
    // std::cout << "Transfer size : " << ir->size() << "\n";
    ir->movePeer(src_gpu_id, dest_gpu_id);
    this->intermediateResult = ir;

    return 0;
}

void TransferJob::print() const {
  std::cout << "\tTRANSFER JOB\n";
}

std::string TransferJob::jobTypeName() const {
    return "Transfer Job";
}
