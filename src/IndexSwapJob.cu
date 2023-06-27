#include <algorithm>
#include <vector>
#include <chrono>
#include <cassert>
#include <string>
#include "QueryExecutor.h"
#include "IndexSwapJob.h"

IndexSwapJob::IndexSwapJob(QueryJob *beforeJob, std::string swapVar, 
                            mgpu::standard_context_t* context, \
                            std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound,
                            EmptyIntervalDict *ei_dict) {
    this->beforeJob = beforeJob;
    this->swapVar = swapVar;
    this->context = context;
    this->variables_bound = variables_bound;
    this->ei_dict = ei_dict;
}

IndexSwapJob::~IndexSwapJob() {

}

IR* IndexSwapJob::getIR() {
    return this->intermediateResult;
}

int IndexSwapJob::startJob(int gpuId) {
    this->gpuId = gpuId;
    auto swap_index_start = std::chrono::high_resolution_clock::now();
    
    FullRelationIR *beforeIr = dynamic_cast<FullRelationIR *>(beforeJob->getIR());
    size_t columnToSwap = beforeIr->getColumnId(swapVar);
    beforeIr->swapColumn(0, columnToSwap);
    beforeIr->sortByFirstColumn();
    
    std::string varToSwap = beforeIr->getHeader(0) + " <-> " + beforeIr->getHeader(columnToSwap);
    QueryExecutor::exe_log.push_back( ExecuteLogRecord(gpuId, SWAP_OP, varToSwap, beforeIr->size(), beforeIr->getColumnNum()) );
    
    auto swap_index_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::swap_index_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(swap_index_end-swap_index_start).count();
    
    // Update Bound
    // if (beforeIr->size() > 0) {
        // auto copy_start = std::chrono::high_resolution_clock::now();
        // TYPEID start_value = (*beforeIr->getRelation(0))[0];
        // TYPEID end_value = (*beforeIr->getRelation(0))[beforeIr->size()-1];
        // auto copy_end = std::chrono::high_resolution_clock::now();
        // QueryExecutor::updateBoundDict(variables_bound, varToSwap, start_value, end_value);
    // }

    // Update EIF
    if (beforeIr->size() > 2 && beforeIr->size() < EIF_ALPHA) {
        auto eif_start = std::chrono::high_resolution_clock::now();
        TYPEID_DEVICE_VEC diff_vec(beforeIr->size() - 1);
        thrust::transform(thrust::device,
            beforeIr->getRelation(0)->begin() + 1, beforeIr->getRelation(0)->end(), 
            beforeIr->getRelation(0)->begin(),
            diff_vec.begin(), thrust::minus<TYPEID>());
            auto max_iter = thrust::max_element(thrust::device, diff_vec.begin(), diff_vec.end());
        auto max_diff_pos = max_iter - diff_vec.begin();

        TYPEID v[6], b[4] = {0, 0};
        auto start_copy = (max_diff_pos - 2 >= 0) ? max_diff_pos - 2 : 0;
        auto end_copy = (max_diff_pos + 4 <= beforeIr->size()) ? max_diff_pos + 4 : beforeIr->size();
        auto size_copy = end_copy - start_copy;
        cudaMemcpy(v, beforeIr->getRelationRawPointer(0) + start_copy, size_copy * sizeof(TYPEID), cudaMemcpyDeviceToHost);

        size_t firstMax = 0, secondMax = 0;
        for (size_t i = 1; i < size_copy; i++) {
            auto interval_size = v[i] - v[i-1];
            if (interval_size >= firstMax) {
                secondMax = firstMax;
                b[0] = b[2], b[1] = b[3];
                firstMax = interval_size;
                b[2] = v[i-1], b[3] = v[i];
            } else if (interval_size > secondMax) {
                secondMax = interval_size;
                b[0] = v[i-1], b[1] = v[i];
            }
        }
        if (b[2] < b[0]) { std::swap(b[0], b[2]); std::swap(b[1], b[3]); }
        
        std::string var = beforeIr->getHeader(0);
        QueryExecutor::eif_count++;
        ei_dict->updateBound(var, b[0] + 1, b[1] - 1, b[2] + 1, b[3] - 1);

        auto eif_end = std::chrono::high_resolution_clock::now();
        QueryExecutor::eif_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(eif_end-eif_start).count();
    }
    
    this->intermediateResult = beforeIr;

    return 0;
}

std::string IndexSwapJob::jobTypeName() const {
    return "Index Swap Job";
}

void IndexSwapJob::print() const {
    std::cout << "\tINDEX SWAP JOB - swap column " << swapVar << "\n";
}