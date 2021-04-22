#include <chrono>
#include "QueryExecutor.h"
#include "SwapIndexJob.h"

SwapIndexJob::SwapIndexJob() {

}

IR* swap(FullRelationIR *ir, size_t swap_column) {
    auto swap_index_start = std::chrono::high_resolution_clock::now();
    if (swap_column != 0) {
        ir->swapColumn(0, swap_column);
        ir->sortByFirstColumn();
    }
    auto swap_index_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::swap_index_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(swap_index_end-swap_index_start).count();

    return ir;
}
