#include <algorithm>
#include <vector>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <utility>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_bulkremove.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/memory.hxx>
#include "QueryExecutor.h"
#include "JoinQueryJob.h"

JoinQueryJob::JoinQueryJob(QueryJob *leftJob, QueryJob *rightJob, std::string joinVariable, mgpu::standard_context_t* context,
                           std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound, 
                           EmptyIntervalDict *ei_dict, bool lastJoinForVar) {
    this->leftJob = leftJob;
    this->rightJob = rightJob;
    this->joinVariable = joinVariable;
    this->context = context;
    this->variables_bound = variables_bound;
    this->ei_dict = ei_dict;
}

JoinQueryJob::~JoinQueryJob() {

}

IR* JoinQueryJob::getIR() {
    return this->intermediateResult;
}

int JoinQueryJob::startJob() {
    auto join_start = std::chrono::high_resolution_clock::now();

    FullRelationIR *frLeftIr = dynamic_cast<FullRelationIR *>(leftJob->getIR());
    FullRelationIR *frRightIr = dynamic_cast<FullRelationIR *>(rightJob->getIR());
    IndexIR *iirLeftIr = dynamic_cast<IndexIR *>(leftJob->getIR());
    IndexIR *iirRightIr = dynamic_cast<IndexIR *>(rightJob->getIR());

    if (iirLeftIr && iirRightIr)
        intermediateResult = join(iirLeftIr, iirRightIr);
    else if (frLeftIr && frRightIr)
        intermediateResult = join(frLeftIr, frRightIr);
    else if (frLeftIr && iirRightIr)
        intermediateResult = join(frLeftIr, iirRightIr);
    else if (frRightIr && iirLeftIr)
        intermediateResult = join(frRightIr, iirLeftIr);
    else
        assert(false);

    auto join_end = std::chrono::high_resolution_clock::now();
    auto totalNanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(join_end-join_start).count();

    FullRelationIR *ir = dynamic_cast<FullRelationIR *>(intermediateResult);
    if (this->planTreeNode != nullptr) {
        this->planTreeNode->resultSize = ir->size();
        this->planTreeNode->nanosecTime = totalNanosec;
    }

    QueryExecutor::join_ns += totalNanosec;

    return 0; // TODO: -1 if error
}

mgpu::mem_t<int2> JoinQueryJob::innerJoinMulti(TYPEID* a, int a_count, TYPEID* b, int b_count) {
    // Compute lower and upper bounds of a into b.
    mgpu::mem_t<int> lower(a_count, *context);
    mgpu::mem_t<int> upper(a_count, *context);
    mgpu::sorted_search<mgpu::bounds_lower, mgpu::empty_t>(a, a_count, b, b_count, lower.data(), mgpu::less_t<TYPEID>(), *context);
    mgpu::sorted_search<mgpu::bounds_upper, mgpu::empty_t>(a, a_count, b, b_count, upper.data(), mgpu::less_t<TYPEID>(), *context);

    // Compute output ranges by scanning upper - lower. Retrieve the reduction
    // of the scan, which specifies the size of the output array to allocate.
    mgpu::mem_t<int> scanned_sizes(a_count, *context);
    const int* lower_data = lower.data();
    const int* upper_data = upper.data();

    mgpu::mem_t<int> count(1, *context);
    mgpu::transform_scan<int>([=] __device__(int index) {
        return upper_data[index] - lower_data[index];
    }, a_count, scanned_sizes.data(), mgpu::plus_t<int>(), count.data(), *context);

    // Allocate an int2 output array and use load-balancing search to compute
    // the join.
    int join_count = from_mem(count)[0];
    mgpu::mem_t<int2> output(join_count, *context);
    int2* output_data = output.data();

    // Use load-balancing search on the segmens. The output is a pair with
    // a_index = seg and b_index = lower_data[seg] + rank.
    /*auto k = [=] __device__(int index, int seg, int rank, mgpu::tuple<int> lower) {
        output_data[index] = make_int2(seg, mgpu::get<0>(lower) + rank);
    };
    mgpu::transform_lbs<mgpu::empty_t>(k, join_count, scanned_sizes.data(), a_count,
    mgpu::make_tuple(lower_data), *context);*/

    return output;
}

void split(std::string const &str, const char delim, std::vector<std::string> &out) {
    size_t start, end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

void JoinQueryJob::mergeColumns(DTYPEID* irMergeRelation, FullRelationIR *ir, size_t col0, size_t col1) {
    TYPEID *irFirstColPtr = ir->getRelationRawPointer(col0);
    TYPEID *irSecondColPtr = ir->getRelationRawPointer(col1);
    mgpu::transform([=] __device__(int index) {
        DTYPEID v = irFirstColPtr[index];
        irMergeRelation[index] = (v << 32) | irSecondColPtr[index];
    }, ir->size(), *context);
}

IR* JoinQueryJob::join(FullRelationIR *lir, FullRelationIR *rir) {
    // XXX: this line can help so much
    if (lir->size() > rir->size()) std::swap(lir, rir);

    auto join_func_start = std::chrono::high_resolution_clock::now();

    // Join with 2 variables
    std::vector<std::string> joinVars;
    split(joinVariable, ',', joinVars);
    bool Join1Var = joinVars.size() == 1;

    // Decrease variable count
    size_t new_col_num = lir->getColumnNum() + rir->getColumnNum() - joinVars.size();
    for (int i = lir->getColumnNum() - 1; i >= 0; --i) {
        for (int v = 0; v < joinVars.size(); v++) {
            if ((*(this->query_variable_counter))[lir->getHeader(i)] == 0 && lir->getHeader(i) != joinVars[v]) {
                lir->removeColumn(i, joinVars[v]);
                new_col_num -= 1;
            }
        }
    }
    for (int i = rir->getColumnNum() - 1; i >= 0; --i) {
        for (int v = 0; v < joinVars.size(); v++) {
            if ((*(this->query_variable_counter))[rir->getHeader(i)] == 0 && rir->getHeader(i) != joinVars[v]) {
                rir->removeColumn(i, joinVars[v]);
                new_col_num -= 1;
            }
        }
    }
    for (int v = 0; v < joinVars.size(); v++)
        (*(this->query_variable_counter))[joinVars[v]] -= 1;


    auto swap_index_start = std::chrono::high_resolution_clock::now();
    size_t lJoinIdx = 0, rJoinIdx = 0;
    mgpu::mem_t<DTYPEID> *lMergedPtr, *rMergedPtr;

    if (Join1Var) {
        lJoinIdx = lir->getColumnId(joinVars[0]);
        rJoinIdx = rir->getColumnId(joinVars[0]);
        // if lJoinIdx != 0 sort column lJoinIdx
        if (lJoinIdx != 0) {
            lir->swapColumn(0, lJoinIdx);
            lir->sortByFirstColumn();
            lJoinIdx = 0;
            // std::cout << "Implicit Index Swap LIR\n";
            QueryExecutor::exe_log.push_back( ExecuteLogRecord(SWAP_OP, "", lir->size(), lir->getColumnNum()) );
        }
        // if rJoinIdx != 0 sort column rJoinIdx
        if (rJoinIdx != 0) {
            rir->swapColumn(0, rJoinIdx);
            rir->sortByFirstColumn();
            rJoinIdx = 0;
            // std::cout << "Implicit Index Swap RIR\n";
            QueryExecutor::exe_log.push_back( ExecuteLogRecord(SWAP_OP, "", rir->size(), rir->getColumnNum()) );
        }
    } else {
        assert(joinVars.size() == 2);
        assert(lir->getColumnNum() >= joinVars.size());
        assert(rir->getColumnNum() >= joinVars.size());

        size_t rCol0 = 0, rCol1 = 1, lCol0 = 0, lCol1 = 1;

        if ((lir->getColumnId(joinVars[0]) == rir->getColumnId(joinVars[0])) &&
            (lir->getColumnId(joinVars[1]) == rir->getColumnId(joinVars[1])) &&
            ((lir->getColumnId(joinVars[0]) == 0 && lir->getColumnId(joinVars[1]) == 1) || ((lir->getColumnId(joinVars[0]) == 1 && lir->getColumnId(joinVars[1]) == 0)))) {
            // Can join without swap
            // TODO: check first 2 columns are sorted

        } else {

            // TODO: Optimize later
            /*if (lir->getColumnId(joinVars[0]) == 0 && lir->getColumnId(joinVars[1]) == 1) {
                rCol0 = rir->getColumnId(joinVars[0]);
                rCol1 = rir->getColumnId(joinVars[1]);
                lCol0 = 0, lCol1 = 1;
            } else if (lir->getColumnId(joinVars[0]) == 1 && lir->getColumnId(joinVars[1]) == 0) {
                rCol0 = rir->getColumnId(joinVars[1]);
                rCol1 = rir->getColumnId(joinVars[0]);
                lCol0 = 1, lCol1 = 0;
            } else {*/
                rCol0 = rir->getColumnId(joinVars[0]);
                rir->swapColumn(0, rCol0); rCol0 = 0;
                rCol1 = rir->getColumnId(joinVars[1]);
                rir->swapColumn(1, rCol1); rCol1 = 1;
                lCol0 = lir->getColumnId(joinVars[0]);
                lir->swapColumn(0, lCol0); lCol0 = 0;
                lCol1 = lir->getColumnId(joinVars[1]);
                lir->swapColumn(1, lCol1); lCol1 = 1;

                rir->sort(); // TODO: Sort only first 2 columns
                lir->sort(); // TODO: Sort only first 2 columns
            /*}*/
            // TODO: Check is sorted?
        }

        lMergedPtr = new mgpu::mem_t<DTYPEID>(lir->size(), *context);
        rMergedPtr = new mgpu::mem_t<DTYPEID>(rir->size(), *context);
        mergeColumns(lMergedPtr->data(), lir, lCol0, lCol1);
        mergeColumns(rMergedPtr->data(), rir, rCol0, rCol1);
    }


    auto swap_index_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::swap_index_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(swap_index_end-swap_index_start).count();

    assert(lJoinIdx == 0);
    assert(rJoinIdx == 0);

    // inner join
    TYPEID* leftRelationPtr = lir->getRelationRawPointer(lJoinIdx);
    TYPEID* rightRelationPtr = rir->getRelationRawPointer(rJoinIdx);
    size_t column_num = new_col_num;

    auto join_start = std::chrono::high_resolution_clock::now();

    /* If left or right is zero */
    if (lir->size() == 0 || rir->size() == 0) {
        FullRelationIR *newIr = new FullRelationIR(column_num, 0);

        size_t headerIdx = 0;
        size_t lSize = lir->getColumnNum(), rSize = rir->getColumnNum();
        for (size_t i = 0; i < joinVars.size(); ++i)
            newIr->setHeader(headerIdx++, lir->getHeader(i), lir->getIsPredicate(i));
        for (size_t i = joinVars.size(); i < lSize; ++i) newIr->setHeader(headerIdx++, lir->getHeader(i), lir->getIsPredicate(i));
        for (size_t i = joinVars.size(); i < rSize; ++i) newIr->setHeader(headerIdx++, rir->getHeader(i), rir->getIsPredicate(i));

        std::string joinDetail = lir->getHeaders("") + " x " + rir->getHeaders("");
        QueryExecutor::exe_log.push_back( ExecuteLogRecord(JOIN_OP, joinDetail, lir->size(), rir->size(), 0) );

        return newIr;
    } else {

    }

    mgpu::mem_t<int2> joined_d;
    if (Join1Var) {
        joined_d = mgpu::inner_join(
            leftRelationPtr, static_cast<int>(lir->size()),
            rightRelationPtr, static_cast<int>(rir->size()), mgpu::less_t<TYPEID>(), *context);
    } else {
        joined_d = mgpu::inner_join(
            (DTYPEID*)(lMergedPtr->data()), static_cast<int>(lir->size()),
            (DTYPEID*)(rMergedPtr->data()), static_cast<int>(rir->size()), mgpu::less_t<DTYPEID>(), *context);
    }

    auto join_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::join_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(join_end-join_start).count();

#ifdef TIME_DEBUG
    std::cout << "\tJoin time : " << std::chrono::duration_cast<std::chrono::microseconds>(join_end-join_start).count() << " microsecond ( " <<
                 lir->size() << " x " << rir->size() << ")\n";
    std::cout << "\tResult size : " << joined_d.size() << "\n";
#endif

    size_t result_size = joined_d.size();


    // return new FullRelationIR with union of headers, start with join variable
    FullRelationIR *newIr = new FullRelationIR(column_num, result_size);

    // Create new header
    size_t headerIdx = 0;
    size_t lSize = lir->getColumnNum(), rSize = rir->getColumnNum();
    for (size_t i = 0; i < joinVars.size(); ++i)
        newIr->setHeader(headerIdx++, lir->getHeader(i), lir->getIsPredicate(i));
    for (size_t i = joinVars.size(); i < lSize; ++i) newIr->setHeader(headerIdx++, lir->getHeader(i), lir->getIsPredicate(i));
    for (size_t i = joinVars.size(); i < rSize; ++i) newIr->setHeader(headerIdx++, rir->getHeader(i), rir->getIsPredicate(i));

    // Setup new relation
    TYPEID* firstCol = newIr->getRelationRawPointer(0);
    TYPEID* lJoinCol = lir->getRelationRawPointer(0);
    int2* joinPtr = joined_d.data();

    thrust::host_vector<TYPEID*> newIrRelations(column_num);
    thrust::host_vector<TYPEID*> leftIrRelations(lir->getColumnNum());
    thrust::host_vector<TYPEID*> rightIrRelations(rir->getColumnNum());
    newIr->getRelationPointers(thrust::raw_pointer_cast(newIrRelations.data()));
    lir->getRelationPointers(thrust::raw_pointer_cast(leftIrRelations.data()));
    rir->getRelationPointers(thrust::raw_pointer_cast(rightIrRelations.data()));

    auto alloc_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<TYPEID*> newIrD = newIrRelations;
    thrust::device_vector<TYPEID*> lIrD = leftIrRelations, rIrD = rightIrRelations;
    auto alloc_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::alloc_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(alloc_end-alloc_start).count();


    auto bp_start = std::chrono::high_resolution_clock::now();
    TYPEID** newIrPtr = thrust::raw_pointer_cast(newIrD.data());
    TYPEID** leftIrPtr = thrust::raw_pointer_cast(lIrD.data());
    TYPEID** rightIrPtr = thrust::raw_pointer_cast(rIrD.data());

    //if (lastJoinForVar) {
        // TODO: 
    //}

    if (Join1Var) {
        if (lSize == 1 && rSize == 1) {
            mgpu::transform([=] __device__(int index) {
                firstCol[index] = lJoinCol[ joinPtr[index].x ];
            }, joined_d.size(), *context);
        } else if (lSize == 1) {
            mgpu::transform([=] __device__(int index) {
                size_t resultIdx = 1;
                int right_idx = joinPtr[index].y;
                firstCol[index] = lJoinCol[ joinPtr[index].x ];
                for (size_t i = 1; i < rSize; ++i) {
                    newIrPtr[resultIdx++][index] = rightIrPtr[i][right_idx];
                }
            }, joined_d.size(), *context);
        } else if (rSize == 1) {
            mgpu::transform([=] __device__(int index) {
                size_t resultIdx = 1;
                int left_idx = joinPtr[index].x;
                firstCol[index] = lJoinCol[left_idx];
                for (size_t i = 1; i < lSize; ++i) {
                    newIrPtr[resultIdx++][index] = leftIrPtr[i][left_idx];
                }
            }, joined_d.size(), *context);
        } else {
            mgpu::transform([=] __device__(int index) {
                int left_idx = joinPtr[index].x;
                int right_idx = joinPtr[index].y;
                size_t resultIdx = 1;
                firstCol[index] = lJoinCol[left_idx];

                for (size_t i = 1; i < lSize; ++i) {
                    newIrPtr[resultIdx++][index] = leftIrPtr[i][left_idx];
                }
                for (size_t i = 1; i < rSize; ++i) {
                    newIrPtr[resultIdx++][index] = rightIrPtr[i][right_idx];
                }
            }, joined_d.size(), *context);
        }
    } else {
        TYPEID* secondCol = newIr->getRelationRawPointer(1);
        delete rMergedPtr;

        DTYPEID* mergeCol = lMergedPtr->data();
        mgpu::transform([=] __device__(int index) {
            int left_idx = joinPtr[index].x;
            int right_idx = joinPtr[index].y;
            size_t resultIdx = 2;

            secondCol[index] = mergeCol[left_idx] & 0xFFFFFFFF;
            firstCol[index] = (mergeCol[left_idx] >> 32) & 0xFFFFFFFF;

            for (size_t i = 2; i < lSize; ++i)
                newIrPtr[resultIdx++][index] = leftIrPtr[i][left_idx];
            for (size_t i = 2; i < rSize; ++i)
                newIrPtr[resultIdx++][index] = rightIrPtr[i][right_idx];
        }, joined_d.size(), *context);

        delete lMergedPtr;
    }

    auto bp_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::join_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(bp_end-bp_start).count();

    std::string joinDetail = lir->getHeaders("") + " x " + rir->getHeaders("");
    QueryExecutor::exe_log.push_back( ExecuteLogRecord(JOIN_OP, joinDetail, lir->size(), rir->size(), result_size) );

    if (QueryExecutor::ENABLE_UPDATE_BOUND_AFTER_JOIN) {
        // TODO: for 2 variable
        if (result_size > 0) {
            auto copy_start = std::chrono::high_resolution_clock::now();
            TYPEID start_value = (*newIr->getRelation(0))[0];
            TYPEID end_value = (*newIr->getRelation(0))[result_size-1];
            auto copy_end = std::chrono::high_resolution_clock::now();

            QueryExecutor::updateBoundDict(variables_bound, joinVariable, start_value, end_value);
            
            if (!lastJoinForVar && result_size > 2 && result_size < EIF_ALPHA) {
                // TODO: Find largest empty interval on GPU
                auto eif_start = std::chrono::high_resolution_clock::now();
                TYPEID_DEVICE_VEC diff_vec(newIr->size() - 1);
                thrust::transform(thrust::device,
                    newIr->getRelation(0)->begin() + 1, newIr->getRelation(0)->end(), 
                    newIr->getRelation(0)->begin(),
                    diff_vec.begin(), thrust::minus<TYPEID>());
                auto max_iter = thrust::max_element(thrust::device, diff_vec.begin(), diff_vec.end());

                auto max_diff_pos = max_iter - diff_vec.begin();
                TYPEID v[6], b[4] = {0, 0};
                auto start_copy = (max_diff_pos - 2 >= 0) ? max_diff_pos - 2 : 0;
                auto end_copy = (max_diff_pos + 4 <= newIr->size()) ? max_diff_pos + 4 : newIr->size();
                auto size_copy = end_copy - start_copy;
                cudaMemcpy(v, newIr->getRelationRawPointer(0) + start_copy, size_copy * sizeof(TYPEID), cudaMemcpyDeviceToHost);

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
                QueryExecutor::eif_count++;
                ei_dict->updateBound(joinVariable, b[0] + 1, b[1] - 1, b[2] + 1, b[3] - 1);
                auto eif_end = std::chrono::high_resolution_clock::now();
                QueryExecutor::eif_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(eif_end-eif_start).count();
            }

        } else {
            /* result_size == 0 */
            QueryExecutor::updateBoundDict(variables_bound, joinVariable, 0, 0);
        }
        
    }

    auto join_func_end = std::chrono::high_resolution_clock::now();

#ifdef TIME_DEBUG
    std::cout << "\tDevice mem allocation time : " << std::chrono::duration_cast<std::chrono::microseconds>(alloc_end-alloc_start).count() << " microsecond\n";
    // std::cout << "\tD2D memory copy time : " << std::chrono::duration_cast<std::chrono::microseconds>(d2d_copy_end-d2d_copy_start).count() << " microsecond\n";
    std::cout << "\tFull relation upload time : " << std::chrono::duration_cast<std::chrono::microseconds>(bp_end-bp_start).count() << " microsecond\n";
    std::cout << "\tJoin function time : " << std::chrono::duration_cast<std::chrono::microseconds>(join_func_end-join_func_start).count() << " microsecond\n";
    // std::cout << "Copy bound from GPU time : " << std::chrono::duration_cast<std::chrono::microseconds>(copy_end-copy_start).count() << " microsecond\n";
#endif

    return newIr;
}

IR* JoinQueryJob::join(FullRelationIR *lir, IndexIR *rir) {
    //size_t lJoinIdx = lir->getColumnId(joinVariable);

    assert(false);

    if (rir->getIndexVariable() != joinVariable) {
        // TODO: compare cost of swap index to create IndexIR from FullRelationIR
        FullRelationIR *fullRir = rir->toFullRelationIR();
        return join(lir, fullRir);
    }

    if (lir->getColumnNum() == 1) {
        // XXX: we can join fill ir with index and ignore Left fill ir
#ifdef DEBUG
        std::cout << "Special case\n";
#endif

        TYPEID* leftRelationPtr = thrust::raw_pointer_cast(lir->getRelation(0)->data());
        TYPEID* rightIdxVarPtr = thrust::raw_pointer_cast(rir->getIndexVars()->data());
        mgpu::mem_t<int2> joined_d = mgpu::inner_join(
          leftRelationPtr, static_cast<int>(lir->getRelationSize(0)),
          rightIdxVarPtr, static_cast<int>(rir->getIndexSize()),
          mgpu::less_t<TYPEID>(), *context);
        size_t joined_idx_size = joined_d.size();
        int2* joinPtr = joined_d.data();

        IndexIR *newIr = new IndexIR(rir->getColumnNum(), joined_idx_size, context);
        newIr->setIndexVariable(rir->getIndexVariable(), rir->getIndexIsPredicate());
        this->filterJoinedIndexVars(newIr->getIndexVarsRawPointer(), rir->getIndexVarsRawPointer(), joinPtr, joined_idx_size);

        // skip left size

        // right size
        // std::cout << "Right relation number : " << rir->getColumnNum() << "\n";
        // TODO: resize old ir size
        for (size_t c = 0; c < rir->getColumnNum(); c++) {
            // Calculate joined indexed offset size
            TYPEID_DEVICE_VEC right_offst_size(joined_idx_size);
            this->calculateJoinIndexOffsetSize(rir->getIndexOffsetsRawPtr(c), rir->getIndexOffset(c)->size(),
                                               thrust::raw_pointer_cast(right_offst_size.data()),
                                               joinPtr, joined_idx_size, rir->getRelationSize(c), false);
            thrust::exclusive_scan(thrust::device, right_offst_size.begin(), right_offst_size.end(), newIr->getIndexOffset(c)->begin());

            TYPEID_DEVICE_VEC* right_relation = rir->getRelationData(c);

            size_t right_relation_size = thrust::reduce(thrust::device, right_offst_size.begin(), right_offst_size.end());

            size_t new_right_size = this->filterJoinedRelation(rir->getIndexOffsetsRawPtr(c), thrust::raw_pointer_cast(right_offst_size.data()),
                                   right_relation, rir->getRelationSize(c), right_relation_size, joinPtr,
                                   joined_idx_size, false);
            newIr->setHeader(c, rir->getHeader(c), rir->getIsPredicate(c));
            newIr->resizeRelation(c, new_right_size);
            newIr->setRelationData(c, right_relation->begin(), right_relation->begin() + new_right_size);
            // std::cout << "Right relation " << c << " size after filter : " << new_right_size << "\n";
        }

        return newIr;
    } else {
        assert(false && "Not implement join for multi-column full relation"); // TODO: not implement yet

        return nullptr;
    }
}

void JoinQueryJob::filterJoinedIndexVars(TYPEID *new_idx_vars, TYPEID *orig_idx_vars, int2* joined_idx, size_t joined_idx_size) {
    auto set_new_idx = [=] __device__(size_t index) {
        new_idx_vars[index] = orig_idx_vars[joined_idx[index].x];
    };
    mgpu::transform(set_new_idx, joined_idx_size, *context);
}

void JoinQueryJob::calculateJoinIndexOffsetSize(TYPEID *original_offset, size_t offset_num, TYPEID *offset_size,
                                                int2* joined_idx, size_t joined_idx_size,
                                                size_t orig_relation_size, bool left) {
    mgpu::transform([=] __device__(size_t index) {

        int i = left? joined_idx[index].x : joined_idx[index].y;
        bool isLast = i == (offset_num - 1);
        size_t next_offset = isLast? orig_relation_size : original_offset[i+1];
        offset_size[index] = next_offset - original_offset[i];
    }, joined_idx_size, *context);
}

struct not_marked_test
{
    CUDA_CALLABLE_MEMBER
    bool operator()(const thrust::tuple<TYPEID, int>& a) {
        return thrust::get<1>(a) == 0;
    }
};

size_t JoinQueryJob::filterJoinedRelation(TYPEID *orig_idx_offst_ptr, TYPEID *orig_idx_size_ptr, TYPEID_DEVICE_VEC* orig_relation,
                          size_t orig_relation_size, size_t new_relation_size,
                          int2* joined_idx, size_t joined_idx_size, bool left) {
    thrust::device_vector<int> marked(orig_relation_size);
    thrust::fill(marked.begin(), marked.end(), 0);
    int *marked_ptr = thrust::raw_pointer_cast(marked.data());

    // set start offset to 1
    mgpu::transform([=] __device__(size_t index) {
        int i = left? joined_idx[index].x : joined_idx[index].y;
        size_t start_offset = orig_idx_offst_ptr[i];
        marked_ptr[start_offset] += 1;
    }, joined_idx_size, *context);

    // decrease end offset by 1
    mgpu::transform([=] __device__(size_t index) {
        int i = left? joined_idx[index].x : joined_idx[index].y;
        size_t end_offset = orig_idx_offst_ptr[i] + orig_idx_size_ptr[index];
        if (end_offset < orig_relation_size) {
            marked_ptr[end_offset] += -1;
        }
    }, joined_idx_size, *context);

#ifdef DEBUG
    std::cout << "MARKED 1 : ";
    for (size_t i = 0; i < orig_relation_size; ++i) std::cout << marked[i] << " ";
    std::cout << "\n";
#endif

    // prefix sum
    thrust::inclusive_scan(marked.begin(), marked.end(), marked.begin() /* inplace prefix sum */);

#ifdef DEBUG
    std::cout << "MARKED 2 : ";
    for (size_t i = 0; i < orig_relation_size; ++i) std::cout << marked[i] << " ";
    std::cout << "\n";
#endif

    // filter

    auto begin_it = thrust::make_zip_iterator(thrust::make_tuple(orig_relation->begin(), marked.begin()));
    auto end_it   = thrust::make_zip_iterator(thrust::make_tuple(orig_relation->end(), marked.end()));
    auto new_zipped_end = thrust::remove_if(begin_it, end_it, not_marked_test());
    size_t new_size = static_cast<size_t>(thrust::distance(begin_it, new_zipped_end));

    //assert(new_size == new_relation_size);

#ifdef DEBUG
    std::cout << "NEW RELATION : ";
    for (size_t i = 0; i < new_relation_size; ++i) std::cout << (*orig_relation)[i] << " ";
    std::cout << "\n";
#endif

    return new_size;
}

void printDeviceVecData(std::string s, TYPEID_DEVICE_VEC *v) {

    std::cout << s << " : ";
    TYPEID_HOST_VEC hv(v->size());
    thrust::copy(v->begin(), v->end(), hv.begin());
    for (TYPEID o : hv) {
        std::cout << o << " ";
    }
    std::cout << "\n";
}

IR* JoinQueryJob::join(IndexIR *lir, IndexIR *rir) {
    // thrust copy_if
    // thrust remove_if

    assert(false);

    assert(lir->getIndexVariable() == rir->getIndexVariable());

    // TODO: after join if one column data is 0, result IR is empty
    size_t column_num = lir->getColumnNum() + rir->getColumnNum();


    auto join_start = std::chrono::high_resolution_clock::now();
    // inner join, join the index variables
    TYPEID* leftIdxVarPtr = thrust::raw_pointer_cast(lir->getIndexVars()->data());
    TYPEID* rightIdxVarPtr = thrust::raw_pointer_cast(rir->getIndexVars()->data());
    mgpu::mem_t<int2> joined_d = mgpu::inner_join(
      leftIdxVarPtr, static_cast<int>(lir->getIndexSize()),
      rightIdxVarPtr, static_cast<int>(rir->getIndexSize()),
      mgpu::less_t<TYPEID>(), *context);
    size_t joined_idx_size = joined_d.size();
    int2* joinPtr = joined_d.data();
    auto join_end = std::chrono::high_resolution_clock::now();


#ifdef DEBUG
    std::cout << "Indexed IR Join size : " << joined_idx_size << "\n";
    std::cout << "COLUMN NUM : " << column_num << "\n";
    std::vector<int2> c_host = from_mem(joined_d);
    for (int2 p : c_host) std::cout << "\t" << p.x << " - " << p.y << "\n";

    printDeviceVecData("Left Index Offset", lir->getIndexOffset(0));
    printDeviceVecData("Right Index Offset", rir->getIndexOffset(0));
#endif

    auto clean_index_start = std::chrono::high_resolution_clock::now();
    // Create new IR for store joined result
    IndexIR *newIr = new IndexIR(column_num, joined_idx_size, context);
    newIr->setIndexVariable(lir->getIndexVariable(), lir->getIndexIsPredicate());
    this->filterJoinedIndexVars(newIr->getIndexVarsRawPointer(), lir->getIndexVarsRawPointer(), joinPtr, joined_idx_size);
    auto clean_index_end = std::chrono::high_resolution_clock::now();

    // Find new relation size
    size_t newIrHeaderIdx = 0;

    double left_copy_acc = 0;
    auto clean_left_start = std::chrono::high_resolution_clock::now();
    for (size_t c = 0; c < lir->getColumnNum(); c++) {
        // Calculate joined indexed offset size
        TYPEID_DEVICE_VEC left_offst_size(joined_idx_size);
        this->calculateJoinIndexOffsetSize(lir->getIndexOffsetsRawPtr(c), lir->getIndexOffset(c)->size(),
                                           thrust::raw_pointer_cast(left_offst_size.data()),
                                           joinPtr, joined_idx_size, lir->getRelationSize(c), true);
        thrust::exclusive_scan(thrust::device, left_offst_size.begin(), left_offst_size.end(), newIr->getIndexOffset(newIrHeaderIdx)->begin());

        TYPEID_DEVICE_VEC* left_relation = lir->getRelationData(c);
        size_t left_relation_size = thrust::reduce(thrust::device, left_offst_size.begin(), left_offst_size.end());

//        std::cout << "Left relation " << c << " size before filter : " << left_relation_size << " (relation : " << lir->getRelationSize(c) << ") \n";
        size_t new_left_size = this->filterJoinedRelation(lir->getIndexOffsetsRawPtr(c), thrust::raw_pointer_cast(left_offst_size.data()),
                                   left_relation, lir->getRelationSize(c), left_relation_size, joinPtr,
                                   joined_idx_size, true);
        newIr->setHeader(newIrHeaderIdx, lir->getHeader(c), lir->getIsPredicate(c));
        auto copy_left_start = std::chrono::high_resolution_clock::now();
        newIr->resizeRelation(newIrHeaderIdx, new_left_size);
        newIr->setRelationData(newIrHeaderIdx, left_relation->begin(), left_relation->begin() + new_left_size);
        auto copy_left_end = std::chrono::high_resolution_clock::now();
        left_copy_acc += std::chrono::duration_cast<std::chrono::milliseconds>(copy_left_end - copy_left_start).count();
//        std::cout << "Left relation " << c << " size after filter : " << new_left_size << "\n";
        newIrHeaderIdx++;
    }
    auto clean_left_end = std::chrono::high_resolution_clock::now();

    double right_copy_acc = 0;
    auto clean_right_start = std::chrono::high_resolution_clock::now();
//    std::cout << "Right relation number : " << rir->getColumnNum() << "\n";
    for (size_t c = 0; c < rir->getColumnNum(); c++) {
        // Calculate joined indexed offset size
        TYPEID_DEVICE_VEC right_offst_size(joined_idx_size);
        this->calculateJoinIndexOffsetSize(rir->getIndexOffsetsRawPtr(c), rir->getIndexOffset(c)->size(),
                                           thrust::raw_pointer_cast(right_offst_size.data()),
                                           joinPtr, joined_idx_size, rir->getRelationSize(c), false);
        thrust::exclusive_scan(thrust::device, right_offst_size.begin(), right_offst_size.end(), newIr->getIndexOffset(newIrHeaderIdx)->begin());

        TYPEID_DEVICE_VEC* right_relation = rir->getRelationData(c);

        size_t right_relation_size = thrust::reduce(thrust::device, right_offst_size.begin(), right_offst_size.end());

        size_t new_right_size = this->filterJoinedRelation(rir->getIndexOffsetsRawPtr(c), thrust::raw_pointer_cast(right_offst_size.data()),
                               right_relation, rir->getRelationSize(c), right_relation_size, joinPtr,
                               joined_idx_size, false);
        newIr->setHeader(newIrHeaderIdx, rir->getHeader(c), rir->getIsPredicate(c));
        auto copy_right_start = std::chrono::high_resolution_clock::now();
        newIr->resizeRelation(newIrHeaderIdx, new_right_size);
        newIr->setRelationData(newIrHeaderIdx, right_relation->begin(), right_relation->begin() + new_right_size);
        auto copy_right_end = std::chrono::high_resolution_clock::now();
        right_copy_acc += std::chrono::duration_cast<std::chrono::milliseconds>(copy_right_end - copy_right_start).count();
        newIrHeaderIdx++;
    }
    auto clean_right_end = std::chrono::high_resolution_clock::now();

    // TODO: eliminate copy relation
    //newIr->setRelationData(0, left_relation->begin(), left_relation->begin() + new_left_size);
    //newIr->setRelationData(1, right_relation->begin(), right_relation->begin() + new_right_size);

    std::cout << "IndexIR join time  : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(join_end-join_start).count() << " ms.\n";
    std::cout << "Clean index time   : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(clean_index_end-clean_index_start).count() << " ms.\n";
    std::cout << "Clean left time    : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(clean_left_end-clean_left_start).count() << " ms.\n";
    std::cout << "Copy left time    : " << std::setprecision(3) << left_copy_acc << " ms.\n";
    std::cout << "Clean right time   : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(clean_right_end-clean_right_start).count() << " ms.\n";
    std::cout << "Copy right time    : " << std::setprecision(3) << right_copy_acc << " ms.\n";

    return newIr;
}

IR* JoinQueryJob::multiJoinIndexedIr(std::vector<IndexIR*> &irs) {
    for (auto it = irs.begin() + 1; it != irs.end(); it++)
        assert((*irs.begin())->getIndexVariable() == (*it)->getIndexVariable());

//    size_t column_num = std::accumulate(irs.begin(), irs.end(), 0, [](const IndexIR* ir, size_t acc) {
//        return ir->getColumnNum() + acc;
//    });
    return nullptr;
}

void JoinQueryJob::print() const {
    std::cout << "\tJOIN JOB - join with " << joinVariable << "\n";
}

std::string JoinQueryJob::jobTypeName() const {
    return "Join Job";
}

void JoinQueryJob::setQueryVarCounter(std::map<std::string, size_t> *query_variable_counter) {
    this->query_variable_counter = query_variable_counter;
}

unsigned JoinQueryJob::getLeftIRSize() const {
    return dynamic_cast<FullRelationIR*>(this->leftJob->getIR())->size();
}

unsigned JoinQueryJob::getRightIRSize() const {
    return dynamic_cast<FullRelationIR*>(this->rightJob->getIR())->size();
}

std::string JoinQueryJob::getJoinVariable() const {
    return this->joinVariable;
}