#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "vedas.h"
#include "QueryExecutor.h"
#include "SelectQueryJob.h"
#include "IndexIR.h"
#include "FullRelationIR.h"

using namespace std;

SelectQueryJob::SelectQueryJob(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                               TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                               TYPEID_HOST_VEC *l2_data,
                               string v1, string v2, TYPEID id1, TYPEID_HOST_VEC *data, TYPEID_DEVICE_VEC *device_data,
                               std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound,
                               bool *is_predicates, bool is_second_var_used, mgpu::standard_context_t* context) {
    this->l1_index_values = l1_index_values;
    this->l1_index_offsets = l1_index_offsets;
    this->l2_index_values = l2_index_values;
    this->l2_index_offsets = l2_index_offsets;
    this->l2_data = l2_data;
    this->variables[0] = v1;
    this->variables[1] = v2;
    this->is_predicates[0] = is_predicates[0];
    this->is_predicates[1] = is_predicates[1];
    variable_num = 2;
    this->dataid[0] = id1;
    this->data = data;
    this->device_data = device_data;
    this->variables_bound = variables_bound;
    this->is_second_var_used = is_second_var_used;

    this->context = context;
}

SelectQueryJob::SelectQueryJob(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                               TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                               string v1, TYPEID id1, TYPEID id2, TYPEID_HOST_VEC *data, TYPEID_DEVICE_VEC *device_data,
                               std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound,
                               bool *is_predicates, mgpu::standard_context_t* context) {
    this->l1_index_values = l1_index_values;
    this->l1_index_offsets = l1_index_offsets;
    this->l2_index_values = l2_index_values;
    this->l2_index_offsets = l2_index_offsets;
    this->variables[0] = v1;
    this->is_predicates[0] = is_predicates[0];
    variable_num = 1;
    this->dataid[0] = id1;
    this->dataid[1] = id2;
    this->data = data;
    this->device_data = device_data;
    this->variables_bound = variables_bound;

    this->context = context;
}

SelectQueryJob::~SelectQueryJob() {
    if (intermediateResult != nullptr) {
        delete intermediateResult;
        intermediateResult = nullptr;
    }
}

void SelectQueryJob::startJob() {
    // std::cout << "SELECT FOR ID " << dataid[0] << "\n";
    auto l2_offst = QueryExecutor::findL2OffsetFromL1(l1_index_values, l1_index_offsets, dataid[0], data->size());

    if (getVariableNum() == 1) {
        auto data_offst_pair = (l2_data != nullptr)?
                    QueryExecutor::findDataOffsetFromL2(l2_data, l2_offst.first, l2_offst.second, dataid[1], data->size()) :
                    QueryExecutor::findDataOffsetFromL2(l2_index_values, l2_index_offsets, l2_offst.first, l2_offst.second, dataid[1], data->size());

        // Tighten bound
        if (variables_bound != nullptr && variables_bound->count(this->variables[0]) > 0) {
            auto bound = (*variables_bound)[this->variables[0]];

            /*TYPEID curr_min = *(l2_index_values->begin() + start_offst2);
            TYPEID curr_max = *(l2_index_values->begin() + end_offst2 - 1);

            if (bound.first > curr_min) {
                auto new_offset_bit = thrust::lower_bound(thrust::host, l2_begin, l2_end, bound.first);
                start_offst2 = thrust::distance(l2_index_values->begin(), new_offset_bit);
            }
            if (bound.second < curr_max) {
                auto new_offst_eit = thrust::upper_bound(thrust::host, l2_begin, l2_end, bound.second);
                end_offst2 = thrust::distance(l2_index_values->begin(), new_offst_eit);
            }*/
        }

        if (data_offst_pair.first > data_offst_pair.second)
            data_offst_pair.second = data_offst_pair.first;

        // select from first and second indices
        size_t columnNum = 1;
        size_t relationNum = data_offst_pair.second - data_offst_pair.first; // XXX: not plus 1, end iterator is exclusive
        FullRelationIR *fullIr = new FullRelationIR(columnNum, relationNum);

        fullIr->setHeader(0, variables[0], is_predicates[0]);
        auto upload_start = std::chrono::high_resolution_clock::now();
        if (this->device_data != nullptr)
            fullIr->setRelation(0, device_data->begin() + data_offst_pair.first, device_data->begin() + data_offst_pair.second);
        else
            fullIr->setRelation(0, data->begin() + data_offst_pair.first, data->begin() + data_offst_pair.second);
        auto upload_end = std::chrono::high_resolution_clock::now();

        QueryExecutor::upload_ms += std::chrono::duration_cast<std::chrono::milliseconds>(upload_end-upload_start).count();

#ifdef TIME_DEBUG
        std::cout << "Full relation upload time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(upload_end-upload_start).count()
             << " ms. (Data size = " << data_offst_pair.second - data_offst_pair.first << ")\n";
#endif

        intermediateResult = fullIr;
    } else if (getVariableNum() == 2) {

//        std::cout << "L2 START OFFSET : " << l2_offst.first << "\n";
//        std::cout << "L2 END OFFSET : " << l2_offst.second << "\n";

        // Find variable bound
        if (variables_bound != nullptr && variables_bound->count(this->variables[0]) > 0) {
            auto bound = (*variables_bound)[this->variables[0]];
            if (l2_data != nullptr) {
                TYPEID curr_min = *(l2_data->begin() + l2_offst.first);
                TYPEID curr_max = *(l2_data->begin() + l2_offst.second - 1);
#ifdef TIME_DEBUG
                std::cout << "\tCURRENT MIN_ : " << curr_min << "\n";
                std::cout << "\tBOUNDED MIN_ : " << bound.first << "\n";
                std::cout << "\tCURRENT MAX_ : " << curr_max << "\n";
                std::cout << "\tBOUNDED MAX_ : " << bound.second << "\n";
#endif

                auto l2_begin = l2_data->begin() + l2_offst.first;
                auto l2_end = l2_data->begin() + l2_offst.second;

                if (bound.first > curr_min) {
                    auto new_offset_bit = thrust::lower_bound(thrust::host, l2_begin, l2_end, bound.first);
                    l2_offst.first = thrust::distance(l2_data->begin(), new_offset_bit);
                }
                if (bound.second < curr_max) {
                    auto new_offst_eit = thrust::upper_bound(thrust::host, l2_begin, l2_end, bound.second);
                    l2_offst.second = thrust::distance(l2_data->begin(), new_offst_eit);
                }
            } else {
                TYPEID curr_min = *(l2_index_values->begin() + l2_offst.first);
                TYPEID curr_max = *(l2_index_values->begin() + l2_offst.second - 1);
#ifdef TIME_DEBUG
                std::cout << "\tCURRENT MIN : " << curr_min << "\n";
                std::cout << "\tBOUNDED MIN : " << bound.first << "\n";
                std::cout << "\tCURRENT MAX : " << curr_max << "\n";
                std::cout << "\tBOUNDED MAX : " << bound.second << "\n";
#endif
                auto l2_begin = l2_index_values->begin() + l2_offst.first;
                auto l2_end = l2_index_values->begin() + l2_offst.second;

                if (bound.first > curr_min) {
                    auto new_offset_bit = thrust::lower_bound(thrust::host, l2_begin, l2_end, bound.first);
                    l2_offst.first = thrust::distance(l2_index_values->begin(), new_offset_bit);
                }
                if (bound.second < curr_max) {
                    auto new_offst_eit = thrust::upper_bound(thrust::host, l2_begin, l2_end, bound.second);
                    l2_offst.second = thrust::distance(l2_index_values->begin(), new_offst_eit);
                }
            }
        } // end find variable bound

        size_t data_start_offst = l2_offst.first, data_end_offst = l2_offst.second;

        if (data_start_offst > data_end_offst) data_end_offst = data_start_offst;

        // Copy only first column
        if (!is_second_var_used) {
            std::cout << "\tDONT USE " << this->variables[1] << " - size : " << l2_offst.second - l2_offst.first << "\n";
            FullRelationIR *fullIr = new FullRelationIR(1, l2_offst.second - l2_offst.first);
            fullIr->setHeader(0, variables[0], is_predicates[0]);
            auto upload_start = std::chrono::high_resolution_clock::now();
            if (l2_data != nullptr) {
                fullIr->setRelation(0, l2_data->begin() + l2_offst.first, l2_data->begin() + l2_offst.second);
                fullIr->removeDuplicate();
            } else {
                fullIr->setRelation(0, l2_index_values->begin() + l2_offst.first, l2_index_values->begin() + l2_offst.second);
            }
            auto upload_end = std::chrono::high_resolution_clock::now();
            QueryExecutor::upload_ms += std::chrono::duration_cast<std::chrono::milliseconds>(upload_end-upload_start).count();

            intermediateResult = fullIr;
            return;
        }
        
        size_t columnNum = 2;
        size_t relationNum = data_end_offst - data_start_offst;
        FullRelationIR *fullIr = new FullRelationIR(columnNum, relationNum);

        fullIr->setHeader(0, variables[0], is_predicates[0]);
        fullIr->setHeader(1, variables[1], is_predicates[1]);

        if (l2_data != nullptr) {
            auto upload_l2_start = std::chrono::high_resolution_clock::now();
            fullIr->setRelation(0, l2_data->begin() + data_start_offst, l2_data->begin() + data_end_offst);
            auto upload_l2_end = std::chrono::high_resolution_clock::now();
            QueryExecutor::upload_ms += std::chrono::duration_cast<std::chrono::milliseconds>(upload_l2_end-upload_l2_start).count();
#ifdef TIME_DEBUG
            std::cout << "Upload L2 data time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(upload_l2_end-upload_l2_start).count()
                 << "\n";
#endif
        } else {
            auto upload_idx_start = std::chrono::high_resolution_clock::now();
            TYPEID_DEVICE_VEC index_values(l2_offst.second - l2_offst.first);
            TYPEID_DEVICE_VEC index_offsets(l2_offst.second - l2_offst.first);
            thrust::copy(l2_index_values->begin() + l2_offst.first, l2_index_values->begin() + l2_offst.second, index_values.begin());
            thrust::copy(l2_index_offsets->begin() + l2_offst.first, l2_index_offsets->begin() + l2_offst.second, index_offsets.begin());
            TYPEID first_offset = *(l2_index_offsets->begin() + l2_offst.first);

            TYPEID *relation = fullIr->getRelationRawPointer(0);
            TYPEID *offset_ptr = thrust::raw_pointer_cast(index_offsets.data());
            TYPEID *value_ptr = thrust::raw_pointer_cast(index_values.data());
            auto upload_idx_end = std::chrono::high_resolution_clock::now();

            auto transform_start = std::chrono::high_resolution_clock::now();
            // set start point
            mgpu::transform([=] __device__(size_t index) {
                relation[offset_ptr[index] - first_offset] = value_ptr[index];
            }, l2_offst.second - l2_offst.first, *context);

            // decrease at end point
            mgpu::transform([=] __device__(size_t index) {
                relation[offset_ptr[index+1] - first_offset] += -value_ptr[index];
            }, l2_offst.second - l2_offst.first - 1, *context);

            thrust::inclusive_scan(fullIr->getRelation(0)->begin(), fullIr->getRelation(0)->end(), fullIr->getRelation(0)->begin()/* inplace prefix sum */);
            auto transform_end = std::chrono::high_resolution_clock::now();
#ifdef TIME_DEBUG
            std::cout << "Upload L2 index time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(upload_idx_end-upload_idx_start).count()
                 << "\n";
            std::cout << "Transform to full relation time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(transform_end-transform_start).count()
                 << "\n";
#endif

            QueryExecutor::upload_ms += std::chrono::duration_cast<std::chrono::milliseconds>(upload_idx_end-upload_idx_start).count();
            QueryExecutor::upload_ms += std::chrono::duration_cast<std::chrono::milliseconds>(transform_end-transform_start).count();
        }


        auto upload_start = std::chrono::high_resolution_clock::now();
        if (this->device_data != nullptr)
            fullIr->setRelation(1, device_data->begin() + data_start_offst, device_data->begin() + data_end_offst);
        else
            fullIr->setRelation(1, data->begin() + data_start_offst, data->begin() + data_end_offst);
        auto upload_end = std::chrono::high_resolution_clock::now();
        QueryExecutor::upload_ms += std::chrono::duration_cast<std::chrono::milliseconds>(upload_end-upload_start).count();

#ifdef TIME_DEBUG
        std::cout << "Full relation upload time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(upload_end-upload_start).count()
             << " ms. (Data size = " << data_end_offst - data_start_offst << ")\n";
#endif

        intermediateResult = fullIr;
    } else {
        assert(false);
    }
}

void SelectQueryJob::startJobCpu() {

    auto l1_bit = thrust::lower_bound(thrust::host, l1_index_values->begin(), l1_index_values->end(), dataid[0]);
    auto start_offst = thrust::distance(l1_index_values->begin(), l1_bit);
    auto end_offst = start_offst + 1;

    auto l1_offst_lower_bound = *(l1_index_offsets->begin() + start_offst);
    auto l1_offst_upper_bound = (end_offst == l1_index_offsets->size())? data->size() :  *(l1_index_offsets->begin() + end_offst);

#ifdef DEBUG
    std::cout << "Search for l2 offset " << l1_offst_lower_bound << " to " << l1_offst_upper_bound << "\n";
#endif

    // TODO: if not exist, l1_bit == l1_index_values->end()

    auto l1_offst_bit = thrust::lower_bound(thrust::host, l2_index_offsets->begin(), l2_index_offsets->end(), l1_offst_lower_bound);
    auto l1_offst_eit = thrust::lower_bound(thrust::host, l2_index_offsets->begin(), l2_index_offsets->end(), l1_offst_upper_bound);
    auto start_offst2 = thrust::distance(l2_index_offsets->begin(), l1_offst_bit);
    auto end_offst2 = thrust::distance(l2_index_offsets->begin(), l1_offst_eit);

    auto l2_begin = l2_index_values->begin() + start_offst2;
    auto l2_end = l2_index_values->begin() + end_offst2;

#ifdef DEBUG
    std::cout << "L1 Index : ";
    for (size_t i = 0; i < l1_index_values->size(); ++i)
        std::cout << *(l1_index_values->begin() + i) << " ";
    std::cout << "\nFind : " << dataid[0] << "\n";
    std::cout << "Lower bound index : " << start_offst << " , value = " << *(l1_index_offsets->begin() + start_offst) << "\n";
    std::cout << "Upper bound index : " << end_offst << " , value = " << *(l1_index_offsets->begin() + end_offst) << "\n";
#endif

    if (getVariableNum() == 1) {

        auto l2_bit = thrust::lower_bound(thrust::host, l2_begin, l2_end, dataid[1]);
        auto l2_eit = thrust::upper_bound(thrust::host, l2_begin, l2_end, dataid[1]);
        auto l2_start_offst = thrust::distance(l2_index_values->begin(), l2_bit);
        auto l2_end_offst = thrust::distance(l2_index_values->begin(), l2_eit);

        // TODO: if not exist, l2_eit == l2_index_values->end()
        size_t data_start_offst = *(l2_index_offsets->begin() + l2_start_offst);
        size_t data_end_offst = *(l2_index_offsets->begin() + l2_end_offst);

#ifdef DEBUG
        std::cout << "L2 Index : ";
        for (size_t i = 0; i < l2_index_values->size(); ++i)
            std::cout << *(l2_index_values->begin() + i) << " ";
        std::cout << "\nFind : " << dataid[1] << "\n";
        std::cout << "Lower bound index : " << l2_start_offst << " , value = " << data_start_offst << "\n";
        std::cout << "Upper bound index : " << l2_end_offst << " , value = " << data_end_offst << "\n";
#endif

        // select from first and second indices
        size_t columnNum = 1;
        size_t relationNum = data_end_offst - data_start_offst; // XXX: not plus 1, end iterator is exclusive
        FullRelationIR *fullIr = new FullRelationIR(columnNum, relationNum);

        fullIr->setHeader(0, variables[0], is_predicates[0]);
        fullIr->setRelation(0, data->begin() + data_start_offst, data->begin() + data_end_offst);

#ifdef DEBUG
        std::cout << "\nLoad Full Relation\n";
        fullIr->print();
#endif
        intermediateResult = fullIr;
    } else if (getVariableNum() == 2) {

        size_t data_start_offst = *(l2_index_offsets->begin() + start_offst2);
        size_t data_end_offst = *(l2_index_offsets->begin() + end_offst2);

        auto index_row_count = end_offst2 - start_offst2;
        vector<size_t> relation_sizes = {data_end_offst - data_start_offst};

        IndexIR *indexIr = new IndexIR(1, index_row_count, relation_sizes, context);
        indexIr->setIndex(variables[0], is_predicates[0], l2_index_values->begin() + start_offst2, l2_index_values->begin() + end_offst2);
        indexIr->setHeader(0, variables[1], is_predicates[1]);
        indexIr->setIndexOffsetAndNormalize(0, l2_index_offsets->begin() + start_offst2, l2_index_offsets->begin() + end_offst2);
        indexIr->setRelationData(0, data->begin() + data_start_offst, data->begin() + data_end_offst);

#ifdef DEBUG
        std::cout << "SELECTED JOB Copy data from " << data_start_offst << " to " << data_end_offst << "\n";
        std::cout << "Data : ";
        for (auto ii = data->begin() + data_start_offst; ii < data->begin() + data_end_offst; ++ii) {
            std::cout << *ii << " ";
        }std::cout << "\n";
#endif

        //std::cout << "\nLoad Indexed Relation\n";
        //indexIr->print();

        intermediateResult = indexIr;
    } else {
        assert(false);
    }
}

IR* SelectQueryJob::getIR() {
    return intermediateResult;
}

std::string SelectQueryJob::getVariable(size_t i) const {
    assert(/*i >= 0 && */i <= 2);
    return this->variables[i];
}

size_t SelectQueryJob::getVariableNum() const {
    return variable_num;
}

TYPEID SelectQueryJob::getId(size_t i) const {
    assert(/*i >= 0 && */i <= 1);
    return this->dataid[i];
}

void SelectQueryJob::print() const {
    std::cout << "\tSELECT JOB - select ";
    if (getVariableNum() == 2) {
        std::cout << "[" << getVariable(0) << "," << getVariable(1) << "] index ";
        std::cout << getId(0) << "(" << l1_index_values << "," << l1_index_offsets << ")\n";
    } else {
        std::cout << "[" << getVariable(0) << "] index ";
        std::cout << getId(0) << "(" << l1_index_values << "," << l1_index_offsets << ") and ";
        std::cout << getId(1) << "(" << l2_index_values << "," << l2_index_offsets << ")\n";
    }
}
