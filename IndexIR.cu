#include <cassert>
#include <iomanip>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include "IndexIR.h"
#include "QueryExecutor.h"

IndexIR::IndexIR(size_t columnNum, size_t indexSize, mgpu::standard_context_t* context)
{
    auto result_alloc_start = std::chrono::high_resolution_clock::now();

    variables.resize(columnNum);
    is_predicates.resize(columnNum);
    indexes_vars.resize(indexSize);

    indexes_offsets.resize(columnNum);
    for (size_t i = 0; i < columnNum; ++i) indexes_offsets[i].resize(indexSize);
    datas.resize(columnNum);

    auto result_alloc_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::alloc_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(result_alloc_end-result_alloc_start).count();

    this->context = context;
}

IndexIR::IndexIR(size_t columnNum, size_t indexSize, std::vector<size_t> relationSizes, mgpu::standard_context_t* context)
{
    assert(columnNum == relationSizes.size());

    auto result_alloc_start = std::chrono::high_resolution_clock::now();

    variables.resize(columnNum);
    is_predicates.resize(columnNum);
    indexes_vars.resize(indexSize);

    indexes_offsets.resize(columnNum);
    for (size_t i = 0; i < columnNum; ++i) indexes_offsets[i].resize(indexSize);
    datas.resize(columnNum);
    for (size_t i = 0; i < columnNum; ++i) datas[i].resize(relationSizes[i]);

    auto result_alloc_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::alloc_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(result_alloc_end-result_alloc_start).count();

    this->context = context;
}

void IndexIR::removeDuplicate() {
    // TODO: Implement
}

void IndexIR::resizeRelation(size_t i, size_t new_size) {
    datas[i].resize(new_size);
}

void IndexIR::resizeRelations(std::vector<size_t> relationSizes) {
    assert(datas.size() == relationSizes.size());

    for (size_t i = 0; i < datas.size(); ++i) datas[i].resize(relationSizes[i]);
}

void IndexIR::setIndexVariable(std::string idx_name, bool is_predicate) {
    this->index_variable = idx_name;
    this->index_is_predicate = is_predicate;
}

void IndexIR::setIndex(std::string &idx_name, bool is_predicate, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    setIndexVariable(idx_name, is_predicate);
    thrust::copy(bit, eit, this->indexes_vars.begin());
}

void IndexIR::setIndexOffset(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    thrust::copy(bit, eit, indexes_offsets[i].begin());
}

void IndexIR::setIndexOffset(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit) {
    thrust::copy(bit, eit, indexes_offsets[i].begin());
}

void IndexIR::setIndexOffsetAndNormalize(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    this->setIndexOffset(i, bit, eit);
    // normalize
    size_t offset_num = thrust::distance(bit, eit);
    TYPEID *offsets_ptr = thrust::raw_pointer_cast(indexes_offsets[i].data());

#ifdef DEBUG
    std::cout << "LOAD INDEX OFFSET TO (" << i << ") : ";
    for (auto it = bit; it != eit; ++it) {
        std::cout << *it << " ";
    }std::cout << "\n";
#endif

    TYPEID first_offset = *bit;
    mgpu::transform([=] __device__(size_t index) {
        offsets_ptr[index] -= first_offset;
    }, offset_num, *context);
}

size_t IndexIR::getColumnNum() const {
    return variables.size();
}

size_t IndexIR::getColumnId(std::string var) const {
    assert(var_column_map.count(var) > 0);
    return var_column_map.at(var);
}

size_t IndexIR::getRelationSize(size_t i) const {
    return datas[i].size();
}

void IndexIR::getRelationPointers(TYPEID** relations) {
    for (size_t i = 0; i < this->datas.size(); ++i) {
        relations[i] = thrust::raw_pointer_cast(this->datas[i].data());
    }
}

void IndexIR::setRelationData(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    thrust::copy(bit, eit, datas[i].begin());
}

void IndexIR::setRelationData(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit) {
    thrust::copy(bit, eit, datas[i].begin());
}

void IndexIR::setHeader(size_t i, std::string var, bool is_predicate) {
    variables[i] = var;
    var_column_map[var] = i;
    is_predicates[i] = is_predicate;
}

std::string IndexIR::getHeader(size_t i) const {
    return variables[i];
}

bool IndexIR::getIsPredicate(size_t i) const {
    return is_predicates[i];
}

std::string IndexIR::getIndexVariable() const {
    return index_variable;
}

bool IndexIR::getIndexIsPredicate() const {
    return index_is_predicate;
}

size_t IndexIR::getIndexSize() const {
    return indexes_vars.size();
}

TYPEID* IndexIR::getIndexVarsRawPointer() {
    return thrust::raw_pointer_cast(indexes_vars.data());
}

TYPEID* IndexIR::getIndexOffsetsRawPtr(size_t i) {
    return thrust::raw_pointer_cast(indexes_offsets[i].data());
}

TYPEID_DEVICE_VEC* IndexIR::getIndexVars() {
    return &indexes_vars;
}

TYPEID_DEVICE_VEC* IndexIR::getIndexOffset(size_t i) {
    return &(indexes_offsets[i]);
}

TYPEID_DEVICE_VEC* IndexIR::getRelationData(size_t i) {
    return &(datas[i]);
}

void IndexIR::calculateOffsetSizeArray(TYPEID_DEVICE_VEC *offsets, TYPEID_DEVICE_VEC *sizes, size_t relation_size) {

    size_t offset_num = offsets->size();
    TYPEID *offsets_ptr = thrust::raw_pointer_cast(offsets->data());
    TYPEID *sizes_ptr = thrust::raw_pointer_cast(sizes->data());

    mgpu::transform([=] __device__(size_t index) {
        bool isLast = index == (offset_num - 1);
        size_t old_offset = offsets_ptr[index];
        size_t next_offset = isLast? relation_size : offsets_ptr[index+1];

        sizes_ptr[index] = next_offset - old_offset;
    }, offset_num, *context);
}

void IndexIR::setRelationRecursive(FullRelationIR *fir, size_t i, size_t pre_k, size_t k, std::vector<TYPEID_HOST_VEC> &acc_size_matrix,
                                   std::vector<TYPEID_HOST_VEC> &index_offst_matrix, size_t relation_offset,
                                   size_t curr_relation_num, size_t prev_relation_num) {

#ifdef DEBUG
    std::cout << "Call set recursive with i(" << i << "), k(" << k << "), relation_offset("
              << relation_offset << "), curr_relation_num(" << curr_relation_num << "), prev_relation_num("
              << prev_relation_num << ")\n";
    std::cout << "   acc_size_matrix(";
    for (size_t t = 0; t < acc_size_matrix.size(); ++t) {
        std::cout << "[";
        for (size_t s = 0; s < acc_size_matrix[t].size(); ++s) {
            std::cout << acc_size_matrix[t][s] << ",";
        }
        std::cout << "],";
    }
    std::cout << ")\n";
    std::cout << "   index_offst_matrix(";
    for (size_t t = 0; t < index_offst_matrix.size(); ++t) {
        std::cout << "[";
        for (size_t s = 0; s < index_offst_matrix[t].size(); ++s) {
            std::cout << index_offst_matrix[t][s] << ",";
        }
        std::cout << "],";
    }
    std::cout << ")\n";
#endif

    size_t begin_idx = index_offst_matrix[i-1][pre_k];
    size_t end_idx = (pre_k + 1 < getIndexOffset(i-1)->size())? index_offst_matrix[i-1][pre_k+1] : getRelationData(i-1)->size();
    size_t var_num = end_idx - begin_idx;
    size_t stride = curr_relation_num / var_num;

    TYPEID *relation = thrust::raw_pointer_cast(fir->getRelationRawPointer(i));
    // TYPEID *offset = thrust::raw_pointer_cast(getIndexOffset(i-1)->data());
    TYPEID *vars = thrust::raw_pointer_cast(getRelationData(i-1)->data()) + begin_idx;

#ifdef DEBUG
    /*TYPEID_HOST_VEC tmp(var_num);
    thrust::copy(getRelationData(i-1)->begin() + static_cast<long>(begin_idx),
                 getRelationData(i-1)->begin() + static_cast<long>(end_idx), tmp.begin());
    std::cout << "\tCopy vars : ";
    for (size_t t = 0; t < var_num; ++t) {
        std::cout << tmp[t] << " ";
    }*/
    std::cout << "\n\tBegin idx : " << begin_idx << " , End idx : " << end_idx << " , VAR NUM : " << var_num << "\n";
    std::cout << "\tFill data from " << relation_offset << " to " << relation_offset + curr_relation_num << " and stride is " << stride << "\n";

#endif

    // set start point
    mgpu::transform([=] __device__(size_t index) {
        relation[relation_offset + index * stride] = vars[index];
    }, var_num, *context);

    // decrease at end point
    mgpu::transform([=] __device__(size_t index) {
        relation[relation_offset + (index+1) * stride] += -vars[index];
    }, var_num - 1, *context);

    // prefix sum
    thrust::inclusive_scan(thrust::device, relation + relation_offset, relation + relation_offset + curr_relation_num,
                           relation + relation_offset/* inplace prefix sum */);
#ifdef DEBUG
    /*std::cout << "\tAfter fill data : (" << relation_offset << "," <<  relation_offset + curr_relation_num<< ") : ";
    TYPEID_HOST_VEC tp(curr_relation_num);
    thrust::copy(fir->getRelation(i)->begin() + relation_offset, fir->getRelation(i)->begin() + relation_offset + prev_relation_num, tp.begin());
    for (auto tt : tp) std::cout << tt << " ";
    std::cout << "\n";*/
#endif

    if (i == getColumnNum()) return;

    size_t sub_offset = relation_offset;
    size_t sub_size = acc_size_matrix[i][k];
    size_t sub_loop_num = curr_relation_num / sub_size;
    for (size_t v1 = 0; v1 < sub_loop_num; ++v1) {
        this->setRelationRecursive(fir, i+1, k, v1, acc_size_matrix, index_offst_matrix, sub_offset, sub_size, curr_relation_num);
        sub_offset += sub_size;
    }
}

void IndexIR::setRelation(FullRelationIR *fir, size_t i, std::vector<TYPEID_HOST_VEC> &acc_size_matrix,
                          std::vector<TYPEID_HOST_VEC> &index_offst_matrix, size_t relation_offst,
                          size_t curr_relation_num, TYPEID *offset, TYPEID *vars, size_t var_size) {
    TYPEID *relation = thrust::raw_pointer_cast(fir->getRelationRawPointer(i));

    // set start point
    mgpu::transform([=] __device__(size_t index) {
        relation[relation_offst + offset[index]] = vars[index];
    }, var_size, *context);

    // decrease at end point
    mgpu::transform([=] __device__(size_t index) {
        relation[relation_offst + offset[index+1]] += -vars[index];
    }, var_size - 1, *context);

    // prefix sum
    thrust::inclusive_scan(thrust::device, relation + relation_offst, relation + relation_offst + curr_relation_num,
                           relation + relation_offst/* inplace prefix sum */);

    size_t sub_offset = relation_offst;
    for (size_t v1 = 0; v1 < indexes_vars.size(); ++v1) {
        size_t sub_size = acc_size_matrix[i][v1];
        this->setRelationRecursive(fir, i+1, v1, v1, acc_size_matrix, index_offst_matrix, sub_offset, sub_size, curr_relation_num);
        sub_offset += sub_size;
    }
}

FullRelationIR* IndexIR::toFullRelationIR() {

    //this->print();
//    std::cout << "Transform Indexed IR => Full IR - relation size (";
//    for (size_t i = 0; i < getColumnNum(); ++i)
//        std::cout << this->getRelationSize(i) << ",";
//    std::cout << ")\n";

    for (size_t i = 0; i < indexes_offsets.size(); ++i) {
        assert(indexes_offsets[i].size() == indexes_vars.size());
    }

    std::vector<TYPEID_HOST_VEC> acc_size_matrix(getColumnNum()); // store multiple accumulate size
    std::vector<TYPEID_HOST_VEC> index_offst_matrix(getColumnNum());

    TYPEID_DEVICE_VEC identity_size_array(indexes_vars.size());
    thrust::fill(thrust::device, identity_size_array.begin(), identity_size_array.end(), 1);
    // Find size of each index pair

    TYPEID_DEVICE_VEC offset_sizes(indexes_vars.size()); // reuse
    for (size_t i = getColumnNum() - 1, l = 0; l < getColumnNum(); --i, ++l) {
        calculateOffsetSizeArray(getIndexOffset(i), &offset_sizes, getRelationSize(i));
//        std::cout << "Offset [" << i << "] : ";
//        for (auto it = indexes_offsets[i].begin(); it != indexes_offsets[i].end(); ++it) std::cout << *it << " ";
//        std::cout << "\n";
//        std::cout << "Offset Size [" << i << "] : ";
//        for (auto it = offset_sizes.begin(); it != offset_sizes.end(); ++it) std::cout << *it << " ";
//        std::cout << "\n";

        thrust::transform(identity_size_array.begin(), identity_size_array.end(),
                          offset_sizes.begin(), identity_size_array.begin(), thrust::multiplies<TYPEID>());

        acc_size_matrix[i].resize(indexes_vars.size());
        thrust::copy(identity_size_array.begin(), identity_size_array.end(), acc_size_matrix[i].begin());
        index_offst_matrix[i].resize(getIndexOffset(i)->size());
        thrust::copy(getIndexOffset(i)->begin(), getIndexOffset(i)->end(), index_offst_matrix[i].begin());
    }
    size_t total_relation_size = thrust::reduce(thrust::device, identity_size_array.begin(), identity_size_array.end());
//    std::cout << "total_relation_size : " << total_relation_size << "\n";

#ifdef DEBUG
    std::cout << "SIZE MATRIX : \n";
    for (size_t ll = 0; ll < getColumnNum(); ll++) {
        for (size_t kk = 0; kk < acc_size_matrix[ll].size(); kk++) {
            std::cout << std::setw(4) << acc_size_matrix[ll][kk];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
#endif

    FullRelationIR *fir = new FullRelationIR(this->getColumnNum() + 1, total_relation_size);
    fir->setHeader(0, this->getIndexVariable(), index_is_predicate);
    for (size_t i = 1; i <= this->getColumnNum(); ++i) {
        fir->setHeader(i, this->getHeader(i-1), is_predicates[i-1]);
    }

//    TYPEID *relation = fir->getRelationRawPointer(0);
//    TYPEID *group_var = this->getIndexVarsRawPointer();

    TYPEID_DEVICE_VEC total_offset(indexes_vars.size());
    thrust::exclusive_scan(thrust::device, identity_size_array.begin(), identity_size_array.end(), total_offset.begin());

    TYPEID *total_offst_ptr = thrust::raw_pointer_cast(total_offset.data());
    this->setRelation(fir, 0, acc_size_matrix, index_offst_matrix, 0, total_relation_size,
                      total_offst_ptr, getIndexVarsRawPointer(), indexes_vars.size());

//    std::cout << "Finish set relation\n";

    return fir;
}

void IndexIR::adjustIndexOffset(TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    if (eit == bit) return;
    TYPEID firstElem = *bit;
    auto adjust_func = [firstElem] __device__ __host__(TYPEID e) {
        return e - firstElem;
    };
    thrust::transform(bit, eit, bit, adjust_func);
}

void IndexIR::print() const {
    if (variables.size() == 0) {
        std::cout << "Empty Relation\n";
        return;
    }

    std::cout << "=========================== Indexed IR ==========================\n";
    std::cout << "Index [" << index_variable << "] : ";
    for (size_t i = 0; i < indexes_vars.size(); ++i) std::cout << indexes_vars[i] << ",";
    std::cout << "\n";

    for (size_t e = 0; e < variables.size(); e++) {
        std::cout << "\t(" << variables[e] << ") : ";
        for (size_t i = 0; i < indexes_offsets[e].size(); ++i) {
            std::cout << indexes_offsets[e][i] << ",";
        }
        std::cout << "\n\t Data : ";
        for (size_t i = 0; i < datas[e].size(); ++i) {
            std::cout << datas[e][i] << ",";
        }
        std::cout << "\n";
    }
}
