#include <iostream>
#include <cassert>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "FullRelationIR.h"
#include "QueryExecutor.h"

FullRelationIR::FullRelationIR()
{

}

FullRelationIR::FullRelationIR(size_t columnNum, size_t relationSize) {
    auto result_alloc_start = std::chrono::high_resolution_clock::now();

    this->headers.resize(columnNum);
    this->is_predicates.resize(columnNum);
    this->relation.resize(columnNum);
    for (size_t i = 0; i < columnNum; ++i) {
        this->relation[i] = new TYPEID_DEVICE_VEC(relationSize);
    }

    auto result_alloc_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::alloc_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(result_alloc_end-result_alloc_start).count();
}

FullRelationIR::~FullRelationIR()
{
    auto result_alloc_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < this->relation.size(); ++i) {
        delete this->relation[i];
    }
    auto result_alloc_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::alloc_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(result_alloc_end-result_alloc_start).count();
}

typedef TYPEID_DEVICE_VEC::iterator DEV_VEC_IT;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT> Tuple2;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple3;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple4;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple5;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple6;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple7;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple8;

typedef thrust::tuple<TYPEID, TYPEID> TupleZip2;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID> TupleZip3;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID> TupleZip4;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip5;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip6;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip7;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip8;

void FullRelationIR::removeDuplicate() {

    auto eliminate_start = std::chrono::high_resolution_clock::now();
    switch (relation.size()) {
        case 1:
            {
                auto new_end = thrust::unique(relation[0]->begin(), relation[0]->end());
                relation[0]->erase(new_end, relation[0]->end());
            }
            break;
        case 2:
            {
                Tuple2 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
            }
            break;
        case 3:
            {
                Tuple3 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                thrust::get<2>(begin_tuple) = relation[2]->begin();
                thrust::get<2>(end_tuple) = relation[2]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
                relation[2]->erase(thrust::get<2>(it_end_tuple), relation[2]->end());
            }
            break;
        case 4:
            {
                Tuple4 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                thrust::get<2>(begin_tuple) = relation[2]->begin();
                thrust::get<2>(end_tuple) = relation[2]->end();
                thrust::get<3>(begin_tuple) = relation[3]->begin();
                thrust::get<3>(end_tuple) = relation[3]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
                relation[2]->erase(thrust::get<2>(it_end_tuple), relation[2]->end());
                relation[3]->erase(thrust::get<3>(it_end_tuple), relation[3]->end());
            }
            break;
        case 5:
            {
                Tuple5 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                thrust::get<2>(begin_tuple) = relation[2]->begin();
                thrust::get<2>(end_tuple) = relation[2]->end();
                thrust::get<3>(begin_tuple) = relation[3]->begin();
                thrust::get<3>(end_tuple) = relation[3]->end();
                thrust::get<4>(begin_tuple) = relation[4]->begin();
                thrust::get<4>(end_tuple) = relation[4]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
                relation[2]->erase(thrust::get<2>(it_end_tuple), relation[2]->end());
                relation[3]->erase(thrust::get<3>(it_end_tuple), relation[3]->end());
                relation[4]->erase(thrust::get<4>(it_end_tuple), relation[4]->end());
            }
            break;
        case 6:
            {
                Tuple6 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                thrust::get<2>(begin_tuple) = relation[2]->begin();
                thrust::get<2>(end_tuple) = relation[2]->end();
                thrust::get<3>(begin_tuple) = relation[3]->begin();
                thrust::get<3>(end_tuple) = relation[3]->end();
                thrust::get<4>(begin_tuple) = relation[4]->begin();
                thrust::get<4>(end_tuple) = relation[4]->end();
                thrust::get<5>(begin_tuple) = relation[5]->begin();
                thrust::get<5>(end_tuple) = relation[5]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
                relation[2]->erase(thrust::get<2>(it_end_tuple), relation[2]->end());
                relation[3]->erase(thrust::get<3>(it_end_tuple), relation[3]->end());
                relation[4]->erase(thrust::get<4>(it_end_tuple), relation[4]->end());
                relation[5]->erase(thrust::get<5>(it_end_tuple), relation[5]->end());
            }
            break;
        case 7:
            {
                Tuple7 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                thrust::get<2>(begin_tuple) = relation[2]->begin();
                thrust::get<2>(end_tuple) = relation[2]->end();
                thrust::get<3>(begin_tuple) = relation[3]->begin();
                thrust::get<3>(end_tuple) = relation[3]->end();
                thrust::get<4>(begin_tuple) = relation[4]->begin();
                thrust::get<4>(end_tuple) = relation[4]->end();
                thrust::get<5>(begin_tuple) = relation[5]->begin();
                thrust::get<5>(end_tuple) = relation[5]->end();
                thrust::get<6>(begin_tuple) = relation[6]->begin();
                thrust::get<6>(end_tuple) = relation[6]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
                relation[2]->erase(thrust::get<2>(it_end_tuple), relation[2]->end());
                relation[3]->erase(thrust::get<3>(it_end_tuple), relation[3]->end());
                relation[4]->erase(thrust::get<4>(it_end_tuple), relation[4]->end());
                relation[5]->erase(thrust::get<5>(it_end_tuple), relation[5]->end());
                relation[6]->erase(thrust::get<6>(it_end_tuple), relation[6]->end());
            }
            break;
        case 8:
            {
                Tuple8 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[0]->begin();
                thrust::get<0>(end_tuple) = relation[0]->end();
                thrust::get<1>(begin_tuple) = relation[1]->begin();
                thrust::get<1>(end_tuple) = relation[1]->end();
                thrust::get<2>(begin_tuple) = relation[2]->begin();
                thrust::get<2>(end_tuple) = relation[2]->end();
                thrust::get<3>(begin_tuple) = relation[3]->begin();
                thrust::get<3>(end_tuple) = relation[3]->end();
                thrust::get<4>(begin_tuple) = relation[4]->begin();
                thrust::get<4>(end_tuple) = relation[4]->end();
                thrust::get<5>(begin_tuple) = relation[5]->begin();
                thrust::get<5>(end_tuple) = relation[5]->end();
                thrust::get<6>(begin_tuple) = relation[6]->begin();
                thrust::get<6>(end_tuple) = relation[6]->end();
                thrust::get<7>(begin_tuple) = relation[7]->begin();
                thrust::get<7>(end_tuple) = relation[7]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
                relation[2]->erase(thrust::get<2>(it_end_tuple), relation[2]->end());
                relation[3]->erase(thrust::get<3>(it_end_tuple), relation[3]->end());
                relation[4]->erase(thrust::get<4>(it_end_tuple), relation[4]->end());
                relation[5]->erase(thrust::get<5>(it_end_tuple), relation[5]->end());
                relation[6]->erase(thrust::get<6>(it_end_tuple), relation[6]->end());
                relation[7]->erase(thrust::get<7>(it_end_tuple), relation[7]->end());
            }
            break;
        default:
            std::cout << "COLUMN NUM : " << relation.size() << "\n";
            assert(false);
    }
    auto eliminate_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::eliminate_duplicate_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(eliminate_end-eliminate_start).count();
}

size_t FullRelationIR::size() const {
    if (relation.size() == 0) return 0;
    return relation[0]->size();
}

size_t FullRelationIR::getColumnNum() const {
    return headers.size();
}

size_t FullRelationIR::getColumnId(std::string var) const {
    assert(var_column_map.count(var) > 0);
    return var_column_map.at(var);
}

TYPEID* FullRelationIR::getRelationRawPointer(size_t i) {
    return thrust::raw_pointer_cast(relation[i]->data());
}

TYPEID_DEVICE_VEC* FullRelationIR::getRelation(size_t i) {
    return relation[i];
}

void FullRelationIR::getRelationPointers(TYPEID** relations) {
    for (size_t i = 0; i < this->relation.size(); ++i) {
        relations[i] = this->getRelationRawPointer(i);
    }
}

size_t FullRelationIR::getRelationSize(size_t i) const {
    return relation[i]->size();
}

void FullRelationIR::setHeader(size_t i, std::string h, bool is_predicate) {
    this->headers[i] = h;
    var_column_map[h] = i;
    this->is_predicates[i] = is_predicate;
}

std::string FullRelationIR::getHeader(size_t i) const {
    return this->headers[i];
}

bool FullRelationIR::getIsPredicate(size_t i) const {
    return this->is_predicates[i];
}

void FullRelationIR::setRelation(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    thrust::copy(bit, eit, this->relation[i]->begin());
}

void FullRelationIR::setRelation(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit) {
    thrust::copy(bit, eit, this->relation[i]->begin());
}

struct cmp_first_col2{
    __host__ __device__ bool operator()(const TupleZip2 &lhs, const TupleZip2 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_col3{
    __host__ __device__ bool operator()(const TupleZip3 &lhs, const TupleZip3 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_col4{
    __host__ __device__ bool operator()(const TupleZip4 &lhs, const TupleZip4 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_col5{
    __host__ __device__ bool operator()(const TupleZip5 &lhs, const TupleZip5 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_col6{
    __host__ __device__ bool operator()(const TupleZip6 &lhs, const TupleZip6 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_col7{
    __host__ __device__ bool operator()(const TupleZip7 &lhs, const TupleZip7 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare

void FullRelationIR::sortByColumn(size_t i) {
    std::vector<size_t> tuple_idx(relation.size() - 1);
    for (size_t k = 0; k < relation.size(); ++k) {
        if (k != i) tuple_idx.push_back(k);
    }

    switch (relation.size()) {
        case 2:
            {
                Tuple2 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[i]->begin();
                thrust::get<0>(end_tuple) = relation[i]->end();
                thrust::get<1>(begin_tuple) = relation[ tuple_idx[0] ]->begin();
                thrust::get<1>(end_tuple) = relation[ tuple_idx[0] ]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end/*, cmp_first_col2()*/);
            }
            break;
        case 3:
            {
                Tuple3 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[i]->begin();
                thrust::get<0>(end_tuple) = relation[i]->end();
                thrust::get<1>(begin_tuple) = relation[ tuple_idx[0] ]->begin();
                thrust::get<1>(end_tuple) = relation[ tuple_idx[0] ]->end();
                thrust::get<2>(begin_tuple) = relation[ tuple_idx[1] ]->begin();
                thrust::get<2>(end_tuple) = relation[ tuple_idx[1] ]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end/*, cmp_first_col3()*/);
            }
            break;
        case 4:
            {
                Tuple4 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[i]->begin();
                thrust::get<0>(end_tuple) = relation[i]->end();
                thrust::get<1>(begin_tuple) = relation[ tuple_idx[0] ]->begin();
                thrust::get<1>(end_tuple) = relation[ tuple_idx[0] ]->end();
                thrust::get<2>(begin_tuple) = relation[ tuple_idx[1] ]->begin();
                thrust::get<2>(end_tuple) = relation[ tuple_idx[1] ]->end();
                thrust::get<3>(begin_tuple) = relation[ tuple_idx[2] ]->begin();
                thrust::get<3>(end_tuple) = relation[ tuple_idx[2] ]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end/*, cmp_first_col4()*/);
            }
            break;
        case 5:
            {
                Tuple5 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[i]->begin();
                thrust::get<0>(end_tuple) = relation[i]->end();
                thrust::get<1>(begin_tuple) = relation[ tuple_idx[0] ]->begin();
                thrust::get<1>(end_tuple) = relation[ tuple_idx[0] ]->end();
                thrust::get<2>(begin_tuple) = relation[ tuple_idx[1] ]->begin();
                thrust::get<2>(end_tuple) = relation[ tuple_idx[1] ]->end();
                thrust::get<3>(begin_tuple) = relation[ tuple_idx[2] ]->begin();
                thrust::get<3>(end_tuple) = relation[ tuple_idx[2] ]->end();
                thrust::get<4>(begin_tuple) = relation[ tuple_idx[3] ]->begin();
                thrust::get<4>(end_tuple) = relation[ tuple_idx[3] ]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end/*, cmp_first_col5()*/);
            }
            break;
        case 6:
            {
                Tuple6 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[i]->begin();
                thrust::get<0>(end_tuple) = relation[i]->end();
                thrust::get<1>(begin_tuple) = relation[ tuple_idx[0] ]->begin();
                thrust::get<1>(end_tuple) = relation[ tuple_idx[0] ]->end();
                thrust::get<2>(begin_tuple) = relation[ tuple_idx[1] ]->begin();
                thrust::get<2>(end_tuple) = relation[ tuple_idx[1] ]->end();
                thrust::get<3>(begin_tuple) = relation[ tuple_idx[2] ]->begin();
                thrust::get<3>(end_tuple) = relation[ tuple_idx[2] ]->end();
                thrust::get<4>(begin_tuple) = relation[ tuple_idx[3] ]->begin();
                thrust::get<4>(end_tuple) = relation[ tuple_idx[3] ]->end();
                thrust::get<5>(begin_tuple) = relation[ tuple_idx[4] ]->begin();
                thrust::get<5>(end_tuple) = relation[ tuple_idx[4] ]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end/*, cmp_first_col6()*/);
            }
            break;
        case 7:
            {
                Tuple7 begin_tuple, end_tuple;
                thrust::get<0>(begin_tuple) = relation[i]->begin();
                thrust::get<0>(end_tuple) = relation[i]->end();
                thrust::get<1>(begin_tuple) = relation[ tuple_idx[0] ]->begin();
                thrust::get<1>(end_tuple) = relation[ tuple_idx[0] ]->end();
                thrust::get<2>(begin_tuple) = relation[ tuple_idx[1] ]->begin();
                thrust::get<2>(end_tuple) = relation[ tuple_idx[1] ]->end();
                thrust::get<3>(begin_tuple) = relation[ tuple_idx[2] ]->begin();
                thrust::get<3>(end_tuple) = relation[ tuple_idx[2] ]->end();
                thrust::get<4>(begin_tuple) = relation[ tuple_idx[3] ]->begin();
                thrust::get<4>(end_tuple) = relation[ tuple_idx[3] ]->end();
                thrust::get<5>(begin_tuple) = relation[ tuple_idx[4] ]->begin();
                thrust::get<5>(end_tuple) = relation[ tuple_idx[4] ]->end();
                thrust::get<6>(begin_tuple) = relation[ tuple_idx[5] ]->begin();
                thrust::get<6>(end_tuple) = relation[ tuple_idx[5] ]->end();
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end/*, cmp_first_col7()*/);
            }
            break;
        default:
            std::cout << "COLUMN NUM : " << relation.size() << "\n";
            assert(false);
    }
}

void FullRelationIR::swapColumn(size_t i ,size_t j) {
    TYPEID_DEVICE_VEC* relationTmp = relation[i];
    relation[i] = relation[j];
    relation[j] = relationTmp;

    bool boolTmp = is_predicates[i];
    is_predicates[i] = is_predicates[j];
    is_predicates[j] = boolTmp;

    std::string headerTmp = headers[i];
    headers[i] = headers[j];
    headers[j] = headerTmp;

    var_column_map[ headers[i] ] = i;
    var_column_map[ headers[j] ] = j;
}

IndexIR* FullRelationIR::toIndexIR(std::string idx_var) {
    // TODO: check index var is already sort


    // TODO: Transformation implementation
    return nullptr;
}

void FullRelationIR::print() const {
    if (relation.size() == 0) {
        std::cout << "Empty Relation\n";
        return;
    }

    std::vector<TYPEID_HOST_VEC> host_relation(getColumnNum());
    for (size_t i = 0; i < getColumnNum(); ++i) {
        host_relation[i].resize(relation[i]->size());
        thrust::copy(relation[i]->begin(), relation[i]->end(), host_relation[i].begin());
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        std::cout << headers[i] << " ";
    }
    std::cout << "\n";
    for (size_t i = 0; i < host_relation[0].size(); ++i) {
        for (size_t c = 0; c < host_relation.size(); ++c) {
            std::cout << host_relation[c][i] << " ";
        }
        std::cout << "\n";
    }
}

void FullRelationIR::print(REVERSE_DICTTYPE *r_so_map, REVERSE_DICTTYPE *r_p_map) const {
    if (relation.size() == 0) {
        std::cout << "Empty Relation\n";
        return;
    }

    std::vector<TYPEID_HOST_VEC> host_relation(getColumnNum());
    for (size_t i = 0; i < getColumnNum(); ++i) {
        host_relation[i].resize(relation[i]->size());
        thrust::copy(relation[i]->begin(), relation[i]->end(), host_relation[i].begin());
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        std::cout << headers[i] << " ";
    }
    std::cout << "\n";
    for (size_t i = 0; i < host_relation[0].size(); ++i) {
        for (size_t c = 0; c < host_relation.size(); ++c) {
            if (is_predicates[c]) {
                std::cout << (*r_p_map)[host_relation[c][i]] << " ";
            } else {
                std::cout << (*r_so_map)[host_relation[c][i]] << " ";
            }

        }
        std::cout << "\n";
    }
}
