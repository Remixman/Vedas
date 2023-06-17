#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
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

#define ADD_TO_TUPLE(i,begin_tuple,end_tuple,relation)({thrust::get<i>(begin_tuple)=relation[i]->begin();thrust::get<i>(end_tuple) = relation[i]->end();})

typedef TYPEID_DEVICE_VEC::iterator DEV_VEC_IT;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT> Tuple2;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple3;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple4;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple5;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple6;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple7;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple8;
typedef thrust::tuple<DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT, DEV_VEC_IT> Tuple9;

typedef thrust::tuple<TYPEID, TYPEID> TupleZip2;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID> TupleZip3;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID> TupleZip4;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip5;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip6;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip7;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip8;
typedef thrust::tuple<TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID, TYPEID> TupleZip9;

struct cmp_first_col2{
    __host__ __device__ bool operator()(const TupleZip2 &lhs, const TupleZip2 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        return (thrust::get<1>(lhs) < thrust::get<1>(rhs));
    }
}; // end compare
struct cmp_first_col3{
    __host__ __device__ bool operator()(const TupleZip3 &lhs, const TupleZip3 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        return (thrust::get<2>(lhs) < thrust::get<2>(rhs));
    }
}; // end compare
struct cmp_first_col4{
    __host__ __device__ bool operator()(const TupleZip4 &lhs, const TupleZip4 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        if (thrust::get<2>(lhs) < thrust::get<2>(rhs)) return true;
        if (thrust::get<2>(lhs) > thrust::get<2>(rhs)) return false;
        return (thrust::get<3>(lhs) < thrust::get<3>(rhs));
    }
}; // end compare
struct cmp_first_col5{
    __host__ __device__ bool operator()(const TupleZip5 &lhs, const TupleZip5 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        if (thrust::get<2>(lhs) < thrust::get<2>(rhs)) return true;
        if (thrust::get<2>(lhs) > thrust::get<2>(rhs)) return false;
        if (thrust::get<3>(lhs) < thrust::get<3>(rhs)) return true;
        if (thrust::get<3>(lhs) > thrust::get<3>(rhs)) return false;
        return (thrust::get<4>(lhs) < thrust::get<4>(rhs));
    }
}; // end compare
struct cmp_first_col6{
    __host__ __device__ bool operator()(const TupleZip6 &lhs, const TupleZip6 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        if (thrust::get<2>(lhs) < thrust::get<2>(rhs)) return true;
        if (thrust::get<2>(lhs) > thrust::get<2>(rhs)) return false;
        if (thrust::get<3>(lhs) < thrust::get<3>(rhs)) return true;
        if (thrust::get<3>(lhs) > thrust::get<3>(rhs)) return false;
        if (thrust::get<4>(lhs) < thrust::get<4>(rhs)) return true;
        if (thrust::get<4>(lhs) > thrust::get<4>(rhs)) return false;
        return (thrust::get<5>(lhs) < thrust::get<5>(rhs));
    }
}; // end compare
struct cmp_first_col7{
    __host__ __device__ bool operator()(const TupleZip7 &lhs, const TupleZip7 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        if (thrust::get<2>(lhs) < thrust::get<2>(rhs)) return true;
        if (thrust::get<2>(lhs) > thrust::get<2>(rhs)) return false;
        if (thrust::get<3>(lhs) < thrust::get<3>(rhs)) return true;
        if (thrust::get<3>(lhs) > thrust::get<3>(rhs)) return false;
        if (thrust::get<4>(lhs) < thrust::get<4>(rhs)) return true;
        if (thrust::get<4>(lhs) > thrust::get<4>(rhs)) return false;
        if (thrust::get<5>(lhs) < thrust::get<5>(rhs)) return true;
        if (thrust::get<5>(lhs) > thrust::get<5>(rhs)) return false;
        return (thrust::get<6>(lhs) < thrust::get<6>(rhs));
    }
}; // end compare
struct cmp_first_col8{
    __host__ __device__ bool operator()(const TupleZip8 &lhs, const TupleZip8 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        if (thrust::get<2>(lhs) < thrust::get<2>(rhs)) return true;
        if (thrust::get<2>(lhs) > thrust::get<2>(rhs)) return false;
        if (thrust::get<3>(lhs) < thrust::get<3>(rhs)) return true;
        if (thrust::get<3>(lhs) > thrust::get<3>(rhs)) return false;
        if (thrust::get<4>(lhs) < thrust::get<4>(rhs)) return true;
        if (thrust::get<4>(lhs) > thrust::get<4>(rhs)) return false;
        if (thrust::get<5>(lhs) < thrust::get<5>(rhs)) return true;
        if (thrust::get<5>(lhs) > thrust::get<5>(rhs)) return false;
        if (thrust::get<6>(lhs) < thrust::get<6>(rhs)) return true;
        if (thrust::get<6>(lhs) > thrust::get<6>(rhs)) return false;
        return (thrust::get<7>(lhs) < thrust::get<7>(rhs));
    }
}; // end compare
struct cmp_first_col9{
    __host__ __device__ bool operator()(const TupleZip9 &lhs, const TupleZip9 &rhs) const {
        if (thrust::get<0>(lhs) < thrust::get<0>(rhs)) return true;
        if (thrust::get<0>(lhs) > thrust::get<0>(rhs)) return false;
        if (thrust::get<1>(lhs) < thrust::get<1>(rhs)) return true;
        if (thrust::get<1>(lhs) > thrust::get<1>(rhs)) return false;
        if (thrust::get<2>(lhs) < thrust::get<2>(rhs)) return true;
        if (thrust::get<2>(lhs) > thrust::get<2>(rhs)) return false;
        if (thrust::get<3>(lhs) < thrust::get<3>(rhs)) return true;
        if (thrust::get<3>(lhs) > thrust::get<3>(rhs)) return false;
        if (thrust::get<4>(lhs) < thrust::get<4>(rhs)) return true;
        if (thrust::get<4>(lhs) > thrust::get<4>(rhs)) return false;
        if (thrust::get<5>(lhs) < thrust::get<5>(rhs)) return true;
        if (thrust::get<5>(lhs) > thrust::get<5>(rhs)) return false;
        if (thrust::get<6>(lhs) < thrust::get<6>(rhs)) return true;
        if (thrust::get<6>(lhs) > thrust::get<6>(rhs)) return false;
        if (thrust::get<7>(lhs) < thrust::get<7>(rhs)) return true;
        if (thrust::get<7>(lhs) > thrust::get<7>(rhs)) return false;
        return (thrust::get<8>(lhs) < thrust::get<8>(rhs));
    }
}; // end compare

struct cmp_first_from_col2{
    __host__ __device__ inline bool operator()(const TupleZip2 &lhs, const TupleZip2 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col3{
    __host__ __device__ inline bool operator()(const TupleZip3 &lhs, const TupleZip3 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col4{
    __host__ __device__ inline bool operator()(const TupleZip4 &lhs, const TupleZip4 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col5{
    __host__ __device__ inline bool operator()(const TupleZip5 &lhs, const TupleZip5 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col6{
    __host__ __device__ inline bool operator()(const TupleZip6 &lhs, const TupleZip6 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col7{
    __host__ __device__ inline bool operator()(const TupleZip7 &lhs, const TupleZip7 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col8{
    __host__ __device__ inline bool operator()(const TupleZip8 &lhs, const TupleZip8 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare
struct cmp_first_from_col9{
    __host__ __device__ inline bool operator()(const TupleZip9 &lhs, const TupleZip9 &rhs) const {
        return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
    }
}; // end compare

void FullRelationIR::sort() {
    switch (relation.size()) {
        case 1:
            {
            }
            break;
        case 2:
            {
                Tuple2 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col2());
            }
            break;
        case 3:
            {
                Tuple3 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col3());
            }
            break;
        case 4:
            {
                Tuple4 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col4());
            }
            break;
        case 5:
            {
                Tuple5 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col5());
            }
            break;
        case 6:
            {
                Tuple6 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col6());
            }
            break;
        case 7:
            {
                Tuple7 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col7());
            }
            break;
        case 8:
            {
                Tuple8 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(7, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col8());
            }
            break;
        case 9:
            {
                Tuple9 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(7, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(8, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col9());
            }
            break;
        default:
            std::cout << "COLUMN NUM : " << relation.size() << "\n";
            assert(false);
    }
}

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
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col2());
                auto new_end = thrust::unique(zip_begin, zip_end);
                auto it_end_tuple = new_end.get_iterator_tuple();
                relation[0]->erase(thrust::get<0>(it_end_tuple), relation[0]->end());
                relation[1]->erase(thrust::get<1>(it_end_tuple), relation[1]->end());
            }
            break;
        case 3:
            {
                Tuple3 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col3());
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
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col4());
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
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col5());
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
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col6());
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
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col7());
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
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(7, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col8());
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
        case 9:
            {
                Tuple9 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(7, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(8, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_col9());
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
                relation[8]->erase(thrust::get<8>(it_end_tuple), relation[8]->end());
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

std::string FullRelationIR::getHeaders(std::string delimitor = " ") const {
    if (headers.size() == 0) return "";
    std::string str = headers[0];
    for (size_t i = 1; i < headers.size(); ++i)
        str += delimitor + headers[i];
    return str;
}

bool FullRelationIR::getIsPredicate(size_t i) const {
    return this->is_predicates[i];
}

void FullRelationIR::setRelation(size_t i, size_t offset, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    thrust::copy(bit, eit, this->relation[i]->begin() + offset);
}

void FullRelationIR::setRelation(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit) {
    thrust::copy(bit, eit, this->relation[i]->begin());
}

void FullRelationIR::setRelation(size_t i, size_t offset, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit) {
    thrust::copy(bit, eit, this->relation[i]->begin() + offset);
}

void FullRelationIR::setRelation(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit) {
    thrust::copy(bit, eit, this->relation[i]->begin());
}

void FullRelationIR::sortByFirstColumn() {
    switch (this->getColumnNum()) {
        case 1:
            {
                thrust::sort(thrust::device, relation[0]->begin(), relation[0]->end());
            }
            break;
        case 2:
            {
                Tuple2 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col2());
            }
            break;
        case 3:
            {
                Tuple3 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col3());
            }
            break;
        case 4:
            {
                Tuple4 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col4());
            }
            break;
        case 5:
            {
                Tuple5 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col5());
            }
            break;
        case 6:
            {
                Tuple6 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col6());
            }
            break;
        case 7:
            {
                Tuple7 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col7());
            }
            break;
        case 8:
            {
                Tuple8 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(7, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col8());
            }
            break;
        case 9:
            {
                Tuple9 begin_tuple, end_tuple;
                ADD_TO_TUPLE(0, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(1, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(2, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(3, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(4, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(5, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(6, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(7, begin_tuple, end_tuple, relation);
                ADD_TO_TUPLE(8, begin_tuple, end_tuple, relation);
                auto zip_begin = thrust::make_zip_iterator(begin_tuple);
                auto zip_end = thrust::make_zip_iterator(end_tuple);
                thrust::sort(thrust::device, zip_begin, zip_end, cmp_first_from_col9());
            }
            break;
        default:
            std::cout << "COLUMN NUM : " << relation.size() << "\n";
            assert(false);
    }
}

void FullRelationIR::swapColumn(size_t i ,size_t j) {
    if (i == j) return;

    TYPEID_DEVICE_VEC* relationTmp = relation[i];
    relation[i] = relation[j];
    relation[j] = relationTmp;

    std::swap(is_predicates[i], is_predicates[j]);
    std::swap(headers[i], headers[j]);

    var_column_map[ headers[i] ] = i;
    var_column_map[ headers[j] ] = j;
}

void FullRelationIR::removeColumn(size_t i) {
    relation.erase(relation.begin() + i);
    is_predicates.erase(is_predicates.begin() + i);
    headers.erase(headers.begin() + i);
    for (size_t k = i; k < headers.size(); ++k) {
        var_column_map[ headers[k] ] = k;
    }
}

void FullRelationIR::removeColumn(size_t i, std::string &maintain_var) {
    this->removeColumn(i);

    if (i == 0) {
        this->swapColumn(0, this->getColumnId(maintain_var));
        this->sortByFirstColumn();
    }
}

void FullRelationIR::movePeer(size_t src_device_id, size_t dest_device_id) {
    size_t columnNum = this->getColumnNum();
    size_t relationSize = this->size();

    if (relationSize == 0) return;

    // TODO: remove test copy
    // cudaSetDevice(src_device_id);
    // std::cout << "Before transfer device : " << src_device_id << "\n";
    // auto copy_size = (relation[0]->size() < 10)? relation[0]->size() : 10;
    // std::cout << "copy_size : " << copy_size << "\n";
    // for (size_t i = 0; i < columnNum; ++i) {
    //     thrust::host_vector<TYPEID> hh(copy_size);
    //     thrust::copy(relation[i]->begin(), relation[i]->begin() + copy_size, hh.begin());
    //     std::cout << "[" << i << "] : ";
    //     for (size_t k = 0; k < copy_size; ++k) {
    //         std::cout << hh[k] << " ";
    //     }
    //     std::cout << '\n';
    // }

    // Select destination device to work with
    cudaSetDevice(dest_device_id);
    std::vector<TYPEID_DEVICE_VEC*> tmpRelation(columnNum);
    // std::cout << "Move " << columnNum << " columns\n";
    // std::cout << "Transfer from " << src_device_id << " to " << dest_device_id << "\n";
    for (size_t i = 0; i < columnNum; ++i) {
        tmpRelation[i] = new TYPEID_DEVICE_VEC(relationSize);

        TYPEID* dest_ptr = thrust::raw_pointer_cast(tmpRelation[i]->data());
        TYPEID* src_ptr = thrust::raw_pointer_cast(relation[i]->data());
        cudaError_t result = cudaMemcpyPeer(dest_ptr, dest_device_id, src_ptr, src_device_id, relationSize * sizeof(TYPEID));
        if (cudaSuccess != result) {
            const char* errorString = cudaGetErrorString(result);
            std::cout << "CUDA error: " << errorString << std::endl;
        }

        relation[i]->clear(); // deallocate 
        relation[i] = tmpRelation[i];
    }
    std::cout << "Finish transfer\n";

    // TODO: remove test copy
    // cudaSetDevice(dest_device_id);
    // std::cout << "After transfer device : " << dest_device_id << "\n";
    // for (size_t i = 0; i < columnNum; ++i) {
    //     thrust::host_vector<TYPEID> hh(copy_size);
    //     thrust::copy(relation[i]->begin(), relation[i]->begin() + copy_size, hh.begin());
    //     std::cout << "[" << i << "] : ";
    //     for (size_t k = 0; k < copy_size; ++k) {
    //         std::cout << hh[k] << " ";
    //     }
    //     std::cout << "\n";
    // }
}

void FullRelationIR::print() const {
    this->print(false, std::cout);
}

void FullRelationIR::print(bool full_version, std::ostream& out) const {
    if (relation.size() == 0) {
        out << "Empty Relation\n";
        return;
    }

    std::vector<TYPEID_HOST_VEC> host_relation(getColumnNum());
    for (size_t i = 0; i < getColumnNum(); ++i) {
        host_relation[i].resize(relation[i]->size());
        thrust::copy(relation[i]->begin(), relation[i]->end(), host_relation[i].begin());
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        out << headers[i] << " ";
    }
    out << "\n";
    if (relation[0]->size() > 50 && !full_version) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t c = 0; c < host_relation.size(); ++c) {
                out << host_relation[c][i] << " ";
            }
            out << "\n";
        }
        out << "...           ...\n";
        out << "... Other " << (relation[0]->size() - 6) << " rows\n";
        out << "...           ...\n";
        for (size_t i = host_relation[0].size() - 3; i < host_relation[0].size(); ++i) {
            for (size_t c = 0; c < host_relation.size(); ++c) {
                out << host_relation[c][i] << " ";
            }
            out << "\n";
        }
    } else {
        for (size_t i = 0; i < host_relation[0].size(); ++i) {
            for (size_t c = 0; c < host_relation.size(); ++c) {
                out << host_relation[c][i] << " ";
            }
            out << "\n";
        }
    }
}

void FullRelationIR::print(REVERSE_DICTTYPE *r_so_map, REVERSE_DICTTYPE *r_p_map, REVERSE_DICTTYPE *r_l_map) const {
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
            } else if (r_l_map->count(host_relation[c][i]) > 0) {
                std::cout << (*r_l_map)[host_relation[c][i]] << " ";
            } else {
                std::cout << (*r_so_map)[host_relation[c][i]] << " ";
            }

        }
        std::cout << "\n";
    }
}
