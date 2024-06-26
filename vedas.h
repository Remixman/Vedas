#ifndef VEDAS_H
#define VEDAS_H

#include <unordered_map>
#include <map>
#include <string>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_bulkremove.hxx>
#include <moderngpu/memory.hxx>

//#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
//#else
//#define CUDA_CALLABLE_MEMBER
//#endif

#define SCHEMA_SUBJECT 'S'
#define SCHEMA_PREDICATE 'P'
#define SCHEMA_OBJECT 'O'

typedef unsigned TYPEID;
typedef std::unordered_map<std::string, TYPEID> DICTTYPE;
typedef std::unordered_map<TYPEID, std::string> REVERSE_DICTTYPE;

typedef thrust::host_vector<TYPEID> TYPEID_HOST_VEC;
typedef thrust::device_vector<TYPEID> TYPEID_DEVICE_VEC;

typedef thrust::tuple< thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID> > TYPEID_PTR;
typedef thrust::tuple< TYPEID, TYPEID, TYPEID > TRIPLEID;
typedef thrust::zip_iterator< TYPEID_PTR > TYPEID_PTR_ITERATOR;

inline double timeDiff(clock_t b, clock_t e) {
  return double(e-b) / (CLOCKS_PER_SEC / 1000);
}

void sort_merge_join_full_relation(
  mgpu::standard_context_t &context, 
  std::vector<TYPEID> sorted_outer_relation, 
  std::vector<TYPEID> sorted_inner_relation);
void load_rdf(const char *f, std::vector<TYPEID> &s, std::vector<TYPEID> &p, std::vector<TYPEID> &o,
               DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
               REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &l_p_map);
void load_dummy_rdf(std::vector<TYPEID> &s, std::vector<TYPEID> &p, std::vector<TYPEID> &o, DICTTYPE &so_map, DICTTYPE &p_map,
                    REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);

enum ExecuteLogRecordOp { JOIN_OP, UPLOAD_OP, SWAP_OP };

struct ExecuteLogRecord {
  ExecuteLogRecordOp op;
  size_t param1, param2, param3;
  std::string paramstr;

  ExecuteLogRecord(ExecuteLogRecordOp op, std::string paramstr, size_t param1, size_t param2 = 0, size_t param3 = 0) {
    this->op = op;
    this->paramstr = paramstr;
    this->param1 = param1;
    this->param2 = param2;
    this->param3 = param3;
  }
};

#endif
