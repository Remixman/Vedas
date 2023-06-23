#ifndef VEDAS_H
#define VEDAS_H

#include <unordered_map>
#include <map>
#include <tuple>
#include <string>
#include <sstream>
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

#define LITERAL_START_ID 5E7
#define EIF_ALPHA 100

#define HISTOGRAM_EQUAL_WIDTH 0
#define HISTOGRAM_EQUAL_DEPTH 1

#define TYPE_SUBJECT 101
#define TYPE_PREDICATE 102
#define TYPE_OBJECT 103

typedef unsigned TYPEID;
typedef unsigned long long DTYPEID;
typedef int TERM_TYPE;
//typedef std::map<std::string, TYPEID> DICTTYPE;
//typedef std::map<TYPEID, std::string> REVERSE_DICTTYPE;
typedef std::unordered_map<std::string, TYPEID> DICTTYPE;
typedef std::unordered_map<TYPEID, std::string> REVERSE_DICTTYPE;

typedef thrust::host_vector<TYPEID> TYPEID_HOST_VEC;
typedef thrust::device_vector<TYPEID> TYPEID_DEVICE_VEC;
typedef thrust::host_vector<DTYPEID> DTYPEID_HOST_VEC;
typedef thrust::device_vector<DTYPEID> DTYPEID_DEVICE_VEC;

typedef TYPEID_HOST_VEC::iterator TYPEID_HOST_VEC_IT;

typedef thrust::tuple< thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID> > TYPEID_PTR;
typedef thrust::tuple< TYPEID, TYPEID, TYPEID > TRIPLEID;
typedef thrust::zip_iterator< TYPEID_PTR > TYPEID_PTR_ITERATOR;

typedef std::tuple<size_t, size_t, size_t, size_t> VAR_BOUND;
inline size_t VAR_LB(std::tuple<size_t, size_t> b) { return std::get<0>(b); }
inline size_t VAR_UB(std::tuple<size_t, size_t> b) { return std::get<1>(b); }
inline size_t VAR_LB(VAR_BOUND b) { return std::get<0>(b); }
inline size_t VAR_UB(VAR_BOUND b) { return std::get<1>(b); }
inline size_t VAR_LB2(VAR_BOUND b) { return std::get<2>(b); }
inline size_t VAR_UB2(VAR_BOUND b) { return std::get<3>(b); }
inline std::string DOUBLE_BOUND_STRING(VAR_BOUND b) {
    std::stringstream ss;
    ss << '(' << VAR_LB(b) << ',' << VAR_UB(b) << ") and (" << VAR_LB2(b) << ',' << VAR_UB2(b) << ')';
    return ss.str();
}

inline double timeDiff(clock_t b, clock_t e) {
  return double(e-b) / (CLOCKS_PER_SEC / 1000);
}

void sort_merge_join_full_relation(
  mgpu::standard_context_t &context,
  std::vector<TYPEID> sorted_outer_relation,
  std::vector<TYPEID> sorted_inner_relation);
void load_rdf(const char *f, TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
               DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
               REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &l_p_map, bool enable_ldict);
void load_test_rdf(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o, DICTTYPE &so_map, DICTTYPE &p_map,
                   REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);

enum ExecuteLogRecordOp { JOIN_OP, UPLOAD_OP, SWAP_OP };

struct ExecuteLogRecord {
  int deviceId;
  ExecuteLogRecordOp op;
  size_t param1, param2, param3;
  std::string paramstr;

  ExecuteLogRecord(int deviceId, ExecuteLogRecordOp op, std::string paramstr, size_t param1, size_t param2 = 0, size_t param3 = 0) {
    this->deviceId = deviceId;
    this->op = op;
    this->paramstr = paramstr;
    this->param1 = param1;
    this->param2 = param2;
    this->param3 = param3;
  }
};

struct PredicateIndexStat {
    double dist_mean;
    double dist_sd;
};

#endif
