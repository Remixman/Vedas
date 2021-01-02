#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <ctime>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_bulkremove.hxx>
#include <moderngpu/memory.hxx>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/system_error.h>
#include <cudpp_hash.h>
// #include <omp.h>

#define HASH_MISS_ID 4294967295
#define ALL(v) v.begin(),v.end()

// http://dfukunaga.hatenablog.com/

using namespace mgpu;
using namespace std;

typedef unsigned TYPEID;
typedef uint2 TYPEID2;
TYPEID DOMAIN_ID = 15;
TYPEID RANGE_ID = 16;
TYPEID TYPE_ID = 43;
TYPEID SUBCLASSOF_ID = 17;
TYPEID SUBPROP_ID = 33;
typedef thrust::tuple< thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID> > TYPEID_PTR;
typedef thrust::tuple< TYPEID, TYPEID > TRIPLEID;
typedef thrust::zip_iterator< TYPEID_PTR > TYPEID_PTR_ITERATOR;

typedef thrust::tuple< thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID> > TYPEID_IDX_PTR;
typedef thrust::tuple< TYPEID, TYPEID > TYPEID_IDX;
typedef thrust::zip_iterator< TYPEID_IDX_PTR > TYPEID_IDX_PTR_ITERATOR;

typedef thrust::tuple< thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID>, thrust::device_ptr<TYPEID> > TRIPLE_PTR;
typedef thrust::tuple< TYPEID, TYPEID, TYPEID > TRIPLE;
typedef thrust::zip_iterator< TRIPLE_PTR > TRIPLE_PTR_ITERATOR;


struct cmp_subject {
	__host__ __device__ bool operator()(const TRIPLEID &lhs, const TRIPLEID &rhs) const {
		return (thrust::get<0>(lhs) < thrust::get<0>(rhs));
	}
};
struct cmp_object {
	__host__ __device__ bool operator()(const TRIPLEID &lhs, const TRIPLEID &rhs) const {
		return (thrust::get<1>(lhs) < thrust::get<1>(rhs));
	}
};
struct is_hash_miss {
	__device__ bool operator()(const TYPEID_IDX& t) {
		return (thrust::get<1>(t) == HASH_MISS_ID);
	}
};

void initSPO(vector<TYPEID> &subject, vector<TYPEID> &predicates, vector<TYPEID> &objects);
void sortBySubject(TYPEID *d_subjects, TYPEID *d_objects, int n);
void sortByObject(TYPEID *d_subjects, TYPEID *d_objects, int n);
void cpuHashJoin(
  vector<TYPEID> &outer_subject, vector<TYPEID> &outer_object,
  vector<TYPEID> &inner_subject, vector<TYPEID> &inner_object
);
void hashJoin(standard_context_t &context, 
  vector<TYPEID> &outer_subject, vector<TYPEID> &outer_object,
  vector<TYPEID> &inner_subject, vector<TYPEID> &inner_object
);
void sortMergeJoin(standard_context_t &context, 
  vector<TYPEID> &outer_subject, vector<TYPEID> &outer_object,
  vector<TYPEID> &inner_subject, vector<TYPEID> &inner_object
);
TYPEID randomId(int start, int n) {
  return (rand() % n) + start;
}


int main() {
  standard_context_t context;

  int nn = 30000000;
  int start = 100;
  vector<TYPEID> outer_subject, outer_object;
  vector<TYPEID> inner_subject, inner_object;

  for (int i = 0; i < nn; ++i) {
    outer_subject.push_back( randomId(start, nn * 4) );
    outer_object.push_back( randomId(start, nn * 4) );
    inner_subject.push_back( randomId(start, nn * 4) );
    inner_object.push_back( randomId(start, nn * 4) );
  }

  clock_t cpuStart = clock_t();
  cpuHashJoin(outer_subject, outer_object, inner_subject, inner_object);
  clock_t cpuEnd = clock_t();

  cout << "CPU Hash Join  : " << double(cpuEnd - cpuStart) / CLOCKS_PER_SEC << " Sec.\n";


  // warm up
  clock_t hashStart = clock_t();
  // hashJoin(context, outer_subject, outer_object, inner_subject, inner_object);
  clock_t hashEnd = clock_t();
  
  // cout << "Hash join  : " << double(hashEnd - hashStart) / CLOCKS_PER_SEC << " Sec.\n";

  //
  clock_t smjStart = clock_t();
  sortMergeJoin(context, outer_subject, outer_object, inner_subject, inner_object);
  clock_t smjEnd = clock_t();
  
  cout << "Sort Merge Join  : " << double(smjEnd - smjStart) / CLOCKS_PER_SEC << " Sec.\n";
  

  return 0;
}


void sortBySubject(TYPEID *d_subjects, TYPEID *d_objects, int n) {
  thrust::device_ptr<TYPEID> subject_ptr, object_ptr;
  TYPEID_PTR_ITERATOR zip_begin, zip_end;

  subject_ptr    = thrust::device_pointer_cast(d_subjects);
  object_ptr     = thrust::device_pointer_cast(d_objects);

  zip_begin = thrust::make_zip_iterator(thrust::make_tuple(subject_ptr, object_ptr));
  zip_end = thrust::make_zip_iterator(thrust::make_tuple(subject_ptr+n, object_ptr+n));

  thrust::sort(zip_begin, zip_end, cmp_subject());
}

void sortByObject(TYPEID *d_subjects, TYPEID *d_objects, int n) {
  thrust::device_ptr<TYPEID> subject_ptr, object_ptr;
  TYPEID_PTR_ITERATOR zip_begin, zip_end;

  subject_ptr    = thrust::device_pointer_cast(d_subjects);
  object_ptr     = thrust::device_pointer_cast(d_objects);

  zip_begin = thrust::make_zip_iterator(thrust::make_tuple(subject_ptr, object_ptr));
  zip_end = thrust::make_zip_iterator(thrust::make_tuple(subject_ptr+n, object_ptr+n));

  thrust::sort(zip_begin, zip_end, cmp_object());
}

void hashJoin(standard_context_t &context, 
  vector<TYPEID> &outer_subject, vector<TYPEID> &outer_object,
  vector<TYPEID> &inner_subject, vector<TYPEID> &inner_object
) {
  // copy to device memory
  mem_t<TYPEID> outer_subject_d = to_mem(outer_subject, context);
  mem_t<TYPEID> outer_object_d = to_mem(outer_object, context);
  mem_t<TYPEID> inner_subject_d = to_mem(inner_subject, context);
  mem_t<TYPEID> inner_object_d = to_mem(inner_object, context);

  // use inner triple (filtered with subprop predicate) as hash key-value
  mem_t<TYPEID> keys_d = to_mem(outer_subject, context);
  mem_t<TYPEID> vals_d = to_mem(outer_object, context);

  int K = outer_subject.size();
  
  // CUDPP
  CUDPPHandle cudpp;
  cudppCreate(&cudpp);

  CUDPPHashTableConfig config;
  config.type = CUDPP_BASIC_HASH_TABLE;
  config.kInputSize = K;
  config.space_usage = 2.0;
  
  CUDPPHandle hash_handle;
  cudppHashTable(cudpp, &hash_handle, &config);

  cudppHashInsert(hash_handle, keys_d.data(), vals_d.data(), K);
  
  int chunk_size = 2560 /* core numbers */ * 2000;
  mem_t<TYPEID> triple_idx_d(chunk_size, context);
  mem_t<TYPEID> pred_in_d(chunk_size, context), object_out_d(chunk_size, context);
  vector<TYPEID> tmp_new_object(chunk_size), tmp_new_result_idx(chunk_size);
  for (int i = 0; i < (inner_subject.size()+chunk_size-1) / chunk_size; i++) {

    int startChunk = i*chunk_size;
    int endChunk = min((i+1)*chunk_size, (int)inner_subject.size());
    // cout << "COPY FROM " << startChunk << " TO " << endChunk << "\n";
    int N = endChunk - startChunk;
    
    htod(pred_in_d.data(), inner_subject.data() + startChunk, N); 

    cudppHashRetrieve(hash_handle, pred_in_d.data(), object_out_d.data(), N);
    
    thrust::device_ptr<TYPEID> idx_ptr = thrust::device_pointer_cast(triple_idx_d.data());
    thrust::sequence(idx_ptr, idx_ptr+N, startChunk);
    thrust::device_ptr<TYPEID> out_ptr = thrust::device_pointer_cast(object_out_d.data());

    TYPEID_IDX_PTR_ITERATOR zip_begin = thrust::make_zip_iterator(thrust::make_tuple(idx_ptr, out_ptr));
    TYPEID_IDX_PTR_ITERATOR zip_end = thrust::make_zip_iterator(thrust::make_tuple(idx_ptr+N, out_ptr+N));

    zip_end = thrust::remove_if(zip_begin, zip_end, is_hash_miss());
    int match_size = thrust::distance(zip_begin, zip_end);

    // cout << "MATCH SIZE : " << match_size << "\n";
    // new triple
    //if (match_size > 0) {
      dtoh(tmp_new_result_idx, triple_idx_d.data(), match_size);
      dtoh(tmp_new_object, object_out_d.data(), match_size);

      // TODO: change to GPU scatter
      // TODO: filter for next rule
      //for (int j = 0; j < tmp_new_result_idx.size(); ++j) {
        //new_subjects.push_back( outer_objects[ tmp_new_result_idx[j] ] );
        //new_predicates.push_back( TYPE_ID );
        //new_objects.push_back( tmp_new_object[j] );
      //}
    //}
  }

  cudppDestroyHashTable(cudpp, hash_handle);
  cudppDestroy(cudpp);
}

void sortMergeJoin(standard_context_t &context, 
  vector<TYPEID> &outer_subject, vector<TYPEID> &outer_object,
  vector<TYPEID> &inner_subject, vector<TYPEID> &inner_object
) {
  // copy to device memory
  mem_t<TYPEID> outer_subject_d = to_mem(outer_subject, context);
  mem_t<TYPEID> outer_object_d = to_mem(outer_object, context);
  mem_t<TYPEID> inner_subject_d = to_mem(inner_subject, context);
  mem_t<TYPEID> inner_object_d = to_mem(inner_object, context);

  // sort 
  sortBySubject(inner_subject_d.data(), inner_object_d.data(), inner_subject_d.size());
  sortByObject(outer_subject_d.data(), outer_object_d.data(), outer_object_d.size());

  // inner join
  mem_t<int2> joined = inner_join(
    (TYPEID*)outer_object_d.data(), outer_object_d.size(),
    (TYPEID*)inner_subject_d.data(), inner_subject_d.size(), 
    less_t<TYPEID>(), context);

  // add new rule
  mem_t<TYPEID> infered_subjects_d(joined.size(), context);
  mem_t<TYPEID> infered_objects_d(joined.size(), context);
  const TYPEID* o_subject_ptr = outer_subject_d.data();
  const TYPEID* s_object_ptr = inner_object_d.data();
  const int2* joined_ptr = joined.data();
  TYPEID* subjects_device_ptr = infered_subjects_d.data();
  TYPEID* objects_device_ptr = infered_objects_d.data();
  auto add_triple_f = [=] MGPU_DEVICE(int index) {
    subjects_device_ptr[index] = o_subject_ptr[ joined_ptr[index].x ];
    objects_device_ptr[index] = s_object_ptr[ joined_ptr[index].y ];
  };
  transform(add_triple_f, joined.size(), context);
  // copy back
  vector<TYPEID> infered_subject_h = from_mem(infered_subjects_d);
  vector<TYPEID> infered_object_h = from_mem(infered_objects_d);
}

void cpuHashJoin(
  vector<TYPEID> &outer_subject, vector<TYPEID> &outer_object,
  vector<TYPEID> &inner_subject, vector<TYPEID> &inner_object
) {
  vector<TYPEID> result_subject, result_object;

  map<TYPEID, vector<TYPEID> > dict;
  for (int i = 0; i < outer_subject.size(); ++i) {
    dict[outer_object[i]].push_back(outer_subject[i]);
  }

  for (int i = 0; i < inner_subject.size(); ++i) {
    if (dict.count(inner_subject[i]) > 0) {
      vector<TYPEID> &s = dict[inner_subject[i]];
      for (int k = 0; k < s.size(); ++k) {
        result_subject.push_back(s[k]);
        result_object.push_back(inner_object[i]);
      }
    }
  }
}

