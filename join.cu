#include <cstdio>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
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

void initBinarySPO(vector<TYPEID> &subject, vector<TYPEID> &predicates, vector<TYPEID> &objects);
void initSPO(vector<TYPEID> &subject, vector<TYPEID> &predicates, vector<TYPEID> &objects);
// void testJoin();
void entailMultiGPU();
void sortBySubject(TYPEID *d_subjects, TYPEID *d_objects, int n);
void sortByPredicate(TYPEID *d_subjects, TYPEID *d_predicates, int n);
void sortByObject(TYPEID *d_subjects, TYPEID *d_objects, int n);
void transitiveRuleJoin(standard_context_t &context,
  vector<TYPEID> &filtered_subjects, vector<TYPEID> &filtered_objects, TYPEID predicate,
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects);
void rule2Join(standard_context_t &context,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates,
  vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID domain,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects);
void rule2HashJoin(standard_context_t &context,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates,
  vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID domain,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects);
void rule3HashJoin(standard_context_t &context,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates, vector<TYPEID> &outer_objects,
  vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID range,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects);
void hashJoin(standard_context_t &context);
// use hash join
void rule7join(standard_context_t &context,
  vector<TYPEID> &subprop_subjects, vector<TYPEID> &subprop_objects,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates, vector<TYPEID> &outer_objects,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects);

int main() {
  cout << "Start Program\n";
  entailMultiGPU();
  
  return 0;
}

void initBinarySPO(vector<TYPEID> &subjects, vector<TYPEID> &predicates, vector<TYPEID> &objects) {
  TYPEID subject, predicate, object;
  unsigned int domain_num = 0, range_num = 0, type_num = 0, 
    subclassof_num = 0, subprop_num = 0, total_num = 0;

  FILE * triple_f;
  /*if (true) {
    triple_f = fopen("/media/noo/hd2/DATA/dbpedia/dbpedia-3.8-en.tid", "rb");
    cout << "LOAD [DBPEDIA] DATA\n";
    DOMAIN_ID = 11000;
    RANGE_ID = 1190;
    TYPE_ID = 61;
    SUBCLASSOF_ID = 4218;
    SUBPROP_ID = 0;
  } else*/ if (true) {
    triple_f = fopen("/media/noo/hd2/DATA/yago/yago2s-2013-05-08.tid", "rb");
    cout << "LOAD [YAGO2] DATA\n";
    DOMAIN_ID = 17;
    RANGE_ID = 18;
    TYPE_ID = 1;
    SUBCLASSOF_ID = 16;
    SUBPROP_ID = 168;
  } else if (true) {
    triple_f = fopen("/data/B6T/from90/sp2data/sp2-100M.nt-tid", "rb");
    cout << "LOAD [SP2 - 100M] DATA\n";
    DOMAIN_ID = 0;
    RANGE_ID = 0;
    TYPE_ID = 11;
    SUBCLASSOF_ID = 10;
    SUBPROP_ID = 0;
  }

  clock_t load_begin = clock();

  size_t chunk_size = 4096, count;
  TYPEID *data_b = (TYPEID *) malloc(sizeof(TYPEID) * chunk_size * 3);
  while (!feof(triple_f)) {
    count = fread(data_b, sizeof(TYPEID), chunk_size * 3, triple_f);
    for (int k = 0; k < count; k += 3) {
      subject = data_b[k + 0];
      predicate = data_b[k + 1];
      object = data_b[k + 2];

      if (predicate == DOMAIN_ID) ++domain_num;
      else if (predicate == RANGE_ID) ++range_num;
      else if (predicate == TYPE_ID) ++type_num;
      else if (predicate == SUBCLASSOF_ID) ++subclassof_num;
      else if (predicate == SUBPROP_ID) ++subprop_num;

      subjects.push_back(subject);
      predicates.push_back(predicate);
      objects.push_back(object);

      ++total_num;
    }
  }

  clock_t load_end = clock();

  cout << setprecision(10) << fixed;
  cout << "TOTAL TRIPLE : " << total_num << "\n";
  cout << "HAS DOMAIN      : " << domain_num << " OR " << 1.0*domain_num/total_num << "%%\n";
  cout << "HAS RANGE       : " << range_num << " OR " << 1.0*range_num/total_num << "%%\n";
  cout << "HAS TYPE        : " << type_num << " OR " << 1.0*type_num/total_num << "%%\n";
  cout << "HAS SUBCLASSOF  : " << subclassof_num << " OR " << 1.0*subclassof_num/total_num << "%%\n";
  cout << "HAS SUBPROPERTY : " << subprop_num << " OR " << 1.0*subprop_num/total_num << "%%\n";
  cout << setprecision(4) << fixed;
  cout << "Load triple        : " << double(load_end - load_begin) / CLOCKS_PER_SEC << " Sec.\n";
}

void initSPO(vector<TYPEID> &subjects, vector<TYPEID> &predicates, vector<TYPEID> &objects) {
  TYPEID subject, predicate, object;
  unsigned int domain_num = 0, range_num = 0, type_num = 0, 
    subclassof_num = 0, subprop_num = 0, total_num = 0;

  FILE * triple_f;
  triple_f = fopen("/media/noo/hd2/DATA/yago3/yago3.tidx", "r");
  cout << "LOAD [YAGO3] DATA\n";
  DOMAIN_ID = 7277241;   // http://www.w3.org/2000/01/rdf-schema#domain
  RANGE_ID = 7277232;    // http://www.w3.org/2000/01/rdf-schema#range
  TYPE_ID = 629648;     // http://www.w3.org/1999/02/22-rdf-syntax-ns#type
  SUBCLASSOF_ID = 2;    // http://www.w3.org/2000/01/rdf-schema#subClassOf
  SUBPROP_ID = 7277251;  // http://www.w3.org/2000/01/rdf-schema#subPropertyOf

  clock_t load_begin = clock();

  int i = 0;
  while (fscanf(triple_f, "%d%d%d", &subject, &predicate, &object)) {
    if (predicate == DOMAIN_ID) ++domain_num;
    else if (predicate == RANGE_ID) ++range_num;
    else if (predicate == TYPE_ID) ++type_num;
    else if (predicate == SUBCLASSOF_ID) ++subclassof_num;
    else if (predicate == SUBPROP_ID) ++subprop_num;

    subjects.push_back(subject);
    predicates.push_back(predicate);
    objects.push_back(object);

    if (i % 10000000 == 0) {
      cout << "READ " << i << " RECORDS\n";
    } i++;
    if (feof(triple_f)) break;

    ++total_num;
  }

  clock_t load_end = clock();

  cout << setprecision(10) << fixed;
  cout << "TOTAL TRIPLE : " << total_num << "\n";
  cout << "HAS DOMAIN      : " << domain_num << " OR " << 1.0*domain_num/total_num << "%%\n";
  cout << "HAS RANGE       : " << range_num << " OR " << 1.0*range_num/total_num << "%%\n";
  cout << "HAS TYPE        : " << type_num << " OR " << 1.0*type_num/total_num << "%%\n";
  cout << "HAS SUBCLASSOF  : " << subclassof_num << " OR " << 1.0*subclassof_num/total_num << "%%\n";
  cout << "HAS SUBPROPERTY : " << subprop_num << " OR " << 1.0*subprop_num/total_num << "%%\n";
  cout << setprecision(4) << fixed;
  cout << "Load triple        : " << double(load_end - load_begin) / CLOCKS_PER_SEC << " Sec.\n";

  fclose(triple_f);
}

void entailMultiGPU() {

#pragma omp parallel
{
#pragma omp single
{
  standard_context_t context;

  // hashJoin(context);
  
  vector<TYPEID> h_subjects, h_predicates, h_objects;
  vector<TYPEID> h_domain_subjects, h_domain_objects;
  vector<TYPEID> h_range_subjects, h_range_objects;
  vector<TYPEID> h_subprop_subjects, h_subprop_objects;
  vector<TYPEID> h_subclass_subjects, h_subclass_objects;
  initSPO(h_subjects, h_predicates, h_objects);

  // filter and sort with gpu

  clock_t overall_begin = clock();

  // 1. Filter
  int i = 0, j = h_predicates.size()-1;
  int new_size;
  while (i < j) {
    if (h_predicates[i] == SUBCLASSOF_ID) {
      h_subclass_subjects.push_back(h_subjects[i]);
      h_subclass_objects.push_back(h_objects[i]);
      i++;
    } else if (h_predicates[j] == SUBCLASSOF_ID) {
      std::swap(h_subjects[i], h_subjects[j]);
      std::swap(h_predicates[i], h_predicates[j]);
      std::swap(h_objects[i], h_objects[j]);
      h_subclass_subjects.push_back(h_subjects[i]);
      h_subclass_objects.push_back(h_objects[i]);
      i++; j--;
    } else {

      // rule (2) p domain d
      if (h_predicates[j] == DOMAIN_ID) {
        h_domain_subjects.push_back(h_subjects[j]);
        h_domain_objects.push_back(h_objects[j]);
      }
      // rule (3)
      else if (h_predicates[j] == RANGE_ID) {
        h_range_subjects.push_back(h_subjects[j]);
        h_range_objects.push_back(h_objects[j]);
      }
      // rule (5)
      else if (h_predicates[j] == SUBPROP_ID) {
        h_subprop_subjects.push_back(h_subjects[j]);
        h_subprop_objects.push_back(h_objects[j]);
      }

      j--;
    }
  }
  while (h_predicates[i] != SUBCLASSOF_ID) i--;
  new_size = i + 1;
  cout << "FINISH FILTERING, NEW SIZE IS " << new_size << "\n";

  vector<TYPEID> new_subjects(new_size);
  copy(h_subjects.begin(), h_subjects.begin()+new_size, new_subjects.begin());
  vector<TYPEID> new_objects(new_size);
  copy(h_objects.begin(), h_objects.begin()+new_size, new_objects.begin());
  cout << "FINISH COPY NEW MEMORY\n";
  
  clock_t gpu_begin = clock();


  // RULE (5)
  clock_t rule5_begin = clock();
  vector<TYPEID> infered5_subjects_h, infered5_objects_h, infered5_predicate_h;
  transitiveRuleJoin(context, h_subprop_subjects, h_subprop_objects, SUBPROP_ID,
    infered5_subjects_h, infered5_predicate_h, infered5_objects_h);
  // Append result fron rule 5
  h_subprop_subjects.insert(h_subprop_subjects.end(), ALL(infered5_subjects_h));
  h_subprop_objects.insert(h_subprop_subjects.end(), ALL(infered5_objects_h));
  clock_t rule5_end = clock();
  
  // RULE (7)
  clock_t rule7_begin = clock();
  vector<TYPEID> infered7_subjects_h, infered7_predicates_h, infered7_objects_h;
  rule7join(context, h_subprop_subjects, h_subprop_objects,
    h_subjects, h_predicates, h_objects, infered7_subjects_h, infered7_predicates_h, infered7_objects_h);
  // Append result from rule 7
  h_subjects.insert(h_subjects.end(), ALL(infered7_subjects_h));
  h_predicates.insert(h_predicates.end(), ALL(infered7_predicates_h));
  h_objects.insert(h_objects.end(), ALL(infered7_objects_h));
  clock_t rule7_end = clock();  
    
  // RULE (2)
  clock_t rule2_begin = clock();  
  vector<TYPEID> infered2_subjects_h, infered2_predicate_h, infered2_objects_h;
  rule2HashJoin(context, h_subjects, h_predicates, h_domain_subjects, h_domain_objects, DOMAIN_ID,
    infered2_subjects_h, infered2_predicate_h, infered2_objects_h);
  clock_t rule2_end = clock();

  // RULE (3)
  clock_t rule3_begin = clock();
  vector<TYPEID> infered3_subjects_h, infered3_predicate_h, infered3_objects_h;
  rule3HashJoin(context, h_subjects, h_predicates, h_objects, h_range_subjects, h_range_objects, RANGE_ID,
    infered3_subjects_h, infered3_predicate_h, infered3_objects_h);
  clock_t rule3_end = clock();

  // RULE (9)
  

  // RULE (11)
  clock_t rule11_begin = clock();  
  vector<TYPEID> infered11_subjects_h, infered11_predicate_h, infered11_objects_h;
  transitiveRuleJoin(context, h_subclass_subjects, h_subclass_objects, SUBCLASSOF_ID,
    infered11_subjects_h, infered11_predicate_h, infered11_objects_h);
  clock_t rule11_end = clock();

  clock_t gpu_end = clock();

  clock_t overall_end = clock();

  cout << "[RULE (2)] RESULT SIZE   : " << infered2_subjects_h.size() << "\n";
  cout << "[RULE (5)] RESULT SIZE   : " << infered5_subjects_h.size() << "\n";
  cout << "[RULE (7)] RESULT SIZE   : " << infered7_subjects_h.size() << "\n";
  cout << "[RULE (11)] RESULT SIZE  : " << infered11_subjects_h.size() << "\n";
  cout << "[RULE (3)] RESULT SIZE   : " << infered3_subjects_h.size() << "\n";
  cout << "\n";

  cout << "Overall time = " << double(overall_end - overall_begin) / CLOCKS_PER_SEC << "\n";
  cout << "GPU Process time = " << double(gpu_end - gpu_begin) / CLOCKS_PER_SEC << "\n";
  cout << "Rule (2) time  = " << double(rule2_end - rule2_begin) / CLOCKS_PER_SEC << "\n";
  cout << "Rule (5) time  = " << double(rule5_end - rule5_begin) / CLOCKS_PER_SEC << "\n";
  cout << "Rule (7) time  = " << double(rule7_end - rule7_begin) / CLOCKS_PER_SEC << "\n";
  cout << "Rule (3) time  = " << double(rule3_end - rule3_begin) / CLOCKS_PER_SEC << "\n";
  cout << "Rule (11) time = " << double(rule11_end - rule11_begin) / CLOCKS_PER_SEC << "\n";
} // end omp single
} // end omp parallel

}

void hashJoin(standard_context_t &context) {

  vector<TYPEID> keys = {7, 8, 9};
  vector<TYPEID> vals = {107, 108, 109};
  // upload hash key-value to GPU
  mem_t<TYPEID> keys_d = to_mem(keys, context);
  mem_t<TYPEID> vals_d = to_mem(vals, context);

  vector<TYPEID> pred_in(100, 8);
  mem_t<TYPEID> pred_in_d = to_mem(pred_in, context);
  mem_t<TYPEID> pred_out_d(pred_in.size(), context);
  
  int K = keys.size();
  int N = pred_in.size();
  
  // CUDPP
  CUDPPHandle cudpp;
  cudppCreate(&cudpp);
  
  CUDPPHashTableConfig config;
  config.type = CUDPP_BASIC_HASH_TABLE;
  config.kInputSize = K;
  // space factor multiple for the hash table; multiply space_usage by kInputSize to get the actual space allocation in GPU memory. 
  // 1.05 is about the minimum possible to get a working hash table
  config.space_usage = 2.0;
  
  CUDPPHandle hash_handle;
  cudppHashTable(cudpp, &hash_handle, &config);
  
  cudppHashInsert(hash_handle, keys_d.data(), vals_d.data(), K);
  cudppHashRetrieve(hash_handle, pred_in_d.data(), pred_out_d.data(), N);
  
  vector<TYPEID> answer = from_mem(pred_out_d);
  cout << "ANSWER FROM HASH\n";
  for (auto a : answer) {
    cout << a << " ";
  } cout << "\n";
  
  cudppDestroyHashTable(cudpp, hash_handle);
  cudppDestroy(cudpp);
}

void rule7join(standard_context_t &context,
  vector<TYPEID> &subprop_subjects, vector<TYPEID> &subprop_objects,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates, vector<TYPEID> &outer_objects,
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects) {
  
  // cout << "[RULE (7)] OUTER RELATION SIZE : " << outer_subjects.size() << "\n";
  // cout << "[RULE (7)] INNER RELATION SIZE : " << subprop_subjects.size() << "\n";

  // use inner triple (filtered with subprop predicate) as hash key-value
  mem_t<TYPEID> keys_d = to_mem(subprop_subjects, context);
  mem_t<TYPEID> vals_d = to_mem(subprop_objects, context);
  
  int K = subprop_subjects.size();
  
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
  mem_t<TYPEID> pred_in_d(chunk_size, context); // = to_mem(tmp_pred_in, context);  
  mem_t<TYPEID> pred_out_d(chunk_size, context);
  vector<TYPEID> tmp_new_result_pred(chunk_size), tmp_new_result_idx(chunk_size);
  for (int i = 0; i < (outer_predicates.size()+chunk_size-1) / chunk_size; i++) {

    int startChunk = i*chunk_size;
    int endChunk = min((i+1)*chunk_size, (int)outer_predicates.size());
    // cout << "COPY FROM " << startChunk << " TO " << endChunk << "\n";
    int N = endChunk - startChunk;
    
    htod(pred_in_d.data(), outer_predicates.data() + startChunk, N); 

    cudppHashRetrieve(hash_handle, pred_in_d.data(), pred_out_d.data(), N);
    
    thrust::device_ptr<TYPEID> idx_ptr = thrust::device_pointer_cast(triple_idx_d.data());
    thrust::sequence(idx_ptr, idx_ptr+N, startChunk);
    thrust::device_ptr<TYPEID> out_ptr = thrust::device_pointer_cast(pred_out_d.data());

    TYPEID_IDX_PTR_ITERATOR zip_begin = thrust::make_zip_iterator(thrust::make_tuple(idx_ptr, out_ptr));
    TYPEID_IDX_PTR_ITERATOR zip_end = thrust::make_zip_iterator(thrust::make_tuple(idx_ptr+N, out_ptr+N));

    zip_end = thrust::remove_if(zip_begin, zip_end, is_hash_miss());
    int match_size = thrust::distance(zip_begin, zip_end);


    // cout << "MATCH SIZE : " << match_size << "\n";
    // new triple
    if (match_size > 0) {
      dtoh(tmp_new_result_idx, triple_idx_d.data(), match_size);
      dtoh(tmp_new_result_pred, pred_out_d.data(), match_size);

      // TODO: change to GPU scatter
      // TODO: filter for next rule
      for (int j = 0; j < tmp_new_result_idx.size(); ++j) {
        new_subjects.push_back( outer_subjects[ tmp_new_result_idx[j] ] );
        new_predicates.push_back( tmp_new_result_pred[j] );
        new_objects.push_back( outer_objects[ tmp_new_result_idx[j] ] );
      }
    }
  }

  cudppDestroyHashTable(cudpp, hash_handle);
  cudppDestroy(cudpp);
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

void sortByPredicate(TYPEID *d_subjects, TYPEID *d_predicates, int n) {
  sortByObject(d_subjects, d_predicates, n);
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

void rule2Join(standard_context_t &context,
    vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates,
    vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID domain,
    // Output
    vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects
  ) {
  cout << "[RULE (2)] OUTER RELATION SIZE : " << outer_subjects.size() << "\n";
  cout << "[RULE (2)] INNER RELATION SIZE : " << inner_subjects.size() << "\n";

  // copy inner to device
  mem_t<TYPEID> s_subjectd_d = to_mem(inner_subjects, context);
  mem_t<TYPEID> s_objects_d = to_mem(inner_objects, context);

  // TODO: remove this
  vector<TYPEID> outer_subjects2;
  vector<TYPEID> outer_predicates2;
  copy(outer_subjects.begin(), outer_subjects.begin() + 100, outer_subjects2.begin());
  copy(inner_objects.begin(), inner_objects.begin() + 100, outer_predicates2.begin());

  mem_t<TYPEID> p_subject_d = to_mem(outer_subjects2, context);
  mem_t<TYPEID> p_predicate_d = to_mem(outer_predicates2, context);
  cout << "[RULE (2)] FINISH COPY MEMORY TO GPU\n";

  // sort
  sortBySubject(s_subjectd_d.data(), s_objects_d.data(), s_subjectd_d.size());
  sortByPredicate(p_subject_d.data(), p_predicate_d.data(), p_predicate_d.size());
  cout << "[RULE (2)] FINISH SORT\n";

  // inner join
  mem_t<int2> joined = inner_join(
    (TYPEID*)p_predicate_d.data(), p_predicate_d.size(),
    (TYPEID*)s_subjectd_d.data(), s_subjectd_d.size(), 
    less_t<TYPEID>(), context);
  cout << "[RULE (2)] FINISH AFTER JOIN\n";

  // add new rule
  mem_t<TYPEID> infered_subjects_d(joined.size(), context);
  mem_t<TYPEID> infered_objects_d(joined.size(), context);
  const TYPEID* p_subject_ptr = p_subject_d.data();
  const TYPEID* s_object_ptr = s_objects_d.data();
  const int2* joined_ptr = joined.data();
  TYPEID* subjects_device_ptr = infered_subjects_d.data();
  TYPEID* objects_device_ptr = infered_objects_d.data();
  auto add_triple_f = [=] MGPU_DEVICE(int index) {
    subjects_device_ptr[index] = p_subject_ptr[ joined_ptr[index].x ];
    objects_device_ptr[index] = s_object_ptr[ joined_ptr[index].y ];
  };
  transform(add_triple_f, joined.size(), context);

  // copy back
  new_subjects = from_mem(infered_subjects_d);
  new_objects = from_mem(infered_objects_d);
}

void rule2HashJoin(standard_context_t &context,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates,
  vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID domain,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects
) {

  // cout << "[RULE (2)] OUTER RELATION SIZE : " << outer_subjects.size() << "\n";
  // cout << "[RULE (2)] INNER RELATION SIZE : " << inner_subjects.size() << "\n";

  // use inner triple (filtered with subprop predicate) as hash key-value
  mem_t<TYPEID> keys_d = to_mem(inner_subjects, context);
  mem_t<TYPEID> vals_d = to_mem(inner_objects, context);

  int K = inner_subjects.size();
  
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
  for (int i = 0; i < (outer_predicates.size()+chunk_size-1) / chunk_size; i++) {

    int startChunk = i*chunk_size;
    int endChunk = min((i+1)*chunk_size, (int)outer_predicates.size());
    // cout << "COPY FROM " << startChunk << " TO " << endChunk << "\n";
    int N = endChunk - startChunk;
    
    htod(pred_in_d.data(), outer_predicates.data() + startChunk, N); 

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
    if (match_size > 0) {
      dtoh(tmp_new_result_idx, triple_idx_d.data(), match_size);
      dtoh(tmp_new_object, object_out_d.data(), match_size);

      // TODO: change to GPU scatter
      // TODO: filter for next rule
      for (int j = 0; j < tmp_new_result_idx.size(); ++j) {
        new_subjects.push_back( outer_subjects[ tmp_new_result_idx[j] ] );
        new_predicates.push_back( TYPE_ID );
        new_objects.push_back( tmp_new_object[j] );
      }
    }
  }

  cudppDestroyHashTable(cudpp, hash_handle);
  cudppDestroy(cudpp);
}

void rule3Join(standard_context_t &context,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates,
  vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID domain,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects
) {
  cout << "[RULE (3)] OUTER RELATION SIZE : " << outer_subjects.size() << "\n";
  cout << "[RULE (3)] INNER RELATION SIZE : " << inner_subjects.size() << "\n";

  // copy inner to device
  mem_t<TYPEID> s_subjectd_d = to_mem(inner_subjects, context);
  mem_t<TYPEID> s_objects_d = to_mem(inner_objects, context);

  // TODO: remove this
  vector<TYPEID> outer_subjects2;
  vector<TYPEID> outer_predicates2;
  copy(outer_subjects.begin(), outer_subjects.begin() + 10000, outer_subjects2.begin());
  copy(inner_objects.begin(), inner_objects.begin() + 10000, outer_predicates2.begin());

  mem_t<TYPEID> p_subject_d = to_mem(outer_subjects2, context);
  mem_t<TYPEID> p_predicate_d = to_mem(outer_predicates2, context);
  cout << "[RULE (3)] FINISH COPY MEMORY TO GPU\n";

  // sort
  sortBySubject(s_subjectd_d.data(), s_objects_d.data(), s_subjectd_d.size());
  sortByPredicate(p_subject_d.data(), p_predicate_d.data(), p_predicate_d.size());
  cout << "[RULE (3)] FINISH SORT\n";

  // inner join
  mem_t<int2> joined = inner_join(
    (TYPEID*)p_predicate_d.data(), p_predicate_d.size(),
    (TYPEID*)s_subjectd_d.data(), s_subjectd_d.size(), 
    less_t<TYPEID>(), context);
  cout << "[RULE (3)] FINISH AFTER JOIN\n";

  // add new rule
  mem_t<TYPEID> infered_subjects_d(joined.size(), context);
  mem_t<TYPEID> infered_objects_d(joined.size(), context);
  const TYPEID* p_subject_ptr = p_subject_d.data();
  const TYPEID* s_object_ptr = s_objects_d.data();
  const int2* joined_ptr = joined.data();
  TYPEID* subjects_device_ptr = infered_subjects_d.data();
  TYPEID* objects_device_ptr = infered_objects_d.data();
  auto add_triple_f = [=] MGPU_DEVICE(int index) {
    subjects_device_ptr[index] = p_subject_ptr[ joined_ptr[index].x ];
    objects_device_ptr[index] = s_object_ptr[ joined_ptr[index].y ];
  };
  transform(add_triple_f, joined.size(), context);

  // copy back
  new_subjects = from_mem(infered_subjects_d);
  new_objects = from_mem(infered_objects_d);

  cout << "[RULE (3)] RESULT SIZE : " << new_subjects.size() << "\n";
}

void rule3HashJoin(standard_context_t &context,
  vector<TYPEID> &outer_subjects, vector<TYPEID> &outer_predicates, vector<TYPEID> &outer_objects,
  vector<TYPEID> &inner_subjects, vector<TYPEID> &inner_objects, TYPEID range,
  // Output
  vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects
) {
  cout << "[RULE (3)] OUTER RELATION SIZE : " << outer_subjects.size() << "\n";
  cout << "[RULE (3)] INNER RELATION SIZE : " << inner_subjects.size() << "\n";

  // use inner triple (filtered with subprop predicate) as hash key-value
  mem_t<TYPEID> keys_d = to_mem(inner_subjects, context);
  mem_t<TYPEID> vals_d = to_mem(inner_objects, context);

  int K = inner_subjects.size();
  
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
  for (int i = 0; i < (outer_predicates.size()+chunk_size-1) / chunk_size; i++) {

    int startChunk = i*chunk_size;
    int endChunk = min((i+1)*chunk_size, (int)outer_predicates.size());
    // cout << "COPY FROM " << startChunk << " TO " << endChunk << "\n";
    int N = endChunk - startChunk;
    
    htod(pred_in_d.data(), outer_predicates.data() + startChunk, N); 

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
    if (match_size > 0) {
      dtoh(tmp_new_result_idx, triple_idx_d.data(), match_size);
      dtoh(tmp_new_object, object_out_d.data(), match_size);

      // TODO: change to GPU scatter
      // TODO: filter for next rule
      for (int j = 0; j < tmp_new_result_idx.size(); ++j) {
        new_subjects.push_back( outer_objects[ tmp_new_result_idx[j] ] );
        new_predicates.push_back( TYPE_ID );
        new_objects.push_back( tmp_new_object[j] );
      }
    }
  }

  cudppDestroyHashTable(cudpp, hash_handle);
  cudppDestroy(cudpp);
}

/*  sort outer triple by object
    sort inner triple by subject */
void transitiveRuleJoin(standard_context_t &context,
    // Input
    vector<TYPEID> &filtered_subjects, vector<TYPEID> &filtered_objects, TYPEID predicate,
    // Output
    vector<TYPEID> &new_subjects, vector<TYPEID> &new_predicates, vector<TYPEID> &new_objects
  ) {

  // copy to device memory
  mem_t<TYPEID> s_subjectd_d = to_mem(filtered_subjects, context);
  mem_t<TYPEID> s_objects_d = to_mem(filtered_objects, context);
  mem_t<TYPEID> o_subjectd_d = to_mem(filtered_subjects, context);
  mem_t<TYPEID> o_objects_d = to_mem(filtered_objects, context);
  // cout << "FINISH COPY MEMORY TO GPU\n";

  // sort 
  sortBySubject(s_subjectd_d.data(), s_objects_d.data(), s_subjectd_d.size());
  sortByObject(o_subjectd_d.data(), o_objects_d.data(), o_objects_d.size());
  /*if (s_subjectd_d.size() <= 20) {
    cout << "SORT SUBJECT RESULT : \n";
    vector<TYPEID> s_subjectd_h = from_mem(s_subjectd_d);
    vector<TYPEID> s_objects_h = from_mem(s_objects_d);
    for (int i = 0; i < s_subjectd_h.size(); ++i) {
      cout << "(" << s_subjectd_h[i] << "," << s_objects_h[i] << ") ";
    }
    cout << "\n";

    cout << "SORT OBJECT RESULT : \n";
    vector<TYPEID> o_subjectd_h = from_mem(o_subjectd_d);
    vector<TYPEID> o_objects_h = from_mem(o_objects_d);    
    for (int i = 0; i < o_subjectd_h.size(); ++i) {
      cout << "(" << o_subjectd_h[i] << "," << o_objects_h[i] << ") ";
    }
    cout << "\n";
  }*/
  // cout << "FINISH SORTED\n";
  
  // inner join
  mem_t<int2> joined = inner_join(
    (TYPEID*)o_objects_d.data(), o_objects_d.size(),
    (TYPEID*)s_subjectd_d.data(), s_subjectd_d.size(), 
    less_t<TYPEID>(), context);
  /*if (joined.size() <= 50) {
    vector<int2> joined_h = from_mem(joined);
    for (int i = 0; i < joined_h.size(); ++i) {
      cout << "(" << joined_h[i].x << "," << joined_h[i].y << ") ";
    }
    cout << "\n";
  }*/
  cout << "FINISH JOINED\n";
  
  // add new rule
  mem_t<TYPEID> infered_subjects_d(joined.size(), context);
  mem_t<TYPEID> infered_objects_d(joined.size(), context);
  const TYPEID* o_subject_ptr = o_subjectd_d.data();
  const TYPEID* s_object_ptr = s_objects_d.data();
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

  // copy(ALL(infered_subject_h), new_subjects.begin());
  // copy(ALL(infered_object_h), new_objects.begin());
  // new_predicates.resize(joined.size(), predicate);

  cout << "JOIN RESULT SIZE FOR " << predicate << " : " << infered_subject_h.size() << "\n";
}


#if 0
void testJoin() {
  standard_context_t context;

  vector<int> a_host = {1,2,3,4,5,6,7,8};
  vector<int> b_host = {1,2,5,11,12,14,20,7};

  clock_t join_begin = clock();

  mem_t<int> a_device = to_mem(a_host, context);
  mem_t<int> b_device = to_mem(b_host, context);

  mergesort(a_device.data(), a_device.size(), less_t<int>(), context);
  mergesort(b_device.data(), b_device.size(), less_t<int>(), context);

  mem_t<int2> joined = inner_join(
    (int*)a_device.data(), a_device.size(),
    (int*)b_device.data(), b_device.size(), 
    less_t<int>(), context);

  vector<int2> output = from_mem(joined);
  clock_t join_end = clock();

  cout << "Output size = " << output.size() << "\n";
  for (int i = 0; i < output.size(); ++i) {
    cout << "(" << output[i].x << "," << output[i].y << ") ";
  }
  cout << "\n\n";
  for (int i = 0; i < output.size(); ++i) {
    cout << "(" << a_host[output[i].x] << "," << b_host[output[i].y] << ") ";
  }
  cout << "\n\n";

  cout << "Time = " << double(join_end - join_begin) / CLOCKS_PER_SEC << "\n";

}
#endif
