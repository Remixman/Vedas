#ifndef QUERYEXECUTOR_H
#define QUERYEXECUTOR_H

#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <unordered_set>
#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
// #include "ctpl_stl.h"
#include "vedas.h"
#include "EmptyIntervalDict.h"
#include "ExecutionWorker.h"
#include "SparqlQuery.h"
#include "SparqlResult.h"
#include "VedasStorage.h"
#include "QueryJob.h"
#include "QueryPlan.h"
#include "SelectQueryJob.h"
#include "JoinQueryJob.h"
#include "Histogram.h"
#include "PlanTreeNode.h"

struct DataIndex {
    TYPEID_HOST_VEC *data = nullptr;
    TYPEID_HOST_VEC::iterator lower_it, upper_it;
};

class TriplePatternDataIndex {
public:
    TriplePatternDataIndex();
    void updateDataIndex(size_t idx, TYPEID_HOST_VEC * basePtr, TYPEID_HOST_VEC_IT lowerIt, TYPEID_HOST_VEC_IT upperIt);
    // std::tuple<DataIndex> retrieveDataIndex(size_t idx);
    DataIndex retrieveMinLenDataIndex(size_t idx);
private:
    std::map<size_t, std::vector<DataIndex>> tp_data_index; // Map from triple pattern and host pointer tp boundary
};

class QueryExecutor {
public:
    // QueryExecutor(VedasStorage* vedasStorage, ctpl::thread_pool *threadPool, mgpu::standard_context_t* context);
    QueryExecutor(VedasStorage* vedasStorage, ExecutionWorker *worker, mgpu::standard_context_t* context);
    void setGpuIds(const std::vector<int>& gpu_ids);
    void query(SparqlQuery &sq, SparqlResult &sr);

    void updateEmptyIntervalDict(std::string &variable, std::pair<size_t, size_t>& data_offsts, TYPEID_HOST_VEC *data);
    static std::pair<size_t, size_t> findL2OffsetFromL1(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets, TYPEID id1, size_t n);
    static std::pair<size_t, size_t> findDataOffsetFromL2(TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                                          TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound, TYPEID id2, size_t n);
    static std::pair<size_t, size_t> findDataOffsetFromL2(TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound,
                                                          TYPEID id2, size_t n);

    static void updateBoundDict(std::map< std::string, std::pair<TYPEID, TYPEID> > *bound,
                           std::string &variable, TYPEID lowerBound, TYPEID upperBound);

    // static Histogram *objectHistogram, *subjectHistogram;
    static REVERSE_DICTTYPE *r_so_map, *r_p_map, *r_l_map;
    static void initTime();

    static double load_dict_ms;
    static double load_data_ms;
    static double indexing_ms;
    static double upload_ns;
    static double download_ns;
    static double join_ns;
    static double alloc_copy_ns;
    static double swap_index_ns;
    static double eliminate_duplicate_ns;
    static double convert_to_id_ns;
    static double convert_to_iri_ns;
    static double update_db_dict_ns;
    static double update_db_dict2_ns;
    static double scan_to_split_ns;
    static double prescan_extra_ns;
    static double eif_ns;
    static int eif_count;
    static std::vector<ExecuteLogRecord> exe_log;
    static int jobCount;

    static bool ENABLE_FULL_INDEX;
    static bool ENABLE_LITERAL_DICT;
    static bool ENABLE_PREUPLOAD_BOUND_DICT;
    static bool ENABLE_BOUND_DICT_AFTER_JOIN;
    static bool ENABLE_DOUBLE_BOUND_DICT_AFTER_JOIN;
    static bool ENABLE_UPDATE_BOUND_AFTER_JOIN;

private:
    VedasStorage* vedasStorage;
    // ctpl::thread_pool *threadPool;
    ExecutionWorker *worker;
    mgpu::standard_context_t* context;

    int plan_id; // TODO: remove this
    std::map< std::string, std::pair<TYPEID, TYPEID> > variables_bound;
    std::map< std::string, DataIndex > variables_data_index;
    EmptyIntervalDict empty_interval_dict;
    std::set< std::string > selected_variables;
    std::unordered_set< std::string > join_vars;
    std::vector< int > gpu_ids;

    void printBounds() const;
    void createVariableBound(SparqlQuery &sparqlQuery);
    void updateDataBound(TriplePattern *pattern, TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                         TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                         TYPEID_HOST_VEC *l2_data, TYPEID_HOST_VEC *data, std::string &v, TYPEID id1, TYPEID id2,
                         TERM_TYPE term_type);
    void updateL2Bound(TriplePattern *pattern, std::string &var, TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                       TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound,
                       TERM_TYPE ttype1, TERM_TYPE ttype2);

    void squeezeQueryBound(TriplePattern *pattern);
    void squeezeQueryBound1Var(TriplePattern *pattern);
    void squeezeQueryBound2Var(TriplePattern *pattern);

    void estimateRelationSize();

    int postorderTraversal(QueryPlan &plan, PlanTreeNode* root);
    void createPlanExecFromPlanTree(QueryPlan &plan, PlanTreeNode* root);

    SelectQueryJob* createSelectQueryJob(TriplePattern *pattern, std::string index_used = "", std::pair<TYPEID, TYPEID> *bound = nullptr);
    SelectQueryJob* create1VarSelectQueryJob(TriplePattern *pattern, std::string index_used = "", std::pair<TYPEID, TYPEID> *bound = nullptr);
    SelectQueryJob* create2VarSelectQueryJob(TriplePattern *pattern, std::string index_used = "", std::pair<TYPEID, TYPEID> *bound = nullptr);
    SelectQueryJob* create3VarSelectQueryJob(TriplePattern *pattern);
};

#endif
