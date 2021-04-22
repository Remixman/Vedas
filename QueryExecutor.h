#ifndef QUERYEXECUTOR_H
#define QUERYEXECUTOR_H

#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
#include "ctpl_stl.h"
#include "vedas.h"
#include "QueryIndex.h"
#include "SparqlQuery.h"
#include "SparqlResult.h"
#include "VedasStorage.h"
#include "QueryJob.h"
#include "QueryPlan.h"
#include "SelectQueryJob.h"
#include "JoinQueryJob.h"

class QueryExecutor {
public:
    QueryExecutor(VedasStorage* vedasStorage, ctpl::thread_pool *threadPool, bool parallel_plan, mgpu::standard_context_t* context, int plan_id);
    void query(SparqlQuery &sq, SparqlResult &sr);

    static void updateBoundDict(std::map< std::string, std::pair<TYPEID, TYPEID> > *bound,
                                std::string &variable, TYPEID lowerBound, TYPEID upperBound);
    static std::pair<size_t, size_t> findL2OffsetFromL1(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets, TYPEID id1, size_t n);
    static std::pair<size_t, size_t> findDataOffsetFromL2(TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                                          TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound, TYPEID id2, size_t n);
    static std::pair<size_t, size_t> findDataOffsetFromL2(TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound,
                                                          TYPEID id2, size_t n);
    static REVERSE_DICTTYPE *r_so_map, *r_p_map, *r_l_map;
    static void initTime();
    static double upload_ms;
    static double download_ns;
    static double join_ns;
    static double alloc_copy_ns;
    static double swap_index_ns;
    static double eliminate_duplicate_ns;
    static std::vector<ExecuteLogRecord> exe_log;
private:
    VedasStorage* vedasStorage;
    ctpl::thread_pool *threadPool;
    mgpu::standard_context_t* context;

    int plan_id; // TODO: remove this
    bool parallel_plan = false;
    std::map< std::string, std::pair<TYPEID, TYPEID> > variables_bound;
    std::set< std::string > selected_variables;

    void printBounds() const;
    void createVariableBound(SparqlQuery &sparqlQuery);
    void updateDataBound(TriplePattern *pattern, TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                         TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                         TYPEID_HOST_VEC *l2_data, TYPEID_HOST_VEC *data, std::string &v, TYPEID id1, TYPEID id2);
    void updateL2Bound(TriplePattern *pattern, std::string &var, TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                       TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound);

    void squeezeQueryBound(TriplePattern *pattern);
    void squeezeQueryBound1Var(TriplePattern *pattern);
    void squeezeQueryBound2Var(TriplePattern *pattern);

    void estimateRelationSize();

    void manualSchedule(QueryPlan &plan, SparqlQuery &sparqlQuery); // XXX: remove this later

    SelectQueryJob* createSelectQueryJob(TriplePattern *pattern, std::string index_used = "", std::pair<TYPEID, TYPEID> *bound = nullptr);
    SelectQueryJob* create1VarSelectQueryJob(TriplePattern *pattern, std::string index_used = "", std::pair<TYPEID, TYPEID> *bound = nullptr);
    SelectQueryJob* create2VarSelectQueryJob(TriplePattern *pattern, std::string index_used = "", std::pair<TYPEID, TYPEID> *bound = nullptr);
    SelectQueryJob* create3VarSelectQueryJob(TriplePattern *pattern);
};

#endif
