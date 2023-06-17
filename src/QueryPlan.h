#ifndef QUERYPLAN_H
#define QUERYPLAN_H

#include <set>
#include <vector>
#include <string>
#include <unordered_map>
#include "QueryJob.h"
#include "ExecutionWorker.h"
#include "SelectQueryJob.h"
#include "TransferJob.h"
#include "SparqlResult.h"
#include "TriplePattern.h"

class QueryPlan
{
public:
    // QueryPlan(ctpl::thread_pool *threadPool, std::set<std::string> &select_variable_set);
    QueryPlan(ExecutionWorker *worker, std::set<std::string> &select_variable_set);
    ~QueryPlan();
    void pushJob(QueryJob *job, size_t thread_no = 0);
    void pushDynamicJob(QueryJob *job);
    QueryJob* getJob(size_t i, size_t thread_no = 0);
    void setJoinVariables(std::vector<std::string> variables);

    void execute(SparqlResult &sparqlResult, bool singleGPU);
    void print() const;
private:
    // ctpl::thread_pool *threadPool;
    ExecutionWorker *worker;
    std::map<std::string, size_t> query_variable_counter;
    std::vector<TriplePattern*> query_patterns;
    std::vector<std::string> join_variables;

    std::vector<std::vector<QueryJob*>> job_queues;
    std::vector<QueryJob*> dynamicQueue;
    std::vector<std::string> select_variables;

    std::unordered_map<std::string, size_t> var_node_map;
    // std::unordered_map<SelectQueryJob*, size_t> select_node_map;
    std::unordered_map<TriplePattern*, size_t> select_node_map;
    std::vector<std::vector<size_t>> plan_adj_list;

    // For logging
    std::vector<unsigned int> join_sizes;

    std::atomic<int> jobFinished;
    std::atomic<int> transferFinished;
};

#endif // QUERYPLAN_H
