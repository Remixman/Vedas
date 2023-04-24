#ifndef QUERYPLAN_H
#define QUERYPLAN_H

#include <set>
#include <vector>
#include <string>
#include <unordered_map>
// #include "ctpl_stl.h"
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
    QueryJob* getJob(size_t i);
    void setJoinVariables(std::vector<std::string> variables);

    size_t size() const;
    void execute(SparqlResult &sparqlResult);
    void print() const;
private:
    // ctpl::thread_pool *threadPool;
    ExecutionWorker *worker;
    std::map<std::string, size_t> query_variable_counter;
    std::vector<TriplePattern*> query_patterns;
    std::vector<std::string> join_variables;

    std::vector<QueryJob*> job_queue;
    std::vector<size_t> thread_no_queue;
    std::vector<std::string> select_variables;
    void plan();

    std::unordered_map<std::string, size_t> var_node_map;
    // std::unordered_map<SelectQueryJob*, size_t> select_node_map;
    std::unordered_map<TriplePattern*, size_t> select_node_map;
    std::vector<std::vector<size_t>> plan_adj_list;

    // For logging
    std::vector<unsigned int> join_sizes;
};

#endif // QUERYPLAN_H
