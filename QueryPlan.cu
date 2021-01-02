#include <iostream>
#include <iomanip>
#include <queue>
#include <map>
#include <ctime>
#include <tuple>
#include <chrono>
#include <cassert>
#include "QueryPlan.h"
#include "SelectQueryJob.h"
#include "JoinQueryJob.h"

QueryPlan::QueryPlan(ctpl::thread_pool *threadPool) {
    this->threadPool = threadPool;
}

QueryPlan::~QueryPlan() {
    for (size_t i = 0; i < job_queue.size(); ++i)
        delete job_queue[i];
}

void QueryPlan::pushJob(QueryJob *job) {
    this->job_queue.push_back(job);
}

QueryJob* QueryPlan::getJob(size_t i) {
    return this->job_queue[i];
}

void QueryPlan::pushParallelJobs(std::vector<QueryJob*> parallelJobs) {
    this->parallel_job_queue.push_back(parallelJobs);
}

std::vector<QueryJob*>& QueryPlan::getParallelJobs(size_t i) {
    assert(i < this->parallel_job_queue.size());
    return this->parallel_job_queue[i];
}

void QueryPlan::setJoinVariables(std::vector<std::string> variables) {
    this->join_variables = variables;
}

void QueryPlan::setSelectVariables(std::vector<std::string> variables) {
    this->select_variables = variables;
}

size_t QueryPlan::size() const {
    return this->job_queue.size();
}

size_t QueryPlan::parallelSize() const {
    return this->parallel_job_queue.size();
}

void QueryPlan::execute(SparqlResult &sparqlResult, bool parallel_plan) {

    int threadSize = threadPool->size();
    
    // std::cout << "[Start Execute Query Plan]\n";
    if (parallel_plan) {
        // Parallel run jobs
        int l = 0;
        for (std::vector<QueryJob*> jobs : parallel_job_queue) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<std::future<void>> results(jobs.size());
            for (int j = 0; j < jobs.size(); ++j) {
                QueryJob* job = jobs[j];
                results[j] = threadPool->push([job](int){ job->startJob(); });
            }
            for (int j = 0; j < jobs.size(); ++j) {
                results[j].get();
            }
            auto finish = std::chrono::high_resolution_clock::now();
#ifdef TIME_DEBUG
            std::cout << "Query level " << l << " time : " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << " microseconds\n";
#endif
            l++;
        }

        sparqlResult.setResult(parallel_job_queue.back().back()->getIR());
    } else {
        int i = 0;
        for (auto* job : job_queue) {
            auto start = std::chrono::high_resolution_clock::now();
            job->startJob();
            auto finish = std::chrono::high_resolution_clock::now();

            // Loging join size
            JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
            std::tuple<unsigned, unsigned, unsigned> join_size;
            std::get<0>(join_size) = joinJob->getLeftIRSize();
            std::get<1>(join_size) = joinJob->getRightIRSize();
            std::get<2>(join_size) = dynamic_cast<FullRelationIR*>(joinJob->getIR())->size();

#ifdef TIME_DEBUG
            std::cout << "Query(" << i << ") time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << " microseconds\n"; i++;
#endif
        }

        sparqlResult.setResult((*job_queue.rbegin())->getIR());
    }
}

void QueryPlan::plan() {
    // Create join job from edge
    std::queue<size_t> q;
    std::vector<bool> mark(plan_adj_list.size(), false);
    q.push(0); mark[0] = true;
    while(!q.empty()) {
        size_t idx = q.front(); q.pop();
        for (auto adj_idx: plan_adj_list[idx]) {
            if (!mark[idx]) {

                //plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?x", context));

                q.push(idx);
                mark[idx] = true;
            }
        }
    }
}

void QueryPlan::addQueryPattern(TriplePattern *pattern) {
    this->query_patterns.push_back(pattern);
}

void countVariable(std::map<std::string, size_t> &variable_counter, std::string v) {
    if (variable_counter.count(v))
        variable_counter[v] += 1;
    else
        variable_counter[v] = 1;
}

void QueryPlan::findBestPlan() {

    for (auto pattern : query_patterns) {
        if (pattern->subjectIsVariable()) countVariable(query_variable_counter, pattern->getSubject());
        if (pattern->predicateIsVariable()) countVariable(query_variable_counter, pattern->getPredicate());
        if (pattern->objectIsVariable()) countVariable(query_variable_counter, pattern->getObject());
    }

#ifdef DEBUG
    std::cout << "Variable Count : \n";
#endif
    // Create join node from join variable
    for (auto var : query_variable_counter) {
#ifdef DEBUG
        std::cout << var.first << " : " << var.second << "\n";
#endif
        if (var.second > 1) {
            join_variables.push_back(var.first);
            addJoinPlanNode(var.first);
        }
    }

    // Find triple pattern related to join variables
    for (auto pattern : query_patterns) {
        addSelectPlanNode(pattern);
    }
}

void QueryPlan::addJoinPlanNode(std::string joinVar) {
    plan_adj_list.push_back(std::vector<size_t>());
    size_t idx = plan_adj_list.size() - 1;
    var_node_map[joinVar] = idx;
}

//void QueryPlan::addSelectPlanNode(SelectQueryJob *job) {
//    plan_adj_list.push_back(std::vector<size_t>());
//    size_t idx = plan_adj_list.size() - 1;
//    // select_node_map[job] = idx;

//    for (size_t i = 0; i < job->getVariableNum(); ++i) {
//        size_t var_idx = var_node_map[job->getVariable(i)];
//        plan_adj_list[var_idx].push_back(idx);
//        plan_adj_list[idx].push_back(var_idx);
//    }
//}

void QueryPlan::addSelectPlanNode(TriplePattern *pattern) {
    plan_adj_list.push_back(std::vector<size_t>());
    size_t idx = plan_adj_list.size() - 1;
    select_node_map[pattern] = idx;

    if (pattern->subjectIsVariable() && query_variable_counter[pattern->getSubject()] > 1) {
        size_t var_idx = var_node_map[pattern->getSubject()];
        plan_adj_list[var_idx].push_back(idx);
        plan_adj_list[idx].push_back(var_idx);
    }
    if (pattern->predicateIsVariable() && query_variable_counter[pattern->getPredicate()] > 1) {
        size_t var_idx = var_node_map[pattern->getPredicate()];
        plan_adj_list[var_idx].push_back(idx);
        plan_adj_list[idx].push_back(var_idx);
    }
    if (pattern->objectIsVariable() && query_variable_counter[pattern->getObject()] > 1) {
        size_t var_idx = var_node_map[pattern->getObject()];
        plan_adj_list[var_idx].push_back(idx);
        plan_adj_list[idx].push_back(var_idx);
    }
}

void QueryPlan::print() const {
    std::cout << "=========================== Query Plan ============================\n";
    for (auto* job : job_queue) {
        if (SelectQueryJob *sj = dynamic_cast<SelectQueryJob*>(job)) {
            sj->print();
        } else if (JoinQueryJob *jj = dynamic_cast<JoinQueryJob*>(job)) {
            jj->print();
        } else {
            std::cout << "\t INVALID JOB TYPE\n";
        }
    }
    std::cout << "\n";
}
