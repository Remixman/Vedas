#include <future>
#include <iostream>
#include <iomanip>
#include <queue>
#include <map>
#include <ctime>
#include <tuple>
#include <chrono>
#include <thread>
#include <cassert>
#include <climits>
#include "QueryPlan.h"
#include "QueryExecutor.h"
#include "SelectQueryJob.h"
#include "JoinQueryJob.h"
#include "IndexSwapJob.h"
#include "BS_thread_pool.hpp"
#include <moderngpu/context.hxx>

QueryPlan::QueryPlan(ExecutionWorker *worker, std::set<std::string> &select_variable_set) {
    this->worker = worker;
    for (std::string var: select_variable_set) {
        this->select_variables.push_back(var);
    }
    this->job_queues.resize(worker->size());
}

QueryPlan::~QueryPlan() {
    for (size_t t = 0; t < job_queues.size(); ++t)
        for (size_t i = 0; i < job_queues[t].size(); ++i)
            delete job_queues[t][i];
}

void QueryPlan::pushJob(QueryJob *job, size_t thread_no) {
    // XXX: thread_no 0 is run on main thread. 1, 2, 3, ... will run on worker threads
    this->job_queues[thread_no].push_back(job);

    JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
    if (joinJob != nullptr) {
        size_t varCount = this->query_variable_counter[joinJob->getJoinVariable()] + 1;
        this->query_variable_counter[joinJob->getJoinVariable()] = varCount;
    }
}

void QueryPlan::pushDynamicJob(QueryJob *job) {
    this->dynamicQueue.push_back(job);

    JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
    if (joinJob != nullptr) {
        size_t varCount = this->query_variable_counter[joinJob->getJoinVariable()] + 1;
        this->query_variable_counter[joinJob->getJoinVariable()] = varCount;
    }
}

QueryJob* QueryPlan::getJob(size_t i, size_t thread_no) {
    return this->job_queues[thread_no][i];
}

void QueryPlan::setJoinVariables(std::vector<std::string> variables) {
    this->join_variables = variables;
}

void QueryPlan::execute(SparqlResult &sparqlResult, bool singleGPU) {

    for (std::string selVar : select_variables) {
        this->query_variable_counter[selVar] = ULONG_MAX;
    }

    if (!singleGPU && worker->size() > 1) {
        jobFinished = 0;
        transferFinished = 0;

        for (size_t t = 0; t < job_queues.size(); ++t) {
            this->worker->getPool()->submit([&, t]() {
                int gpuId = worker->getGpuId(t);
                cudaSetDevice(gpuId);

                std::thread::id this_id = std::this_thread::get_id();
                // std::cout << "Thread " << this_id << "(" << t << ") use GPU " << gpuId << "\n";

                auto jobs_start = std::chrono::high_resolution_clock::now();
                
                // Loop job for each
                for (auto job: job_queues[t]) {
                    JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
                    if (joinJob != nullptr) {
                        joinJob->setQueryVarCounter(&(this->query_variable_counter));
                    }
                    job->startJob(gpuId);
                }
                
                auto jobs_end = std::chrono::high_resolution_clock::now();
                auto jobtime = std::chrono::duration_cast<std::chrono::nanoseconds>(jobs_end-jobs_start).count();
                
                std::cout << "Thread " << this_id << " execute time : " << std::setprecision(3) << jobtime / 1e6 << " ms. ("
                << std::setprecision(8) << jobtime << " ns.)\n";
                
                // TODO: for job in dynamic queue
                for (size_t d = 0; d < dynamicQueue.size(); ++d) {
                    // finish job
                    if (jobFinished == 0) {
                        
                        jobFinished = t+1;

                        auto p2p_start = std::chrono::high_resolution_clock::now();
                        size_t src = t, dest = (t + 1) % 2;

                        // start transfer
                        int srcGPU = worker->getGpuId(src);
                        int destGPU = worker->getGpuId(dest);
                        TransferJob *transferJob = new TransferJob(src, dest, srcGPU, destGPU, job_queues[t][job_queues[t].size() - 1]);
                        job_queues[dest].push_back(transferJob);
                        transferJob->startJob(destGPU);

                        transferFinished = dest + 1;
                        
                        auto p2p_end = std::chrono::high_resolution_clock::now();
                        QueryExecutor::p2p_transfer_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(p2p_end-p2p_start).count();
                    }

                    auto wait_start = std::chrono::high_resolution_clock::now();
                    while (transferFinished == 0); // wait for transfer finish
                    auto wait_end = std::chrono::high_resolution_clock::now();
                    auto wait_time = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end-wait_start).count();
                    std::cout << "Thread " << this_id << " wait time : " << std::setprecision(3) << wait_time / 1e6 << " ms. ("
                            << std::setprecision(8) << wait_time << " ns.)\n";

                    if (t + 1 == transferFinished) {
                        // TODO: last join
                        int jobSize = job_queues[t].size();
                        QueryJob *leftLastJob = job_queues[t][jobSize - 2];
                        QueryJob *rightLastJob = job_queues[t][jobSize - 1];
                        
                        JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(dynamicQueue[d]);
                        joinJob->setOperands(leftLastJob, rightLastJob);
                        joinJob->setQueryVarCounter(&(this->query_variable_counter));
                        joinJob->startJob(gpuId);
                        
                        bool isLastJoin = dynamicQueue.size() - 1 == d;
                        if (isLastJoin) {
                            FullRelationIR *irResult = dynamic_cast<FullRelationIR*>(dynamicQueue[d]->getIR());
                            bool hasRemove = false; // XXX: Use for sort condition
                            for (int i = irResult->getColumnNum() - 1; i >= 1; --i) {
                                if (this->query_variable_counter[irResult->getHeader(i)] == 0) {
                                    irResult->removeColumn(i); hasRemove = true;
                                }
                            }
                            // Sort columns
                            /*for (size_t i = 0; i < select_variables.size(); i++) {
                                size_t ci = irResult->getColumnId(select_variables[i]);
                                if (ci != i) {
                                    irResult->swapColumn(ci, i);
                                }
                            }*/
                            irResult->removeDuplicate();

                            sparqlResult.setResult(irResult);
                        }
                    }

                }

                
            });
        }

        this->worker->getPool()->wait_for_tasks(); // wait for all tasks in thread pool  
    } else {
        int i = 0;
        for (auto job : job_queues[0]) {

            JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
            if (joinJob != nullptr) {
                joinJob->setQueryVarCounter(&(this->query_variable_counter));
            }

            auto start = std::chrono::high_resolution_clock::now();
            job->startJob(worker->getGpuId(0));
            auto finish = std::chrono::high_resolution_clock::now();

            // Loging join size
            if (joinJob != nullptr) {
                std::tuple<unsigned, unsigned, unsigned> join_size;
                std::get<0>(join_size) = joinJob->getLeftIRSize();
                std::get<1>(join_size) = joinJob->getRightIRSize();
                std::get<2>(join_size) = dynamic_cast<FullRelationIR*>(joinJob->getIR())->size();
            }

#ifdef TIME_DEBUG
            std::cout << "Query(" << i << ") time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << " microseconds\n";
#endif
            i++;
        }

        // Remove not select variable and duplicate rows
        FullRelationIR *irResult = dynamic_cast<FullRelationIR*>((*job_queues[0].rbegin())->getIR());
        bool hasRemove = false; // XXX: Use for sort condition
        for (int i = irResult->getColumnNum() - 1; i >= 1; --i) {
            if (this->query_variable_counter[irResult->getHeader(i)] == 0) {
                irResult->removeColumn(i); hasRemove = true;
            }
        }
        // Sort columns
        /*for (size_t i = 0; i < select_variables.size(); i++) {
            size_t ci = irResult->getColumnId(select_variables[i]);
            if (ci != i) {
                irResult->swapColumn(ci, i);
            }
        }*/
        irResult->removeDuplicate();

        sparqlResult.setResult(irResult);
    }
}

void countVariable(std::map<std::string, size_t> &variable_counter, std::string v) {
    if (variable_counter.count(v))
        variable_counter[v] += 1;
    else
        variable_counter[v] = 1;
}

void QueryPlan::print() const {
    std::cout << "=========================== Query Plan ============================\n";
    for (auto job : job_queues[0]) {
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
