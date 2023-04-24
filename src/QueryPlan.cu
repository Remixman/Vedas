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
#include "SelectQueryJob.h"
#include "JoinQueryJob.h"

// QueryPlan::QueryPlan(ctpl::thread_pool *threadPool, std::set<std::string> &select_variable_set) {
QueryPlan::QueryPlan(ExecutionWorker *worker, std::set<std::string> &select_variable_set) {
    // this->threadPool = threadPool;
    this->worker = worker;
    for (std::string var: select_variable_set) {
        this->select_variables.push_back(var);
    }
}

QueryPlan::~QueryPlan() {
    for (size_t i = 0; i < job_queue.size(); ++i)
        delete job_queue[i];
}

void QueryPlan::pushJob(QueryJob *job, size_t thread_no) {
    // thread_no 0 is run on main thread. 1, 2, 3, ... will run on worker threads

    this->job_queue.push_back(job);
    this->thread_no_queue.push_back(thread_no);

    JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
    if (joinJob != nullptr) {
        size_t varCount = this->query_variable_counter[joinJob->getJoinVariable()] + 1;
        this->query_variable_counter[joinJob->getJoinVariable()] = varCount;
    }
}

QueryJob* QueryPlan::getJob(size_t i) {
    return this->job_queue[i];
}

void QueryPlan::setJoinVariables(std::vector<std::string> variables) {
    this->join_variables = variables;
}

size_t QueryPlan::size() const {
    return this->job_queue.size();
}

void QueryPlan::execute(SparqlResult &sparqlResult) {

    for (std::string selVar : select_variables) {
        this->query_variable_counter[selVar] = ULONG_MAX;
    }

    if (worker->size() > 1) {

        // Create promise and future for worker threads
        std::vector<std::promise<int>> workerProms(worker->size());
        std::vector<std::future<int>> mainFuts(worker->size());
        for (size_t i = 0; i < worker->size(); ++i) {
            mainFuts[i] = workerProms[i].get_future();
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < job_queue.size(); ++i) {
            QueryJob *job = job_queue[i];
            size_t thread_no = thread_no_queue[i];

            JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
            if (joinJob != nullptr) {
                joinJob->setQueryVarCounter(&(this->query_variable_counter));
            }


            ExecutionWorker *w = worker;
            std::function<int(void)> f;

            if (MessageJob *mj = dynamic_cast<MessageJob*>(job)) {
                // Send message to ...
                // TODO: select value to send
                int dest = mj->getDestinationThreadId();
                // std::cout << "Send msg from " << thread_no << " to " << dest << "\n";
                f = [thread_no, w, dest]() {
                    w->sendData(thread_no, dest, 999);
                    std::cout << "Send 999 to " << dest << "\n";
                    return 0;
                };
            } else if (WaitMessageJob *wmj = dynamic_cast<WaitMessageJob*>(job)) {
                int src = wmj->getSourceThreadId();
                // std::cout << "Recv msg from " << src << " to " << thread_no << "\n";
                f = [thread_no, w, src]() {
                    w->receiveData(src, thread_no);
                    std::cout << "Receieve from " << src << "\n";
                    return 0;
                };
            } else {
                if (TransferJob *tj = dynamic_cast<TransferJob*>(job)) {
                    tj->setGpuIds(w->getGpuIds());
                }

                // std::cout << "Assign job to " << thread_no << "\n";
                f = [job, w]() { return job->startJob(); };
            }

            std::cout << "Push job (" << job->jobTypeName() << ") to thread " << thread_no << "\n";
            worker->pushTask(thread_no, f);
        }

        // Push task to tell main thread that all jobs are done
        for (size_t i = 0; i < worker->size(); ++i) {
            worker->pushTask(i, [&workerProms, i]() {
                workerProms[i].set_value(-1);
                return 0;
            });
        }

        // Main thread wait for all worker finish
        for (size_t i = 0; i < worker->size(); ++i) {
            std::cout << "Get future from worker " << i << '\n';
            mainFuts[i].get();
        }

        auto finish = std::chrono::high_resolution_clock::now();
        // std::cout << "Query(" << i << ") time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << " microseconds\n";

        // Remove not select variable and duplicate rows
        FullRelationIR *irResult = dynamic_cast<FullRelationIR*>(job_queue.back()->getIR());
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
        // irResult->removeDuplicate();

        sparqlResult.setResult(irResult);
    } else {
        int i = 0;
        for (auto* job : job_queue) {

            JoinQueryJob *joinJob = dynamic_cast<JoinQueryJob*>(job);
            if (joinJob != nullptr) {
                joinJob->setQueryVarCounter(&(this->query_variable_counter));
            }

            auto start = std::chrono::high_resolution_clock::now();
            job->startJob();
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
        FullRelationIR *irResult = dynamic_cast<FullRelationIR*>((*job_queue.rbegin())->getIR());
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

void countVariable(std::map<std::string, size_t> &variable_counter, std::string v) {
    if (variable_counter.count(v))
        variable_counter[v] += 1;
    else
        variable_counter[v] = 1;
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
