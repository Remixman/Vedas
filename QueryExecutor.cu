#include <algorithm>
#include <vector>
#include <cassert>
#include <string>
#include <chrono>
#include <iomanip>
#include <utility>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "ExecutionPlanTree.h"
#include "QueryExecutor.h"
#include "QueryPlan.h"
#include "JoinQueryJob.h"
#include "JoinGraph/JoinGraph.h"

REVERSE_DICTTYPE *QueryExecutor::r_so_map;
REVERSE_DICTTYPE *QueryExecutor::r_p_map;
double QueryExecutor::upload_ms;
double QueryExecutor::join_ns;
double QueryExecutor::alloc_copy_ns;
double QueryExecutor::download_ns;
double QueryExecutor::swap_index_ns;
double QueryExecutor::eliminate_duplicate_ns;

QueryExecutor::QueryExecutor(VedasStorage *vedasStorage, ctpl::thread_pool *threadPool, bool parallel_plan, mgpu::standard_context_t* context, int plan_id) {
    this->vedasStorage = vedasStorage;
    this->threadPool = threadPool;
    this->parallel_plan = parallel_plan;
    this->context = context;
    this->plan_id = plan_id;
}

void QueryExecutor::query(SparqlQuery &sparqlQuery, SparqlResult &sparqlResult) {
    QueryPlan plan(threadPool);

    auto planing_start = std::chrono::high_resolution_clock::now();

    // TODO: 0 pattern ?
    if (sparqlQuery.getPatternNum() == 1) {
        plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
        plan.print();
        plan.execute(sparqlResult, false);

        return;
    }

    // Copy selected variables for filter
    selected_variables = sparqlQuery.getSelectedVariables();

    createVariableBound(sparqlQuery);
    printBounds();
    estimateRelationSize();

#if 1
    JoinGraph joinGraph(&sparqlQuery); // Construct the join graph
    ExecutionPlanTree *planTree = joinGraph.createPlan();
    std::cout << "====================================================\n";
    std::cout << "|               Execution Plan Tree                |\n";
    std::cout << "====================================================\n";
    planTree->printSequentialOrder();
    std::cout << "Length : " << planTree->getNodeList().size() << "\n";
    std::cout << "====================================================\n";
    for (auto node : planTree->getNodeList()) {
        if (node->planOp == UPLOAD) {
            plan.pushJob(this->createSelectQueryJob(node->tp, node->index));
        } else if (node->planOp == JOIN) {
            QueryJob* job1 = plan.getJob(node->children[0]->order - 1);
            QueryJob* job2 = plan.getJob(node->children[1]->order - 1);
            plan.pushJob(new JoinQueryJob(job1, job2, node->joinVariable, context, &variables_bound));
        }
    }
#else
    switch (plan_id) {
        case 1: // S1

            // Parallel job
            if (parallel_plan) {
                std::cout << "Parallel Jobs\n";
                std::vector<QueryJob*> firstLevelJobs;
                for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) 
                    firstLevelJobs.push_back(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
                std::vector<QueryJob*> secondLevelJobs;
                secondLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[0], firstLevelJobs[2], "?v0", context, &variables_bound));
                secondLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[3], firstLevelJobs[4], "?v0", context, &variables_bound));
                secondLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[5], firstLevelJobs[6], "?v0", context, &variables_bound));
                secondLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[7], firstLevelJobs[8], "?v0", context, &variables_bound));
                std::vector<QueryJob*> thirdLevelJobs;
                thirdLevelJobs.push_back(new JoinQueryJob(secondLevelJobs[0], secondLevelJobs[1], "?v0", context, &variables_bound));
                thirdLevelJobs.push_back(new JoinQueryJob(secondLevelJobs[2], secondLevelJobs[3], "?v0", context, &variables_bound));
                std::vector<QueryJob*> forthLevelJobs;
                forthLevelJobs.push_back(new JoinQueryJob(thirdLevelJobs[0], thirdLevelJobs[1], "?v0", context, &variables_bound));
                std::vector<QueryJob*> fifthLevelJobs;
                fifthLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[1], forthLevelJobs[0], "?v0", context, &variables_bound));
                
                plan.pushParallelJobs(firstLevelJobs);
                plan.pushParallelJobs(secondLevelJobs);
                plan.pushParallelJobs(thirdLevelJobs);
                plan.pushParallelJobs(forthLevelJobs);
                plan.pushParallelJobs(fifthLevelJobs);
            } else {
                // Sequential job
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1))); // 0
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4))); // 1
                plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound)); // 2
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5))); // 3
                plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound)); // 4
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(6))); // 5
                plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound)); // 6
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0))); // 7
                plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(7), "?v0", context, &variables_bound)); // 8
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2))); // 9
                plan.pushJob(new JoinQueryJob(plan.getJob(8), plan.getJob(9), "?v0", context, &variables_bound)); // 10
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3))); // 11
                plan.pushJob(new JoinQueryJob(plan.getJob(10), plan.getJob(11), "?v0", context, &variables_bound)); // 12
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(7))); // 13
                plan.pushJob(new JoinQueryJob(plan.getJob(12), plan.getJob(13), "?v0", context, &variables_bound)); // 14
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(8))); // 15
                plan.pushJob(new JoinQueryJob(plan.getJob(14), plan.getJob(15), "?v0", context, &variables_bound));
            }
            break;
        case 2: // S2
            if (parallel_plan) {
                std::vector<QueryJob*> firstLevelJobs;
                for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i)
                    firstLevelJobs.push_back(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
                std::vector<QueryJob*> secondLevelJobs;
                secondLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[0], firstLevelJobs[2], "?v0", context, &variables_bound));
                secondLevelJobs.push_back(new JoinQueryJob(firstLevelJobs[3], firstLevelJobs[1], "?v0", context, &variables_bound));
                std::vector<QueryJob*> thirdLevelJobs;
                thirdLevelJobs.push_back(new JoinQueryJob(secondLevelJobs[0], secondLevelJobs[1], "?v0", context, &variables_bound));
                plan.pushParallelJobs(firstLevelJobs);
                plan.pushParallelJobs(secondLevelJobs);
                plan.pushParallelJobs(thirdLevelJobs);
            } else {
                for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
                plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(2), "?v0", context, &variables_bound));
                plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(3), "?v0", context, &variables_bound));
                plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound));
            }
            break;
        case 3: // S3
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(2), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(3), plan.getJob(4), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(5), "?v0", context, &variables_bound));
            break;
        case 4: // S4
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2), "POS"));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound));
            break;
        case 5: // S5
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(3), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(2), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound));
            break;
        case 6: // S6
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(2), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(3), "?v0", context, &variables_bound));
            break;
        case 7: // S7
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2), "POS"));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(2), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(3), "?v0", context, &variables_bound));
            break;
        case 8: // L1
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0))); // 0
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2))); // 1
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound)); // 2
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1))); // 3
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v2", context, &variables_bound));
            break;
        case 9: // L2
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(2), "?v2", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(3), "?v1", context, &variables_bound));
            break;
        case 10: // L3
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(0), "?v0", context, &variables_bound));
            break;
        case 11: // L4
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            break;
        case 12: // L5
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(2), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(3), "?v3", context, &variables_bound));
            break;
        case 13: // C1
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound)); // 4
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound)); // 5
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound)); // 6
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4))); // 7
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5))); // 8
            plan.pushJob(new JoinQueryJob(plan.getJob(7), plan.getJob(8), "?v4", context, &variables_bound)); // 9
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(9), "?v4", context, &variables_bound)); // 10
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(6))); // 11
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(7))); // 12
            plan.pushJob(new JoinQueryJob(plan.getJob(11), plan.getJob(12), "?v7", context, &variables_bound)); // 13
            plan.pushJob(new JoinQueryJob(plan.getJob(10), plan.getJob(13), "?v6", context, &variables_bound)); // 14
            break;
        case 14: // C2
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1), "POS")); // 0
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2))); // 1
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3))); // 2
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v2", context, &variables_bound)); // 3
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v2", context, &variables_bound)); // 4
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0))); // 5
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound)); // 6
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4))); // 7
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5))); // 8
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(6))); // 9
            plan.pushJob(new JoinQueryJob(plan.getJob(7), plan.getJob(8), "?v4", context, &variables_bound)); // 10
            plan.pushJob(new JoinQueryJob(plan.getJob(9), plan.getJob(10), "?v4", context, &variables_bound)); // 11
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(8), "POS")); // 12
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(9))); // 13
            plan.pushJob(new JoinQueryJob(plan.getJob(12), plan.getJob(13), "?v8", context, &variables_bound)); // 14
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(7))); // 15
            plan.pushJob(new JoinQueryJob(plan.getJob(11), plan.getJob(15), "?v7", context, &variables_bound)); // 16
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(14), "?v3", context, &variables_bound)); // 17
            plan.pushJob(new JoinQueryJob(plan.getJob(16), plan.getJob(17), "?v3", context, &variables_bound));
            break;
        case 15: // C3
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(7), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(8), plan.getJob(9), "?v0", context, &variables_bound));
            break;
        case 16: // F1
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v3", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v3", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(7), plan.getJob(8), "?v3", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(9), "?v0", context, &variables_bound));
            break;
        case 17: // F2
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(7), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(8), plan.getJob(9), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(10), plan.getJob(11), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(5), plan.getJob(6), "?v1", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(12), plan.getJob(13), "?v1", context, &variables_bound));
            break;
        case 18: // F3
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3), "POS"));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5), "POS"));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(5), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(7), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(3), plan.getJob(4), "?v5", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(8), plan.getJob(9), "?v5", context, &variables_bound));
            break;
        case 19: // F4
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1), "POS"));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(6)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(7)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(8), "POS"));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(8), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(9), plan.getJob(10), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(11), plan.getJob(12), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(7), "?v1", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(5), plan.getJob(14), "?v1", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(13), plan.getJob(15), "?v1", context, &variables_bound));
            break;
        case 20: // F5
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1), "POS"));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(7), "?v0", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?v1", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(8), plan.getJob(9), "?v1", context, &variables_bound));
            break;
        // DBpedia
        case 21: // q1
        case 22: // q2
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            break;
        case 23: // q3
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1), "POS"));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?season", context, &variables_bound));
            break;
        case 24: // q4
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?subject", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?subject", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?subject", context, &variables_bound));
            break;
        case 25: // q5
        case 26: // q6
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?artist", context, &variables_bound));
            break;
        case 27: // q7
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?id", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?id", context, &variables_bound));
            break;
        case 28: // q8
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?person", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?person", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(3)));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(5), "?person", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(4)));
            plan.pushJob(new JoinQueryJob(plan.getJob(6), plan.getJob(7), "?person", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(5)));
            plan.pushJob(new JoinQueryJob(plan.getJob(8), plan.getJob(9), "?person", context, &variables_bound));
            break;
        case 29: // q9
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(1)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?x", context, &variables_bound));
            plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(2)));
            plan.pushJob(new JoinQueryJob(plan.getJob(2), plan.getJob(3), "?x", context, &variables_bound));
            break;

        // LUBM
        case 101: // q1
        case 103: // q3
        case 105: // q5
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(1), "?X", context, &variables_bound));
            break;
        case 102: // q2
            break;
        case 104: // q4
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            plan.pushJob(new JoinQueryJob(plan.getJob(0), plan.getJob(2), "?X", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(1), plan.getJob(3), "?X", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(4), plan.getJob(6), "?X", context, &variables_bound));
            plan.pushJob(new JoinQueryJob(plan.getJob(5), plan.getJob(7), "?X", context, &variables_bound));
            break;
        case 106: // q6
            for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(i)));
            break;

        default:
            if (sparqlQuery.getPatternNum() == 1) {
                plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
            } else {
                assert(false);
            }
            break;
    }
#endif

    auto planing_end = std::chrono::high_resolution_clock::now();
#ifdef TIME_DEBUG
    std::cout << "Planing time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(planing_end-planing_start).count() << " ms.\n";
#endif

    // plan.print();
    plan.execute(sparqlResult, parallel_plan);
}

void QueryExecutor::printBounds() const {
    std::cout << "==== SELECTED BOUND ====\n";
    for (auto &vb: variables_bound) {
        std::cout << std::setw(7) << vb.first << " : [" << vb.second.first << "," << vb.second.second << "]\n";
    }
    std::cout << "========================\n";
}

void QueryExecutor::createVariableBound(SparqlQuery &sparqlQuery) {

    std::set<std::string> join_vars;
    TYPEID min_val = std::numeric_limits<TYPEID>::min();
    TYPEID max_val = std::numeric_limits<TYPEID>::max();

    // 1. Find join variables
    std::map<std::string, int> vars_count;
    for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) {
        TriplePattern *tp = sparqlQuery.getPatternPtr(i);
        tp->estimate_rows = vedasStorage->getTripleSize();

        if (tp->subjectIsVariable()) vars_count[tp->getSubject()] += 1;
        if (tp->predicateIsVariable()) vars_count[tp->getPredicate()] += 1;
        if (tp->objectIsVariable()) vars_count[tp->getObject()] += 1;
    }
    for (auto &c : vars_count) {
        if (c.second > 1) {
            std::cout << "ADD " << c.first << " TO JOIN VAR\n";
            join_vars.insert(c.first);
            // 2. Create initial bound (0, inf) for each join variables
            variables_bound[c.first] = std::make_pair(min_val, max_val);
        }
    }

    // 3. Thigten bound for each vars
    for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) {
        squeezeQueryBound(sparqlQuery.getPatternPtr(i));
    }
}

void QueryExecutor::squeezeQueryBound(TriplePattern *pattern) {
    switch (pattern->getVariableNum()) {
        case 1: squeezeQueryBound1Var(pattern); break;
        case 2: squeezeQueryBound2Var(pattern); break;
    }
}

void QueryExecutor::squeezeQueryBound1Var(TriplePattern *pattern) {
    TYPEID_HOST_VEC *l1_index_values = nullptr, *l1_index_offsets  = nullptr;
    TYPEID_HOST_VEC *l1_index_values2 = nullptr, *l1_index_offsets2  = nullptr;
    TYPEID_HOST_VEC *l2_index_values = nullptr, *l2_index_offsets = nullptr, *data = nullptr, *data2 = nullptr;
    TYPEID_HOST_VEC *l2_data = nullptr, *l2_data2 = nullptr;
    std::string v1;
    TYPEID id1, id2;

    if (pattern->subjectIsVariable()) {
        l1_index_values = this->vedasStorage->getObjectIndexValues();
        l1_index_offsets = this->vedasStorage->getObjectIndexOffsets();
        l2_index_values = this->vedasStorage->getObjectPredicateIndexValues();
        l2_index_offsets = this->vedasStorage->getObjectPredicateIndexOffsets();
        l1_index_values2 = this->vedasStorage->getPredicateIndexValues();
        l1_index_offsets2 = this->vedasStorage->getPredicateIndexOffsets();
        l2_data2 = this->vedasStorage->getPOdata();
        data = this->vedasStorage->getOPSdata();
        data2 = this->vedasStorage->getPOSdata();
        v1 = pattern->getSubject();
        id1 = pattern->getObjectId();
        id2 = pattern->getPredicateId();
    } else if (pattern->predicateIsVariable()) {
        l1_index_values = this->vedasStorage->getSubjectIndexValues();
        l1_index_offsets = this->vedasStorage->getSubjectIndexOffsets();
        l2_index_values = this->vedasStorage->getSubjectObjectIndexValues();
        l2_index_offsets = this->vedasStorage->getSubjectObjectIndexOffsets();
        l1_index_values2 = this->vedasStorage->getObjectIndexValues();
        l1_index_offsets2 = this->vedasStorage->getObjectIndexOffsets();
        l2_data2 = this->vedasStorage->getOSdata();
        data = this->vedasStorage->getSOPdata();
        data2 = this->vedasStorage->getOSPdata();
        v1 = pattern->getPredicate();
        id1 = pattern->getSubjectId();
        id2 = pattern->getObjectId();
    } else {
        l1_index_values = this->vedasStorage->getSubjectIndexValues();
        l1_index_offsets = this->vedasStorage->getSubjectIndexOffsets();
        l2_index_values = this->vedasStorage->getSubjectPredicateIndexValues();
        l2_index_offsets = this->vedasStorage->getSubjectPredicateIndexOffsets();
        l1_index_values2 = this->vedasStorage->getPredicateIndexValues();
        l1_index_offsets2 = this->vedasStorage->getPredicateIndexOffsets();
        l2_data2 = this->vedasStorage->getPSdata();
        data = this->vedasStorage->getSPOdata();
        data2 = this->vedasStorage->getPSOdata();
        v1 = pattern->getObject();
        id1 = pattern->getSubjectId();
        id2 = pattern->getPredicateId();
    }

    updateDataBound(pattern, l1_index_values, l1_index_offsets, l2_index_values, l2_index_offsets, l2_data, data, v1, id1, id2);
    updateDataBound(pattern, l1_index_values2, l1_index_offsets2, nullptr, nullptr, l2_data2, data2, v1, id2, id1);
}


std::pair<size_t, size_t> QueryExecutor::findL2OffsetFromL1(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets, TYPEID id1, size_t n) {
    auto l1_bit = thrust::lower_bound(thrust::host, l1_index_values->begin(), l1_index_values->end(), id1);
    auto start_offst = thrust::distance(l1_index_values->begin(), l1_bit);
    auto end_offst = start_offst + 1;

    auto l1_offst_lower_bound = *(l1_index_offsets->begin() + start_offst);
    auto l1_offst_upper_bound = (end_offst == l1_index_offsets->size())? n :  *(l1_index_offsets->begin() + end_offst);
    // TODO: if not exist, l1_bit == l1_index_values->end()

    return std::make_pair(l1_offst_lower_bound, l1_offst_upper_bound);
}

std::pair<size_t, size_t> QueryExecutor::findDataOffsetFromL2(TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                                              TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound, TYPEID id2, size_t n) {
    auto l1_offst_bit = thrust::lower_bound(thrust::host, l2_index_offsets->begin(), l2_index_offsets->end(), l1_offst_lower_bound);
    auto l1_offst_eit = thrust::lower_bound(thrust::host, l2_index_offsets->begin(), l2_index_offsets->end(), l1_offst_upper_bound);
    auto start_offst2 = thrust::distance(l2_index_offsets->begin(), l1_offst_bit);
    auto end_offst2 = thrust::distance(l2_index_offsets->begin(), l1_offst_eit);

    auto l2_begin = l2_index_values->begin() + start_offst2;
    auto l2_end = l2_index_values->begin() + end_offst2;

    auto l2_bit = thrust::lower_bound(thrust::host, l2_begin, l2_end, id2);
    auto l2_eit = thrust::upper_bound(thrust::host, l2_begin, l2_end, id2);
    auto l2_start_offst = thrust::distance(l2_index_values->begin(), l2_bit);
    auto l2_end_offst = thrust::distance(l2_index_values->begin(), l2_eit);

    // TODO: if not exist, l2_eit == l2_index_values->end()

    size_t data_start_offst = *(l2_index_offsets->begin() + l2_start_offst);
    size_t data_end_offst = *(l2_index_offsets->begin() + l2_end_offst);

    return std::make_pair(data_start_offst, data_end_offst);
}

std::pair<size_t, size_t> QueryExecutor::findDataOffsetFromL2(TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound,
                                                              TYPEID id2, size_t n) {
    auto start_find = l2_data->begin() + l1_offst_lower_bound;
    auto end_find = l2_data->begin() + l1_offst_upper_bound;
    auto start_l2 = thrust::lower_bound(thrust::host, start_find, end_find, id2);
    auto end_l2 = thrust::upper_bound(thrust::host, start_find, end_find, id2);

    size_t data_start_offst = thrust::distance(l2_data->begin(), start_l2);
    size_t data_end_offst = thrust::distance(l2_data->begin(), end_l2);

    return std::make_pair(data_start_offst, data_end_offst);
}

void QueryExecutor::updateDataBound(TriplePattern *pattern, TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                                    TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                    TYPEID_HOST_VEC *l2_data, TYPEID_HOST_VEC *data, std::string &v, TYPEID id1, TYPEID id2) {

    auto l2_offst_pair = findL2OffsetFromL1(l1_index_values, l1_index_offsets, id1, data->size());
    std::pair<size_t, size_t> data_offst_pair = (l2_data != nullptr)?
                findDataOffsetFromL2(l2_data, l2_offst_pair.first, l2_offst_pair.second, id2, data->size()) :
                findDataOffsetFromL2(l2_index_values, l2_index_offsets, l2_offst_pair.first, l2_offst_pair.second, id2, data->size());

    // TODO: estimate again after bounding variable
    pattern->estimate_rows = data_offst_pair.second - data_offst_pair.first;

    if (variables_bound.count(v) > 0) {
        auto bound = variables_bound[v];
#ifdef VERBOSE_DEBUG
        std::cout << "(D)UPDATE " << v << " BOUND FROM [" << bound.first << ","<< bound.second<<"] TO ";
#endif
        auto new_min = std::max(bound.first, *(data->begin() + data_offst_pair.first));
        auto new_max = std::min(bound.second, *(data->begin() + data_offst_pair.second - 1));
#ifdef VERBOSE_DEBUG
        std::cout << "["<<new_min<<","<<new_max<<"]\n";
#endif
        variables_bound[v] = std::make_pair(new_min, new_max);
    }
}

void QueryExecutor::squeezeQueryBound2Var(TriplePattern *pattern) {

    TYPEID_HOST_VEC *l1_index_values = nullptr, *l1_index_offsets = nullptr;
    TYPEID_HOST_VEC *l2_index_values = nullptr, *l2_index_offsets = nullptr;
    TYPEID_HOST_VEC *l2_index_values2 = nullptr, *l2_index_offsets2 = nullptr;
    TYPEID_HOST_VEC *l2_data = nullptr, *l2_data2 = nullptr;
    size_t data_size = this->vedasStorage->getSOPdata()->size();
    std::string v1, v2;
    TYPEID id1;

    switch (pattern->getVariableBitmap()) {
        //return (this->isVar[0] * 4) + (this->isVar[1] * 2) + (this->isVar[2]);
        // S P O
        case 3:
            l1_index_values = this->vedasStorage->getSubjectIndexValues();
            l1_index_offsets = this->vedasStorage->getSubjectIndexOffsets();
            l2_index_values = this->vedasStorage->getSubjectObjectIndexValues();
            l2_index_offsets = this->vedasStorage->getSubjectObjectIndexOffsets();
            l2_index_values2 = this->vedasStorage->getSubjectPredicateIndexValues();
            l2_index_offsets2 = this->vedasStorage->getSubjectPredicateIndexOffsets();
            v1 = pattern->getObject();
            v2 = pattern->getPredicate();
            id1 = pattern->getSubjectId();
            break;
        case 5:
            l1_index_values = this->vedasStorage->getPredicateIndexValues();
            l1_index_offsets = this->vedasStorage->getPredicateIndexOffsets();
            l2_data = this->vedasStorage->getPOdata();
            l2_data2 = this->vedasStorage->getPSdata();
            v1 = pattern->getObject();
            v2 = pattern->getSubject();
            id1 = pattern->getPredicateId();
            break;
        case 6:
            l1_index_values = this->vedasStorage->getObjectIndexValues();
            l1_index_offsets = this->vedasStorage->getObjectIndexOffsets();
            l2_index_values = this->vedasStorage->getObjectPredicateIndexValues();
            l2_index_offsets = this->vedasStorage->getObjectPredicateIndexOffsets();
            l2_data2 = this->vedasStorage->getOSdata();
            v1 = pattern->getPredicate();
            v2 = pattern->getSubject();
            id1 = pattern->getObjectId();
            break;
        default:
            std::cout << "Pattern Bitmap is " << pattern->getVariableBitmap() << "\n";
            assert(false);
    }

    auto l2_offst = findL2OffsetFromL1(l1_index_values, l1_index_offsets, id1, data_size);

    updateL2Bound(pattern, v1, l2_index_values, l2_index_offsets, l2_data, l2_offst.first, l2_offst.second);
    updateL2Bound(pattern, v2, l2_index_values2, l2_index_offsets2, l2_data2, l2_offst.first, l2_offst.second);
}

void QueryExecutor::updateL2Bound(TriplePattern *pattern, std::string &var, TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                  TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound) {
    size_t start_offst2, end_offst2;
    TYPEID_HOST_VEC::iterator l2_begin, l2_end;
    if (l2_data != nullptr) {
        start_offst2 = l1_offst_lower_bound;
        end_offst2 = l1_offst_upper_bound; // XXX: is it correct ??
        l2_begin = l2_data->begin() + start_offst2;
        l2_end = l2_data->begin() + end_offst2;
    } else {
        auto l1_offst_bit = thrust::lower_bound(thrust::host, l2_index_offsets->begin(), l2_index_offsets->end(), l1_offst_lower_bound);
        auto l1_offst_eit = thrust::lower_bound(thrust::host, l2_index_offsets->begin(), l2_index_offsets->end(), l1_offst_upper_bound);
        start_offst2 = thrust::distance(l2_index_offsets->begin(), l1_offst_bit);
        end_offst2 = thrust::distance(l2_index_offsets->begin(), l1_offst_eit);
        l2_begin = l2_index_values->begin() + start_offst2;
        l2_end = l2_index_values->begin() + end_offst2;
    }

    if (variables_bound.count(var) > 0) {
        auto bound = variables_bound[var];
#ifdef VERBOSE_DEBUG
        std::cout << "(L2)UPDATE " << var << " BOUND FROM [" << bound.first << ","<< bound.second<<"] TO ";
#endif
        auto new_min = std::max(bound.first, *l2_begin);
        auto new_max = std::min(bound.second, *(l2_end-1));
#ifdef VERBOSE_DEBUG
        std::cout << "["<<new_min<<","<<new_max<<"]\n";
#endif
        variables_bound[var] = std::make_pair(new_min, new_max);
    }
}

void QueryExecutor::estimateRelationSize() {

}

SelectQueryJob* QueryExecutor::createSelectQueryJob(TriplePattern *pattern, std::string index_used, std::pair<TYPEID, TYPEID> *bound) {
    switch (pattern->getVariableNum()) {
        case 1: return this->create1VarSelectQueryJob(pattern, index_used, bound);
        case 2: return this->create2VarSelectQueryJob(pattern, index_used, bound);
        case 3: return this->create3VarSelectQueryJob(pattern); // Exploration
        default: assert(false);
    }
    return nullptr;
}

SelectQueryJob* QueryExecutor::create1VarSelectQueryJob(TriplePattern *pattern, std::string index_used, std::pair<TYPEID, TYPEID> *bound) {
    bool is_predicates[1];
    if (pattern->subjectIsVariable()) {
#ifdef DEBUG
        std::cout << "Use [OP]S Index\n";
        std::cout << "Search for " << pattern->getSubject() << " " << pattern->getObjectId() << " " << pattern->getPredicateId() << "\n";
#endif
        is_predicates[0] = false;
        return new SelectQueryJob(
                    this->vedasStorage->getObjectIndexValues(),
                    this->vedasStorage->getObjectIndexOffsets(),
                    this->vedasStorage->getObjectPredicateIndexValues(),
                    this->vedasStorage->getObjectPredicateIndexOffsets(),
                    pattern->getSubject(), pattern->getObjectId(), pattern->getPredicateId(),
                    this->vedasStorage->getOPSdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDeviceOPSdata() : nullptr,
                    &variables_bound, is_predicates, context
                );
    } else if (pattern->predicateIsVariable()) {
#ifdef DEBUG
        std::cout << "Use [SO]P Index\n";
        std::cout << "Search for " << pattern->getPredicate() << " " << pattern->getSubjectId() << " " << pattern->getObjectId() << "\n";
#endif
        is_predicates[0] = true;
        return new SelectQueryJob(
                    this->vedasStorage->getSubjectIndexValues(),
                    this->vedasStorage->getSubjectIndexOffsets(),
                    this->vedasStorage->getSubjectObjectIndexValues(),
                    this->vedasStorage->getSubjectObjectIndexOffsets(),
                    pattern->getPredicate(), pattern->getSubjectId(), pattern->getObjectId(),
                    this->vedasStorage->getSOPdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDeviceSOPdata() : nullptr,
                    &variables_bound, is_predicates, context
                );
    } else {
#ifdef DEBUG
        std::cout << "Use [SP]O Index\n";
        std::cout << "Search for " << pattern->getObject() << " " << pattern->getSubjectId() << " " << pattern->getObjectId() << "\n";
#endif
        is_predicates[0] = false;
        return new SelectQueryJob(
                    this->vedasStorage->getSubjectIndexValues(),
                    this->vedasStorage->getSubjectIndexOffsets(),
                    this->vedasStorage->getSubjectPredicateIndexValues(),
                    this->vedasStorage->getSubjectPredicateIndexOffsets(),
                    pattern->getObject(), pattern->getSubjectId(), pattern->getPredicateId(),
                    this->vedasStorage->getSPOdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDeviceSPOdata() : nullptr,
                    &variables_bound, is_predicates, context
                );
    }
}

SelectQueryJob* QueryExecutor::create2VarSelectQueryJob(TriplePattern *pattern, std::string index_used, std::pair<TYPEID, TYPEID> *bound) {
    std::transform(index_used.begin(), index_used.end(),index_used.begin(), ::toupper);

    bool is_second_var_used = true;
    bool is_predicates[2];
    switch (pattern->getVariableBitmap()) {
        //return (this->isVar[0] * 4) + (this->isVar[1] * 2) + (this->isVar[2]);
        // S P O
        case 3:
            // Default is [SPO]
            if (index_used == "SOP") {
//              std::cout << "Use [SOP] Index\n";
                is_second_var_used = (selected_variables.count(pattern->getPredicate()) > 0) || (variables_bound.count(pattern->getPredicate()) > 0);
                is_predicates[0] = false; is_predicates[1] = true;
                return new SelectQueryJob(
                            this->vedasStorage->getSubjectIndexValues(),
                            this->vedasStorage->getSubjectIndexOffsets(),
                            this->vedasStorage->getSubjectObjectIndexValues(),
                            this->vedasStorage->getSubjectObjectIndexOffsets(),
                            nullptr,
                            pattern->getObject(), pattern->getPredicate(), pattern->getSubjectId(),
                            this->vedasStorage->getSOPdata(),
                            this->vedasStorage->isPreload()? this->vedasStorage->getDeviceSOPdata() : nullptr,
                            &variables_bound, is_predicates, is_second_var_used, context
                        );
            } else {
//              std::cout << "Use [SPO] Index\n";
                is_second_var_used = (selected_variables.count(pattern->getObject()) > 0) || (variables_bound.count(pattern->getObject()) > 0);
                is_predicates[0] = true; is_predicates[1] = false;
                return new SelectQueryJob(
                            this->vedasStorage->getSubjectIndexValues(),
                            this->vedasStorage->getSubjectIndexOffsets(),
                            this->vedasStorage->getSubjectPredicateIndexValues(),
                            this->vedasStorage->getSubjectPredicateIndexOffsets(),
                            nullptr,
                            pattern->getPredicate(), pattern->getObject(), pattern->getSubjectId(),
                            this->vedasStorage->getSPOdata(),
                            this->vedasStorage->isPreload()? this->vedasStorage->getDeviceSPOdata() : nullptr,
                            &variables_bound, is_predicates, is_second_var_used, context
                        );
            }
        case 5:
            // Default is [PSO]
            {
                if (index_used == "POS") {
                    // std::cout << "Use [POS] Index\n";
                    is_second_var_used = (selected_variables.count(pattern->getSubject()) > 0) || (variables_bound.count(pattern->getSubject()) > 0);
                    is_predicates[0] = false; is_predicates[1] = false;
                    return new SelectQueryJob(
                                this->vedasStorage->getPredicateIndexValues(),
                                this->vedasStorage->getPredicateIndexOffsets(),
                                nullptr, nullptr,
                                this->vedasStorage->getPOdata(),
                                pattern->getObject(), pattern->getSubject(), pattern->getPredicateId(),
                                this->vedasStorage->getPOSdata(),
                                this->vedasStorage->isPreload()? this->vedasStorage->getDevicePOSdata() : nullptr,
                                &variables_bound, is_predicates, is_second_var_used, context
                            );
                } else {
                    // std::cout << "Use [PSO] Index\n";
                    is_second_var_used = (selected_variables.count(pattern->getObject()) > 0) || (variables_bound.count(pattern->getObject()) > 0);
                    is_predicates[0] = false; is_predicates[1] = false;
                    return new SelectQueryJob(
                                this->vedasStorage->getPredicateIndexValues(),
                                this->vedasStorage->getPredicateIndexOffsets(),
                                nullptr, nullptr,
                                this->vedasStorage->getPSdata(),
                                pattern->getSubject(), pattern->getObject(), pattern->getPredicateId(),
                                this->vedasStorage->getPSOdata(),
                                this->vedasStorage->isPreload()? this->vedasStorage->getDevicePSOdata() : nullptr,
                                &variables_bound, is_predicates, is_second_var_used, context
                            );
                }
            }

        case 6:
            if (index_used == "OPS") {
//              std::cout << "Use [OPS] Index\n";
                is_second_var_used = (selected_variables.count(pattern->getSubject()) > 0) || (variables_bound.count(pattern->getSubject()) > 0);
                is_predicates[0] = true; is_predicates[1] = false;
                return new SelectQueryJob(
                            this->vedasStorage->getObjectIndexValues(),
                            this->vedasStorage->getObjectIndexOffsets(),
                            this->vedasStorage->getObjectPredicateIndexValues(),
                            this->vedasStorage->getObjectPredicateIndexOffsets(),
                            nullptr,
                            pattern->getPredicate(), pattern->getSubject(), pattern->getObjectId(),
                            this->vedasStorage->getOPSdata(),
                            this->vedasStorage->isPreload()? this->vedasStorage->getDeviceOPSdata() : nullptr,
                            &variables_bound, is_predicates, is_second_var_used, context
                        );
            } else {
//              std::cout << "Use [OSP] Index\n";
                is_second_var_used = (selected_variables.count(pattern->getPredicate()) > 0) || (variables_bound.count(pattern->getPredicate()) > 0);
                is_predicates[0] = false; is_predicates[1] = true;
                return new SelectQueryJob(
                            this->vedasStorage->getObjectIndexValues(),
                            this->vedasStorage->getObjectIndexOffsets(),
                            nullptr, nullptr,
                            this->vedasStorage->getOSdata(),
                            pattern->getSubject(), pattern->getPredicate(), pattern->getObjectId(),
                            this->vedasStorage->getOSPdata(),
                            this->vedasStorage->isPreload()? this->vedasStorage->getDeviceOSPdata() : nullptr,
                            &variables_bound, is_predicates, is_second_var_used, context
                        );
            }

        default: assert(false);
    }
    return nullptr;
}

SelectQueryJob* QueryExecutor::create3VarSelectQueryJob(TriplePattern *pattern) {
    assert(false); // TODO: exploration query
    return nullptr;
}
