#include <algorithm>
#include <vector>
#include <cassert>
#include <string>
#include <chrono>
#include <iomanip>
#include <utility>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "ExecutionPlanTree.h"
#include "QueryExecutor.h"
#include "QueryGraph.h"
#include "JoinGraph.h"
#include "JoinQueryJob.h"
#include "IndexSwapJob.h"
#include "TransferJob.h"

// Histogram *QueryExecutor::objectHistogram = nullptr;
// Histogram *QueryExecutor::subjectHistogram = nullptr;

REVERSE_DICTTYPE *QueryExecutor::r_so_map;
REVERSE_DICTTYPE *QueryExecutor::r_p_map;
REVERSE_DICTTYPE *QueryExecutor::r_l_map;
double QueryExecutor::load_dict_ms;
double QueryExecutor::load_data_ms;
double QueryExecutor::indexing_ms;
double QueryExecutor::upload_ns;
double QueryExecutor::join_ns;
double QueryExecutor::alloc_copy_ns;
double QueryExecutor::download_ns;
double QueryExecutor::swap_index_ns;
double QueryExecutor::eliminate_duplicate_ns;
double QueryExecutor::convert_to_id_ns;
double QueryExecutor::convert_to_iri_ns;
double QueryExecutor::scan_to_split_ns;
double QueryExecutor::p2p_transfer_ns;
double QueryExecutor::prescan_extra_ns;
double QueryExecutor::eif_ns;
int QueryExecutor::eif_count;
std::vector<ExecuteLogRecord> QueryExecutor::exe_log;

bool QueryExecutor::ENABLE_FULL_INDEX;
bool QueryExecutor::ENABLE_LITERAL_DICT;
bool QueryExecutor::ENABLE_PREUPLOAD_BOUND_DICT;
bool QueryExecutor::ENABLE_BOUND_DICT_AFTER_JOIN;
bool QueryExecutor::ENABLE_DOUBLE_BOUND_DICT_AFTER_JOIN;
bool QueryExecutor::ENABLE_UPDATE_BOUND_AFTER_JOIN;

void TriplePatternDataIndex::updateDataIndex(size_t idx, TYPEID_HOST_VEC * basePtr, TYPEID_HOST_VEC_IT lowerIt, TYPEID_HOST_VEC_IT upperIt) {
    std::vector<DataIndex> &dataIndices = tp_data_index[idx];
    bool basePtrFound = false;
    for (auto& di : dataIndices) {
        if (di.data == basePtr) {
            di.lower_it = lowerIt;
            di.upper_it = upperIt;
            basePtrFound = true;
        }
    }
    if (!basePtrFound) {
        DataIndex di;
        di.data = basePtr;
        di.lower_it = lowerIt;
        di.upper_it = upperIt;
        tp_data_index[idx].push_back(di);
    }
}

// std::tuple<DataIndex> TriplePatternDataIndex::retrieveDataIndex(size_t idx) {
//     if (tp_data_index.count(idx) == 0) {
//         throw;
//     }

//     tp_data_index[idx]
// }

DataIndex TriplePatternDataIndex::retrieveMinLenDataIndex(size_t idx) {
    if (tp_data_index.count(idx) == 0) throw;

    size_t minLen = std::numeric_limits<size_t>::max();
    DataIndex *minLenDataIndex;
    for (DataIndex &di : tp_data_index[idx]) {
        size_t len = thrust::distance(di.lower_it, di.upper_it);
        std::cout << "\tDATA INDEX LENGTH OF [" << idx << "] = " << len << "\n";
        if (len < minLen) {
            minLen = len;
            minLenDataIndex = &di;
        }
    }
    return *minLenDataIndex;
}

// QueryExecutor::QueryExecutor(VedasStorage *vedasStorage, ctpl::thread_pool *threadPool, mgpu::standard_context_t* context, int plan_id) {
QueryExecutor::QueryExecutor(VedasStorage *vedasStorage, ExecutionWorker *worker, mgpu::standard_context_t* context) {
    this->vedasStorage = vedasStorage;
    this->worker = worker;
    this->context = context;
}

void QueryExecutor::setGpuIds(const std::vector<int>& gpu_ids) {
    this->gpu_ids = gpu_ids;
}

int QueryExecutor::postorderTraversal(QueryPlan &plan, PlanTreeNode* root, size_t thread_no = 0) {
    if (root == nullptr) return -1;

    int j1, j2;
    if (root->child1 != nullptr) j1 = postorderTraversal(plan, root->child1, thread_no);
    if (root->child2 != nullptr) j2 = postorderTraversal(plan, root->child2, thread_no);
    
    // std::cout << root->op << " (" << root->debugName << ") thread " << thread_no << " \n";
    switch (root->op) {
        case UPLOAD:
            if (root->tp->getVariableNum() == 1) {
                plan.pushJob(this->createSelectQueryJob(root->tp), thread_no);
            } else if (root->tp->getVariableNum() == 2) {
                std::string indexUse = (root->tp->getSubject() == root->var)? "PSO" : "POS";
                plan.pushJob(this->createSelectQueryJob(root->tp, indexUse), thread_no);
            } else {
               assert(false);
            }
            break;
        case JOIN:
            plan.pushJob(new JoinQueryJob(plan.getJob(j1, thread_no), plan.getJob(j2, thread_no), root->var, 
                context, &variables_bound, &empty_interval_dict, !root->reuseVar), thread_no);
            break;
        case INDEXSWAP:
            plan.pushJob(new IndexSwapJob(plan.getJob(j1, thread_no), root->var, 
                context, &variables_bound, &empty_interval_dict), thread_no);
            break;
    }
    return jobCount++;
}

void QueryExecutor::createStarJoinPlan(QueryPlan &plan, std::vector<TriplePattern> *tps, std::vector<size_t> &ids, 
                                        std::string joinVar, size_t thread_no) {
    TriplePattern &tp = tps->at(ids[0]);
    if (tp.getVariableNum() == 1) {
        plan.pushJob(this->createSelectQueryJob(&tp), thread_no);
    } else if (tp.getVariableNum() == 2) {
        std::string indexUse = (tp.getSubject() == joinVar)? "PSO" : "POS";
        plan.pushJob(this->createSelectQueryJob(&tp, indexUse), thread_no);
    } else {
        assert(false);
    }

    for (int i = 1; i < ids.size(); ++i) {
        TriplePattern &tp = tps->at(ids[i]);
        if (tp.getVariableNum() == 1) {
            plan.pushJob(this->createSelectQueryJob(&tp), thread_no);
        } else if (tp.getVariableNum() == 2) {
            std::string indexUse = (tp.getSubject() == joinVar)? "PSO" : "POS";
            plan.pushJob(this->createSelectQueryJob(&tp, indexUse), thread_no);
        } else {
           assert(false);
        }
        bool lastJoinForVar = false; // TODO: should be last for reduce EIF
        plan.pushJob(new JoinQueryJob(plan.getJob(i*2-2, thread_no), plan.getJob(i*2-1, thread_no), joinVar,
                context, &variables_bound, &empty_interval_dict, lastJoinForVar), thread_no);
    }
}

PlanTreeNode *createRandomPlan(SparqlQuery *sparqlQuery, int seed) {
    std::srand(seed);
    std::unordered_set<int> used;
    std::unordered_set<std::string> vars;
    
    size_t patternSize = sparqlQuery->getPatternNum();
    PlanTreeNode *root = nullptr;
    for (size_t i = 0; i < patternSize; i++) {
        
        int r = -1;
        while (true) {
            r = std::rand() % patternSize;
            if (used.count(r)) continue; // random again

            TriplePattern *tp = sparqlQuery->getPatternPtr(r);

            if (vars.size() == 0) {
                std::string v;
                if (tp->subjectIsVariable()) { 
                    vars.insert(tp->getSubject());
                    v = tp->getSubject();
                }
                if (tp->objectIsVariable()) {
                    vars.insert(tp->getObject());
                    v = tp->getSubject();
                }

                root = new PlanTreeNode();
                root->op = UPLOAD;
                root->tp = tp;
                root->debugName = tp->toString();
                root->var = v; // to use in next join
            
            } else {
                int match = 0;
                if (tp->subjectIsVariable() && vars.count(tp->getSubject())) match += 1;
                if (tp->objectIsVariable() && vars.count(tp->getObject())) match += 2;
                if (match == 0) continue;

                std::string joinVar = (match == 1) ? tp->getSubject() : tp->getObject();
                if (tp->subjectIsVariable()) vars.insert(tp->getSubject());
                if (tp->objectIsVariable()) vars.insert(tp->getObject());

                if (root->op == UPLOAD) root->var = joinVar;

                PlanTreeNode *node = new PlanTreeNode();
                node->op = UPLOAD;
                node->tp = tp;
                node->debugName = tp->toString();
                node->var = joinVar;

                PlanTreeNode *jnode = new PlanTreeNode();
                jnode->op = JOIN;
                jnode->child1 = root;
                jnode->child2 = node;
                jnode->debugName = joinVar;
                jnode->var = joinVar;
                jnode->reuseVar = false; // TODO:

                root = jnode;
            }
            
            used.insert(r); break;
        }
    }
    return root;
}

int QueryExecutor::jobCount = 0;
void QueryExecutor::createPlanExecFromPlanTree(QueryPlan &plan, PlanTreeNode* root, size_t thread_no = 0) {
    jobCount = 0;
    postorderTraversal(plan, root, thread_no);
}

std::string jobObToStr(PlanOperation op) {
    switch (op) {
        case UPLOAD: return "UPLOAD";
        case JOIN: return "JOIN";
        case INDEXSWAP: return "INDEX-SWAP";
    }
    return "";
}

void printBT(const std::string& prefix, const PlanTreeNode* node, bool isLeft) {
    if (node != nullptr) {
        std::cout << prefix;

        std::cout << (isLeft ? "├──" : "└──" );

        // print the value of the node
        std::cout << jobObToStr(node->op) << "  " << node->debugName << '\n';

        // enter the next tree level - left and right branch
        printBT( prefix + (isLeft ? "│   " : "    "), node->child1, true);
        printBT( prefix + (isLeft ? "│   " : "    "), node->child2, false);
    }
}

void printBT(const PlanTreeNode* node) {
    printBT("", node, false);
}

bool contains(std::vector<int>& v, int k) {
    for (int e: v) 
        if (e == k) return true; 
    return false;
}

// remove duplicate value from a
int removeDuplicate(std::vector<int>& a, std::vector<int> b) {
    for (size_t i = 0; i < b.size(); ++i) {
        if (contains(a, b[i])) {
            a.erase(std::remove(a.begin(), a.end(), b[i]), a.end());
            return b[i];
        }
    }
}

std::string findJoinVar(SparqlQuery &sparqlQuery, std::vector<int>& a, int removeIdx) {
    for (size_t i = 0; i < a.size(); ++i) {
        // sparqlQuery
        TriplePattern *atp = sparqlQuery.getPatternPtr(a[i]);
        TriplePattern *rtp = sparqlQuery.getPatternPtr(removeIdx);
        std::string commonVar = atp->hasCommonVariable(*rtp);
        if (commonVar != "") return commonVar;
    }
    return "";
}

void QueryExecutor::query(SparqlQuery &sparqlQuery, SparqlResult &sparqlResult) {
    // Copy selected variables for filter
    
    try {
    selected_variables = sparqlQuery.getSelectedVariables();

    QueryPlan plan(worker, selected_variables);

    auto planing_start = std::chrono::high_resolution_clock::now();

    // TODO: 0 pattern ?
    if (sparqlQuery.getPatternNum() == 1) {
        // TODO: projection, eliminate duplicate
        plan.pushJob(this->createSelectQueryJob(sparqlQuery.getPatternPtr(0)));
        plan.print();
        plan.execute(sparqlResult, true /* single GPU */);

        return;
    }

    createVariableBound(sparqlQuery);
#ifdef VERBOSE_DEBUG
    printBounds();
#endif

    // TODO: Get meta data
    auto planing2_start = std::chrono::high_resolution_clock::now();

    int processGpuCount = 1;
    if (sparqlQuery.getPatternNum() <= 4 || gpu_ids.size() < 2) {
        // std::cout << "Use Single GPU\n";
        QueryGraph qg(&sparqlQuery, nullptr); // Construct the query graph
        PlanTreeNode* root = qg.generateQueryPlan();
        // printBT(root);
        
        createPlanExecFromPlanTree(plan, root);
        
        // 1-100
        // 2-300
        // 3-500
        // PlanTreeNode* root = createRandomPlan(&sparqlQuery, 100);
        // PlanTreeNode::printBT(root);
        // createPlanExecFromPlanTree(plan, root);
    } else {
        // std::cout << "Use Multi-GPUs\n";
        
        if (sparqlQuery.isStarShaped()) {
            processGpuCount = 2;
            
            // floor(tripleNum/4.0)
            std::vector<std::vector<size_t>> groups;
            std::string cJoinVar = sparqlQuery.getStarCenterVariable();
            sparqlQuery.splitStarQuery(processGpuCount, groups);
            for (int gi = 0; gi < groups.size(); ++gi) {
                int thread_no = gi;
                createStarJoinPlan(plan, sparqlQuery.getPatternsPtr(), 
                                    groups[gi], cJoinVar, thread_no);
            }
            
            // push last dynamic join job 
            plan.pushDynamicJob(new JoinQueryJob(nullptr, nullptr, cJoinVar, context, &variables_bound, &empty_interval_dict));
            
        } else {
            processGpuCount = 2;
            std::vector<std::vector<int>> tpIds;
            JoinGraph jg(&sparqlQuery);
            std::cout << "Join Graph\n";
            // jg.print();
            jg.splitQuery(tpIds, 2);
            std::map<int, int> tpGroupDict;
            int toRemoveGroup;

            int removeTpIdx;
            if (tpIds[0].size() > tpIds[1].size()) {
                removeTpIdx = removeDuplicate(tpIds[0], tpIds[1]);
                toRemoveGroup = 0;
            } else if (tpIds[0].size() < tpIds[1].size()) {
                removeTpIdx = removeDuplicate(tpIds[1], tpIds[0]);
                toRemoveGroup = 1;
            } else {
                // TODO: check cardinality
                removeTpIdx = removeDuplicate(tpIds[1], tpIds[0]);
                toRemoveGroup = 1;
            }
            
            // std::cout << "Set 0 : "; for (auto i: tpIds[0]) std::cout << i << " "; std::cout << "\n";
            // std::cout << "Set 1 : "; for (auto i: tpIds[1]) std::cout << i << " "; std::cout << "\n";

            QueryGraph qg1(&sparqlQuery, &(tpIds[0]));
            PlanTreeNode* root1 = qg1.generateQueryPlan();
            QueryGraph qg2(&sparqlQuery, &(tpIds[1]));
            PlanTreeNode* root2 = qg2.generateQueryPlan();
            // printBT(root1);
            // printBT(root2);
            
            createPlanExecFromPlanTree(plan, root1, 0);
            createPlanExecFromPlanTree(plan, root2, 1);

            // TODO: push last join job 
            std::string cJoinVar = findJoinVar(sparqlQuery, tpIds[toRemoveGroup], removeTpIdx);
            assert(cJoinVar != "");
            plan.pushDynamicJob(new JoinQueryJob(nullptr, nullptr, cJoinVar, context, &variables_bound, &empty_interval_dict));
        }
        
        
    }

    auto planing_end = std::chrono::high_resolution_clock::now();
#ifdef TIME_DEBUG
    std::cout << "Planing time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(planing_end-planing_start).count() << " ms.\n";
    std::cout << "Real Planing time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(planing_end-planing2_start).count() << " ms.\n";
#endif

    // std::cout << "processGpuCount : " << processGpuCount << "\n";
    // plan.print();
    plan.execute(sparqlResult, processGpuCount);
#ifdef AUTO_PLANNER
    auto unixtime = std::chrono::system_clock::now();
    std::stringstream ss;
    ss << "plan_" << std::chrono::duration_cast<std::chrono::seconds>(unixtime.time_since_epoch()).count() << ".gv";
    planTree->writeGraphvizTreeFile(ss.str());
#endif
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << std::current_exception().__cxa_exception_type()->name() << std::endl;
    }
}

void QueryExecutor::printBounds() const {
    std::cout << "==== SELECTED BOUND ====\n";
    for (auto &vb: variables_bound) {
        std::cout << std::setw(7) << vb.first << " : [" << vb.second.first << "," << vb.second.second << "]\n";
    }
    std::cout << "========================\n";
}

void QueryExecutor::createVariableBound(SparqlQuery &sparqlQuery) {

    TYPEID min_val = std::numeric_limits<TYPEID>::min();
    TYPEID max_val = std::numeric_limits<TYPEID>::max();
    join_vars.clear();

    // 1. Find join variables
    std::map<std::string, int>& vars_count = sparqlQuery.getVarCountMap();
    for (auto &c : vars_count) {
        if (c.second > 1) {
#ifdef VERBOSE_DEBUG
            std::cout << "Join variable : " << c.first << " (" << c.second << ")\n";
#endif
            join_vars.insert(c.first);
            // 2. Create initial bound (0, inf) for each join variables
            variables_bound[c.first] = std::make_pair(min_val, max_val);
            variables_data_index[c.first] = DataIndex();
        }
    }
#ifdef VERBOSE_DEBUG
    std::cout << "\n************* PRE-SCAN *************\n";
#endif
    // 3. Thigten bound for each vars (single bound)
    bool zeroResult = false;
    for (size_t i = 0; i < sparqlQuery.getPatternNum(); ++i) {
#ifdef VERBOSE_DEBUG
        std::cout << "  [Squeeze pattern " << sparqlQuery.getPatternPtr(i)->toString() << '\n';
#endif
        squeezeQueryBound(sparqlQuery.getPatternPtr(i));
        zeroResult = zeroResult || (sparqlQuery.getPatternPtr(i)->estimate_rows == 0);
    }
    if (zeroResult) {
        // set all bound to zero (TODO: set only connected vertex)
        for (std::string jv : join_vars) variables_bound[jv] = std::make_pair(0, 0);
    }
#ifdef VERBOSE_DEBUG
    std::cout << "*********** END PRE-SCAN ***********\n\n";
#endif
}

/**
 * @brief Get data column of triple pattern (only 1 var)
 * @param pattern
 */
void QueryExecutor::updateEmptyIntervalDict(std::string &variable, std::pair<size_t, size_t>& data_offsts, TYPEID_HOST_VEC *data) {

    size_t maxLen = 0, secondMaxLen = 0;
    std::pair<size_t, size_t> maxPair, secondMaxPair;
    
    for (size_t i = data_offsts.first + 1; i < data_offsts.second; i++) {
        TYPEID current = *(data->begin() + i);
        TYPEID previous = *(data->begin() + i - 1);
        size_t diff = current - previous;
        if (diff < 20000) continue; // TODO:
        
        if (diff > maxLen) {
            secondMaxPair = maxPair;
            secondMaxLen = maxLen;
            maxLen = diff;
            maxPair = std::make_pair(previous + 1, current - 1);
        } else if (diff > secondMaxLen) {
            secondMaxLen = diff;
            secondMaxPair = std::make_pair(previous + 1, current - 1);
        }
    }
    
    if (secondMaxPair.first > maxPair.first) std::swap(secondMaxPair, maxPair);
    
    // Test print
    // std::cout << maxPair.first << ',' << maxPair.second << '\n';
    // std::cout << secondMaxPair.first << ',' << secondMaxPair.second << '\n';
    // std::cout << " ----- \n";
    
    empty_interval_dict.updateBound(variable, secondMaxPair.first, secondMaxPair.second, maxPair.first, maxPair.second);
}

/**
 * @brief Update bound for each variable in triple pattern (call from createVariableBound only)
 * @param pattern
 */
void QueryExecutor::squeezeQueryBound(TriplePattern *pattern) {
    switch (pattern->getVariableNum()) {
        case 1: squeezeQueryBound1Var(pattern); break;
        case 2: squeezeQueryBound2Var(pattern); break;
    }
}

/**
 * @brief Update bound for variable in 1 free variable triple pattern
 * @param pattern
 */
void QueryExecutor::squeezeQueryBound1Var(TriplePattern *pattern) {
    TYPEID_HOST_VEC *l1_index_values = nullptr, *l1_index_offsets  = nullptr;
    TYPEID_HOST_VEC *l1_index_values2 = nullptr, *l1_index_offsets2  = nullptr;
    TYPEID_HOST_VEC *l2_index_values = nullptr, *l2_index_offsets = nullptr, *data = nullptr, *data2 = nullptr;
    TYPEID_HOST_VEC *l2_data = nullptr, *l2_data2 = nullptr;
    TERM_TYPE ttype;
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
        v1 = pattern->getSubject(); ttype = TYPE_SUBJECT;
        id1 = pattern->getObjectId();
        id2 = pattern->getPredicateId();
    } else if (pattern->predicateIsVariable()) {
        if (!ENABLE_FULL_INDEX) assert(false); // partial index not support predicate as variable
        l1_index_values = this->vedasStorage->getSubjectIndexValues();
        l1_index_offsets = this->vedasStorage->getSubjectIndexOffsets();
        l2_index_values = this->vedasStorage->getSubjectObjectIndexValues();
        l2_index_offsets = this->vedasStorage->getSubjectObjectIndexOffsets();
        l1_index_values2 = this->vedasStorage->getObjectIndexValues();
        l1_index_offsets2 = this->vedasStorage->getObjectIndexOffsets();
        l2_data2 = this->vedasStorage->getOSdata();
        data = this->vedasStorage->getSOPdata();
        data2 = this->vedasStorage->getOSPdata();
        v1 = pattern->getPredicate(); ttype = TYPE_PREDICATE;
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
        v1 = pattern->getObject(); ttype = TYPE_OBJECT;
        id1 = pattern->getSubjectId();
        id2 = pattern->getPredicateId();
    }

    updateDataBound(pattern, l1_index_values, l1_index_offsets, l2_index_values, l2_index_offsets, l2_data, data, v1, id1, id2, ttype);
    updateDataBound(pattern, l1_index_values2, l1_index_offsets2, nullptr, nullptr,               l2_data2, data2, v1, id2, id1, ttype);
}

void QueryExecutor::updateBoundDict(std::map< std::string, std::pair<TYPEID, TYPEID> > *bound,
                                    std::string &variable, TYPEID lowerBound, TYPEID upperBound) {
    auto varBound = (*bound)[variable];
    if ((upperBound < varBound.first) || (lowerBound > varBound.second)) lowerBound = upperBound = 0;
    auto new_min = (lowerBound == 0)? 0 : std::max(varBound.first, lowerBound);
    auto new_max = std::min(varBound.second, upperBound);
#ifdef VERBOSE_DEBUG
    std::cout << "(D)UPDATE " << variable << " BOUND FROM [" << varBound.first << "," << varBound.second
              << "] TO [" << new_min << "," << new_max << "]\n";
#endif
    (*bound)[variable] = std::make_pair(new_min, new_max);
}

std::pair<size_t, size_t> QueryExecutor::findL2OffsetFromL1(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets, TYPEID id1, size_t n) {
    if (l1_index_values->size() == 0) return std::make_pair(0, 0);

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
    if (l2_index_offsets->size() == 0) return std::make_pair(0, 0);

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

// Update bound for 1 variable pattern
void QueryExecutor::updateDataBound(TriplePattern *pattern, TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                                    TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                    TYPEID_HOST_VEC *l2_data, TYPEID_HOST_VEC *data, std::string &v, TYPEID id1, TYPEID id2, TERM_TYPE term_type) {

    if (l1_index_values->size() == 0) {
#ifdef VERBOSE_DEBUG
        std::cout << "QueryExecutor::updateDataBound : empty l1_index_values\n";
#endif
        return;
    }

    auto l2_offst_pair = findL2OffsetFromL1(l1_index_values, l1_index_offsets, id1, data->size());
    std::pair<size_t, size_t> data_offst_pair = (l2_data != nullptr)?
                findDataOffsetFromL2(l2_data, l2_offst_pair.first, l2_offst_pair.second, id2, data->size()) :
                findDataOffsetFromL2(l2_index_values, l2_index_offsets, l2_offst_pair.first, l2_offst_pair.second, id2, data->size());

    // TODO: estimate again after bounding variable
    pattern->estimate_rows = data_offst_pair.second - data_offst_pair.first;

    auto lowerBound = *(data->begin() + data_offst_pair.first);
    auto upperBound = *(data->begin() + data_offst_pair.second - 1);
    if (variables_bound.count(v) > 0) {
        QueryExecutor::updateBoundDict(&variables_bound, v, lowerBound, upperBound);
    }
    
    // For 1 variable triple, find empty intervals
    /*if (l2_index_offsets == nullptr && join_vars.count(v)) {
        updateEmptyIntervalDict(v, data_offst_pair, data);
    }*/
}

void QueryExecutor::squeezeQueryBound2Var(TriplePattern *pattern) {

    TYPEID_HOST_VEC *l1_index_values = nullptr, *l1_index_offsets = nullptr;
    TYPEID_HOST_VEC *l2_index_values = nullptr, *l2_index_offsets = nullptr;
    TYPEID_HOST_VEC *l2_index_values2 = nullptr, *l2_index_offsets2 = nullptr;
    TYPEID_HOST_VEC *l2_data = nullptr, *l2_data2 = nullptr;
    size_t data_size = this->vedasStorage->getSOPdata()->size();
    std::string v1, v2;
    TYPEID id1;
    TERM_TYPE ttype1, ttype2;

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
            v1 = pattern->getObject(); ttype1 = TYPE_OBJECT;
            v2 = pattern->getPredicate(); ttype2 = TYPE_PREDICATE;
            id1 = pattern->getSubjectId();
            break;
        case 5:
            l1_index_values = this->vedasStorage->getPredicateIndexValues();
            l1_index_offsets = this->vedasStorage->getPredicateIndexOffsets();
            l2_data = this->vedasStorage->getPOdata();
            l2_data2 = this->vedasStorage->getPSdata();
            v1 = pattern->getObject(); ttype1 = TYPE_OBJECT;
            v2 = pattern->getSubject(); ttype2 = TYPE_SUBJECT;
            id1 = pattern->getPredicateId();
            break;
        case 6:
            l1_index_values = this->vedasStorage->getObjectIndexValues();
            l1_index_offsets = this->vedasStorage->getObjectIndexOffsets();
            l2_index_values = this->vedasStorage->getObjectPredicateIndexValues();
            l2_index_offsets = this->vedasStorage->getObjectPredicateIndexOffsets();
            l2_data2 = this->vedasStorage->getOSdata();
            v1 = pattern->getPredicate(); ttype1 = TYPE_PREDICATE;
            v2 = pattern->getSubject(); ttype2 = TYPE_SUBJECT;
            id1 = pattern->getObjectId();
            break;
        default:
            std::cout << "Pattern Bitmap is " << pattern->getVariableBitmap() << "\n";
            assert(false);
    }

    auto l2_offst = findL2OffsetFromL1(l1_index_values, l1_index_offsets, id1, data_size);

    if (join_vars.count(v1))
        updateL2Bound(pattern, v1, l2_index_values, l2_index_offsets, l2_data, l2_offst.first, l2_offst.second, ttype1, ttype2);
    if (join_vars.count(v2))
        updateL2Bound(pattern, v2, l2_index_values2, l2_index_offsets2, l2_data2, l2_offst.first, l2_offst.second, ttype2, ttype1);
}

void QueryExecutor::updateL2Bound(TriplePattern *pattern, std::string &var, TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                                  TYPEID_HOST_VEC *l2_data, TYPEID l1_offst_lower_bound, TYPEID l1_offst_upper_bound,
                                  TERM_TYPE ttype1, TERM_TYPE ttype2) {
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
    
    // TODO: estimate again after bounding variable
    // XXX: why end_offst2 == 0
    pattern->estimate_rows = (end_offst2 == 0)? 0 : end_offst2 - start_offst2;
    // std::cout << var << "\n";
    // std::cout << end_offst2 << " - " << start_offst2 << " = " << pattern->estimate_rows << '\n';

    if (variables_bound.count(var) > 0) {
        QueryExecutor::updateBoundDict(&variables_bound, var, *l2_begin, *(l2_end-1));
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
    bool is_predicates[1] = { false };
    if (pattern->subjectIsVariable()) {
#ifdef DEBUG
        std::cout << "Use [PO]S Index\n";
        std::cout << "Search for " << pattern->getSubject() << " " << pattern->getObjectId() << " " << pattern->getPredicateId() << "\n";
#endif
        is_predicates[0] = false;
        if (ENABLE_FULL_INDEX) {
            return new SelectQueryJob(
                    this->vedasStorage->getObjectIndexValues(),
                    this->vedasStorage->getObjectIndexOffsets(),
                    this->vedasStorage->getObjectPredicateIndexValues(),
                    this->vedasStorage->getObjectPredicateIndexOffsets(),
                    pattern->getSubject(), pattern->getObjectId(), pattern->getPredicateId(),
                    this->vedasStorage->getOPSdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDeviceOPSdata() : nullptr,
                    &variables_bound, &empty_interval_dict, is_predicates, context
                );
        } else {
            return new SelectQueryJob(
                    this->vedasStorage->getPredicateIndexValues(),
                    this->vedasStorage->getPredicateIndexOffsets(),
                    this->vedasStorage->getPredicateObjectIndexValues(),
                    this->vedasStorage->getPredicateObjectIndexOffsets(),
                    pattern->getSubject(), pattern->getPredicateId(), pattern->getObjectId(),
                    this->vedasStorage->getPOSdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDevicePOSdata() : nullptr,
                    &variables_bound, &empty_interval_dict, is_predicates, context
                );
        }
    } else if (pattern->predicateIsVariable()) {
        if (!ENABLE_FULL_INDEX) assert(false);
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
                    &variables_bound, &empty_interval_dict, is_predicates, context
                );
    } else {
#ifdef DEBUG
        std::cout << "Use [PS]O Index\n";
        std::cout << "Search for " << pattern->getObject() << " " << pattern->getSubjectId() << " " << pattern->getObjectId() << "\n";
#endif

        is_predicates[0] = false;
        if (ENABLE_FULL_INDEX) {
            return new SelectQueryJob(
                    this->vedasStorage->getSubjectIndexValues(),
                    this->vedasStorage->getSubjectIndexOffsets(),
                    this->vedasStorage->getSubjectPredicateIndexValues(),
                    this->vedasStorage->getSubjectPredicateIndexOffsets(),
                    pattern->getObject(), pattern->getSubjectId(), pattern->getPredicateId(),
                    this->vedasStorage->getSPOdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDeviceSPOdata() : nullptr,
                    &variables_bound, &empty_interval_dict, is_predicates, context
                );
        } else {
            return new SelectQueryJob(
                    this->vedasStorage->getPredicateIndexValues(),
                    this->vedasStorage->getPredicateIndexOffsets(),
                    this->vedasStorage->getPredicateSubjectIndexValues(),
                    this->vedasStorage->getPredicateSubjectIndexOffsets(),
                    pattern->getObject(), pattern->getPredicateId(), pattern->getSubjectId(),
                    this->vedasStorage->getPSOdata(),
                    this->vedasStorage->isPreload()? this->vedasStorage->getDevicePSOdata() : nullptr,
                    &variables_bound, &empty_interval_dict, is_predicates, context
                );
        }
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
            if (!ENABLE_FULL_INDEX) assert(false);
            // Default is [SPO]
            if (index_used == "SOP") {
                // std::cout << "Use [SOP] Index\n";
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
                            &variables_bound, &empty_interval_dict, is_predicates, is_second_var_used, context
                        );
            } else {
                // std::cout << "Use [SPO] Index\n";
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
                            &variables_bound, &empty_interval_dict, is_predicates, is_second_var_used, context
                        );
            }
        case 5:
            // Default is [PSO]
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
                            &variables_bound, &empty_interval_dict, is_predicates, is_second_var_used, context
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
                            &variables_bound, &empty_interval_dict, is_predicates, is_second_var_used, context
                        );
            }
        case 6:
            if (!ENABLE_FULL_INDEX) assert(false);
            if (index_used == "OPS") {
                // std::cout << "Use [OPS] Index\n";
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
                            &variables_bound, &empty_interval_dict, is_predicates, is_second_var_used, context
                        );
            } else {
                // std::cout << "Use [OSP] Index\n";
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
                            &variables_bound, &empty_interval_dict, is_predicates, is_second_var_used, context
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

void QueryExecutor::initTime() {
    QueryExecutor::load_dict_ms = 0.0;
    QueryExecutor::load_data_ms = 0.0;
    QueryExecutor::indexing_ms = 0.0;
    QueryExecutor::upload_ns = 0.0;
    QueryExecutor::scan_to_split_ns = 0.0;
    QueryExecutor::prescan_extra_ns = 0.0;
    QueryExecutor::eif_ns = 0.0;
    QueryExecutor::join_ns = 0.0;
    QueryExecutor::alloc_copy_ns = 0.0;
    QueryExecutor::swap_index_ns = 0.0;
    QueryExecutor::p2p_transfer_ns = 0.0;
    QueryExecutor::download_ns = 0.0;
    QueryExecutor::eliminate_duplicate_ns = 0.0;
    QueryExecutor::convert_to_id_ns = 0.0;
    QueryExecutor::convert_to_iri_ns = 0.0;
    QueryExecutor::eif_count = 0;
    QueryExecutor::exe_log.clear();
}
