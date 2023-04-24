
#ifndef QUERYGRAPH_H
#define QUERYGRAPH_H

#include <set>
#include <unordered_map>
#include <vector>
#include "vedas.h"
#include "SparqlQuery.h"
#include "PlanTreeNode.h"
#include "QueryPlan.h"
#include "EmptyIntervalDict.h"

struct GraphEdge {
    GraphEdge(const std::string id, const std::string s, const std::string o);
    GraphEdge(const char* id, const std::string s, const std::string o);
    void initialize(const char* id, const std::string s, const std::string o);
    bool hasVariable(const std::string v);
    std::string id;
    std::string indexVar1;
    std::string indexVar2;
    std::set<std::string> vars;
    std::string tmpTriple;
    TriplePattern *tp { nullptr };
    size_t estimateCardinality;
    PlanTreeNode *treeNode { nullptr };
};

class QueryGraph {
public:
    QueryGraph(SparqlQuery *sparqlQuery);
    PlanTreeNode* generateQueryPlan();
    void print() const;
private:
    std::unordered_map<std::string, std::vector<std::pair<std::string, GraphEdge*>>> adjacencyList;
    
    void addEdge(std::string source, std::string destination, GraphEdge* edgeData);
    bool edgeExist(const std::string v1, const std::string v2);
    void mergeEdge(const std::string joinVar, const std::string end1, const std::string end2);
    unsigned long long sumOfCardinality(std::string vertex);
    std::tuple<size_t, size_t, std::string> sameMaxDegAndTotalCard(std::vector< std::pair<std::string, GraphEdge*> > &links);
    std::pair<std::string, std::string> findBestStarNode();
    PlanTreeNode* generateJoinPlanForNode(const std::string vertex, const std::string joinVar);
    PlanTreeNode* createUploadJobNode(const std::string tripleStr, TriplePattern *tp, const std::string toJoinVar);
    PlanTreeNode* createJoinJobNode(const std::string joinVar, PlanTreeNode *child1, PlanTreeNode *child2, bool reuseVar);
    PlanTreeNode* createIndexSwapJobNode(PlanTreeNode *child, std::string swapTo);
};

#endif
