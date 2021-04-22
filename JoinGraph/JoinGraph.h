#ifndef JOINGRAPH_H
#define JOINGRAPH_H

#include <map>
#include <utility>
#include <list>
#include <queue>
#include <vector>
#include "../ExecutionPlanTree.h"
#include "../SparqlQuery.h"
#include "TriplePatternNode.h"

typedef std::pair<std::string, int> varDegreePair;

class VariableDegreePairCompare {
public:
    bool operator() (varDegreePair a, varDegreePair b) {
        return a.second < b.second;
    }
};

class JoinGraph {
public:
    JoinGraph(SparqlQuery *sparqlQuery);
    ExecutionPlanTree* createPlan();
    ExecutionPlanTree* createPlanDP();
private:
    void addNewNode(std::string variable, GraphNode *gn);
    void replaceNode(GraphNode *gn, GraphNode *replaceGn);
    void createLinkToVarNode(GraphNode *fromGn, const std::string &variable);
    size_t maxDegreeVarNodeIndex() const;

    std::string getUsedIndex(TriplePattern *tp, std::string &joinVar);
    std::map<GraphNode*, size_t> gNodeDict;
    std::map<std::string, size_t> varNodeDict;
    std::map<size_t, std::string> nodeVarDict;
    std::vector<std::list<GraphNode*>> adjList;
    std::vector<bool> removeMask;

    std::vector<std::string> variables;
    std::map<std::string, unsigned long long> varBitDict; // e.g. ?x = 00001, ?y = 00010, ?z = 00100

    // For plan tree search (DP)
    std::map<std::string, std::string> joinSymDict; // e.g. c = a join b, joinSymDict["([a][b])"] = "{c}";
    std::map<std::string, std::string> reverseJoinSymDict; // e.g. reverseJoinSymDict["{c}"] = "([a][b])";
    std::map<std::string, unsigned> planCostDict;   // e.g. planCostDict["(({f}(a)){k})"] = 50;
    void searchBinaryPlanDP();
    void searchBinaryPlanDPRecursive(std::vector<bool> &varMask);
};

#endif
