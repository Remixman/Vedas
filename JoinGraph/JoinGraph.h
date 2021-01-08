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
};

#endif
