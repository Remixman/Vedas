
#ifndef JOINGRAPH_H
#define JOINGRAPH_H

#include <set>
#include <unordered_map>
#include <vector>
#include "vedas.h"
#include "SparqlQuery.h"
#include "PlanTreeNode.h"
#include "QueryPlan.h"

class JoinGraph {
public:
    JoinGraph(SparqlQuery *sparqlQuery);
    void print() const;
    std::vector<int> findSeeds();
    void splitQuery(std::vector<std::vector<int>>& tpIds, int componentCount);
private:
    SparqlQuery *sparqlQuery;
    std::vector<std::vector<int>> adjacencyList;
    
    int getEdgeCount();
    bool isJoin(TriplePattern *tp1, TriplePattern *tp2);
    void bfs(int root, std::vector<int> & dists);
    void addEdge(int source, int destination);
};

#endif
