#include <cassert>
#include <cmath>
#include <climits>
#include <iostream>
#include <set>
#include <queue>
#include <vector>
#include "TriplePattern.h"
#include "JoinGraph.h"

JoinGraph::JoinGraph(SparqlQuery *sparqlQuery) {
    this->sparqlQuery = sparqlQuery;
    size_t patternSize = sparqlQuery->getPatternNum();
    this->adjacencyList.resize(patternSize);
    for (size_t i = 0; i < patternSize - 1; i++) {
        TriplePattern *tp1 = sparqlQuery->getPatternPtr(i);
        for (size_t k = i + 1; k < patternSize; k++) {
            TriplePattern *tp2 = sparqlQuery->getPatternPtr(k);
            if (isJoin(tp1, tp2)) addEdge(i, k);
        }
    }
}

bool JoinGraph::isJoin(TriplePattern *tp1, TriplePattern *tp2) {
    if (tp1->subjectIsVariable() && tp2->hasVariable(tp1->getSubject())) return true;
    if (tp1->predicateIsVariable() && tp2->hasVariable(tp1->getPredicate())) return true;
    if (tp1->objectIsVariable() && tp2->hasVariable(tp1->getObject())) return true;
    return false;
}

std::vector<int> JoinGraph::findSeeds() {
    // bts with random node (use 0)
    std::vector<int> dists(adjacencyList.size(), 0);
    bfs(0, dists);
    
    // Find fisrt node with max distance
    int maxDist = 0, maxNode = 0;
    for (int i = 0; i < dists.size(); i++) {
        if (dists[i] > maxDist) {
            maxDist = dists[i];
            maxNode = i;
        }
    }
    
    dists.resize(adjacencyList.size(), 0);
    bfs(maxNode, dists);
    
    // Find fisrt node with max distance
    int maxDist2 = 0, maxNode2 = maxNode;
    for (int i = 0; i < dists.size(); i++) {
        if (dists[i] > maxDist2) {
            maxDist2 = dists[i];
            maxNode2 = i;
        }
    }
    
    // std::vector<int> seeds = { maxNode, maxNode2 };
    return { maxNode, maxNode2 };
}

struct SubGraphData {
    double estQueryTime;
    std::set<int> vertices;
    std::set<int> activeVertices;
};

void JoinGraph::splitQuery(std::vector<std::vector<int>>& tpIds, int componentCount = 2) {
    std::vector<int> seeds = findSeeds();
    SubGraphData sgd[seeds.size()];
    tpIds.resize(2);

    int edgeCount = getEdgeCount();
    std::set<int> coveredJoin, coveredTp;
    int coveredJoinCount = 0;
    
    for (int i = 0; i < seeds.size(); ++i) {
        sgd[i].vertices.insert(seeds[i]);
        sgd[i].activeVertices.insert(seeds[i]);
        sgd[i].estQueryTime = sparqlQuery->getPatternPtr(i)->estimate_rows; // TODO: change to estimate time not rows
        coveredTp.insert(seeds[i]);
    }

    // find min subgraph
    int minTime = INT_MAX, minI = 0;
    for (int i = 0; i < seeds.size(); ++i) {
        if (sgd[i].estQueryTime < minTime) {
            minTime = sgd[i].estQueryTime;
            minI = i;
        }
    }
    int si = minI;
    while (coveredJoinCount < edgeCount) {
        // std::cout << "si = " << si << '\n';
        
        // for vertices in active set, find adjacency vertices
        std::set<int> newActives;
        for (auto v: sgd[si].activeVertices) {
            for (auto w: adjacencyList[v]) {
                if (coveredTp.count(w) == 0) {
                    newActives.insert(w);
                    coveredTp.insert(w);
                    // std::cout << "\tadd " << w << '\n';
                }
            }
        }
        
        // Allow use vertices in coveredTp for last iteration
        bool last = false;
        if (coveredTp.size() == adjacencyList.size()) {
            for (auto v: sgd[si].activeVertices) {
                for (auto w: adjacencyList[v]) {
                    if (sgd[si].vertices.count(w) == 0 && newActives.count(w) == 0) {
                        // std::cout << "Last vertex : " << w << '\n';
                        newActives.insert(w);
                        last = true;
                        break;
                    }
                }
                if (last) break;
            }
        }
        
        int activeLinkCount = 0;
        for (auto v: newActives) {
            for (auto w: adjacencyList[v]) {
                if (sgd[si].vertices.count(w) > 0) {
                    coveredJoinCount++;
                } else if (newActives.count(w) > 0) {
                    activeLinkCount++;
                }
            }
        }
        assert(activeLinkCount % 2 == 0);
        coveredJoinCount += (activeLinkCount / 2);
        // std::cout << "coveredJoinCount : " << coveredJoinCount <<  " , edgeCount : " << edgeCount <<'\n';

        sgd[si].activeVertices = newActives;
        for (auto v: newActives) sgd[si].vertices.insert(v);
        
        si++; if (si >= seeds.size()) si = 0;
        
        // int hh; std::cin >> hh;
    }
    
    for (int i = 0; i < seeds.size(); ++i) {
        std::cout << "Set " << i << "\n\t";
        for (auto v: sgd[i].vertices) {
            tpIds[i].push_back(v);
            std::cout << v << ' ';
        }
        std::cout << '\n';
    }
}



int JoinGraph::getEdgeCount() {
    int edgeNum = 0;
    for (auto link: adjacencyList) edgeNum += link.size();
    return edgeNum / 2;
}

void JoinGraph::bfs(int root, std::vector<int> & dists) {
    std::set<int> visited;
    std::queue<int> q;
    visited.insert(root);
    q.push(root);
    
    while (!q.empty()) {
        int currentNode = q.front(); q.pop();
        int level = dists[currentNode];
        for (int v: adjacencyList[currentNode]) {
            if (visited.count(v) == 0) {
                visited.insert(v);
                dists[v] = level + 1;
                q.push(v);
            }
        }
    }
}

void JoinGraph::print() const {
    std::cout << "Query Graph\n";
    for (int i = 0; i < adjacencyList.size(); ++i) {
        std::cout << '\t' << i << " => ";
        for (auto destNode: adjacencyList[i]) {
            std::cout << destNode << ' ';
        }
        std::cout << '\n';
    }
}

void JoinGraph::addEdge(int source, int destination) {
  adjacencyList[source].push_back(destination);
  adjacencyList[destination].push_back(source);
}
