#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "TriplePattern.h"
#include "QueryGraph.h"


GraphEdge::GraphEdge(const std::string id, const std::string s, const std::string o) {
  initialize(id.c_str(), s, o);
}

GraphEdge::GraphEdge(const char* id, const std::string s, const std::string o) {
  initialize(id, s, o);
}

void GraphEdge::initialize(const char* id, const std::string s, const std::string o) {
  this->id = id;
  tmpTriple = s + " " + id + " " + o;
  if (s[0] == '?') {
      vars.insert(s);
      indexVar1 = s;
  }
  if (o[0] == '?') { 
      vars.insert(o);
      if (indexVar1 == "") indexVar1 = o;
      else indexVar2 = o;
  }
}

bool GraphEdge::hasVariable(const std::string v) {
    return vars.count(v) > 0;
}

QueryGraph::QueryGraph(SparqlQuery *sparqlQuery) {
    size_t patternSize = sparqlQuery->getPatternNum();
    for (size_t i = 0; i < patternSize; i++) {
        TriplePattern *tp = sparqlQuery->getPatternPtr(i);
        
        std::string subject = tp->subjectIsVariable() ? tp->getSubject() : std::to_string(tp->getSubjectId());
        std::string object = tp->objectIsVariable() ? tp->getObject() : std::to_string(tp->getObjectId());
        GraphEdge *edgeData = new GraphEdge(std::to_string(tp->getPredicateId()), subject, object);
        edgeData->tp = tp;
        edgeData->estimateCardinality = tp->estimate_rows * tp->getVariableNum();
        // std::cout << "\tTP : " << tp->toString() << " : " << tp->estimate_rows << " rows\n";
        // std::cout << " add edge " <<  subject << ' ' << object << '\n';
        this->addEdge(subject, object, edgeData);
    }
    /*std::set<std::string> joinVariables;

    variables = sparqlQuery->getVariables();
    for (unsigned i = 0; i < variables.size(); i++) {
        unsigned long long ull = 1;
        varBitDict[variables[i]] = (ull << i);
    }*/
}

PlanTreeNode* QueryGraph::generateQueryPlan() {
    PlanTreeNode* root = nullptr;
    while (adjacencyList.size() > 2) {
        auto starCenter = findBestStarNode();
        // std::cout << "Star : " << starCenter.first << " , join with " << starCenter.second << '\n';
        root = generateJoinPlanForNode(starCenter.first, starCenter.second);
    }
    
    // PlanTreeNode::printBT(root);
    return root;
}

void QueryGraph::print() const {
    std::cout << "Query Graph\n";
    for (auto srcNode: adjacencyList) {
        std::cout << '\t' << srcNode.first << " => ";
        for (auto destNode: srcNode.second) {
            std::cout << destNode.first << '(' << destNode.second->id << ") ";
        }
        std::cout << '\n';
    }
}

void QueryGraph::addEdge(std::string source, std::string destination, GraphEdge* edgeData) {
  adjacencyList[source].push_back(make_pair(destination, edgeData));
  adjacencyList[destination].push_back(make_pair(source, edgeData));
}

bool QueryGraph::edgeExist(const std::string v1, const std::string v2) {
  for (auto edge: adjacencyList[v1])
      if (edge.first == v2) return true;
  return false;
}

void QueryGraph::mergeEdge(const std::string joinVar, const std::string end1, const std::string end2) {
  // std::cout << "MERGE EDGE (" << joinVar << "," << end2 << ") to (" << joinVar << "," << end1 << ")\n";
  // Merge edge (joinVar, end2) to (joinVar, end1)
  // XXX: assume no cycle in graph
  for (auto edge : adjacencyList[end2]) {
      if (!edgeExist(end1, edge.first)) {
        //   std::cout << " ADD EDGE " << end1 << " <-> " << edge.first << '\n';
          addEdge(end1, edge.first, edge.second);
      }
      for (auto it = adjacencyList[edge.first].rbegin(); it != adjacencyList[edge.first].rend(); ++it) {
          if (it->first == end2) {
            //   std::cout << " REMOVE EDGE " << edge.first << " -> " << end2 << '\n';
              adjacencyList[edge.first].erase(--(it.base()));
              break;
          }
      }
  }
  
//   std::cout << " REMOVE EDGE " << end2 << " -> *\n";
  adjacencyList.erase(end2); // remove this edge
}

unsigned long long QueryGraph::sumOfCardinality(std::string vertex) {
  unsigned long long sum = 0;
  for (auto edge: adjacencyList[vertex])
      sum += edge.second->estimateCardinality;
  return sum;
}

std::tuple<size_t, size_t, std::string> QueryGraph::sameMaxDegAndTotalCard(std::vector< std::pair<std::string, GraphEdge*> > &links) {
    std::unordered_map<std::string, size_t> varCount;
    std::unordered_map<std::string, size_t> varEstCard;
    
    for (auto node: links) {
        for (auto var: node.second->vars) {
            if (varCount.count(var)) {
                varCount[var] += 1;
                varEstCard[var] += node.second->estimateCardinality;
            } else {
                varCount[var] = 1;
                varEstCard[var] = node.second->estimateCardinality;
            }
        }
    }
    
    size_t maxDeg = 0, maxDegMinCard = 0;
    std::string bestVar;
    for (auto v: varCount) {
        size_t estCard = varEstCard[v.first];
        if (v.second > maxDeg || (v.second == maxDeg && estCard < maxDegMinCard)) {
            maxDeg = v.second;
            maxDegMinCard = estCard;
            bestVar = v.first;
        }
    }
    
    return std::make_tuple(maxDeg, maxDegMinCard, bestVar);
}

std::pair<std::string, std::string> QueryGraph::findBestStarNode() {
    std::string bestNodeName, bestNodeJoinVar;
    size_t maxDegree = 0, maxDegCardSum = 0;
    for (auto node: adjacencyList) {
        
        std::tuple<size_t, size_t, std::string> varDegCard = sameMaxDegAndTotalCard(node.second);
        std::string nodeName = node.first;
        std::string joinVar = std::get<2>(varDegCard);
        size_t degree = std::get<0>(varDegCard);
        size_t totalEstCard = std::get<1>(varDegCard);
        // std::cout << '\t' << nodeName << "," << joinVar << "," << degree << "," << totalEstCard << '\n';
        
        if (degree > maxDegree || (degree == maxDegree && totalEstCard < maxDegCardSum)) {
            maxDegree = degree;
            bestNodeName = nodeName;
            maxDegCardSum = totalEstCard;
            bestNodeJoinVar = joinVar;
        }
    }
    return std::make_pair(bestNodeName, bestNodeJoinVar);
}

bool compareEdge(std::pair<std::string, GraphEdge*> a, std::pair<std::string, GraphEdge*> b) {
    if (a.second->estimateCardinality == b.second->estimateCardinality) {
        return a.second->estimateCardinality < b.second->estimateCardinality;
    }
    return a.second->estimateCardinality < b.second->estimateCardinality;
}

PlanTreeNode* QueryGraph::createUploadJobNode(const std::string tripleStr, TriplePattern *tp, const std::string toJoinVar) {
    PlanTreeNode *node = new PlanTreeNode();
    node->op = UPLOAD;
    node->tp = tp;
    node->debugName = tripleStr;
    node->var = toJoinVar;
    return node;
}

PlanTreeNode* QueryGraph::createJoinJobNode(const std::string joinVar, PlanTreeNode *child1, PlanTreeNode *child2, bool reuseVar) {
    PlanTreeNode *node = new PlanTreeNode();
    node->op = JOIN;
    node->child1 = child1;
    node->child2 = child2;
    node->debugName = joinVar;
    node->var = joinVar;
    node->reuseVar = reuseVar;
    return node;
}

PlanTreeNode* QueryGraph::createIndexSwapJobNode(PlanTreeNode *child, std::string swapTo) {
    PlanTreeNode *node = new PlanTreeNode();
    node->op = INDEXSWAP;
    node->child1 = child;
    node->debugName = swapTo;
    node->var = swapTo;
    return node;
}

PlanTreeNode* QueryGraph::generateJoinPlanForNode(const std::string vertex, const std::string joinVar) {
    auto &adjList = adjacencyList[vertex];
    std::sort(adjList.begin(), adjList.end(), compareEdge);

    size_t firstIndex = 0;
    while (!adjList[firstIndex].second->hasVariable(joinVar)) ++firstIndex;

    std::string indexVar = joinVar;
    
    size_t lastJoinIdx = 0;
    for (size_t i = firstIndex; i < adjList.size(); ++i) {
        if (!adjList[i].second->hasVariable(joinVar)) continue;
        lastJoinIdx = i;
    }
  
    for (size_t i = firstIndex; i < adjList.size(); ++i) {
        if (!adjList[i].second->hasVariable(joinVar)) continue;
    
      // data still on main memory, upload to GPU memory
    //   std::cout << "   (" << vertex << "," << adjList[i].first << ") ^ Est.Card. : " 
        //   << adjList[i].second->estimateCardinality << "\n";
        if (!adjList[i].second->treeNode) {
            adjList[i].second->treeNode = createUploadJobNode(adjList[i].second->tmpTriple, adjList[i].second->tp, indexVar);
        }

        if (i > firstIndex) {
            GraphEdge *edgeData1 = adjList[firstIndex].second;
            GraphEdge *edgeData2 = adjList[i].second;
            
            if (edgeData1->indexVar1 != indexVar && edgeData1->indexVar2 != indexVar) {
                adjList[firstIndex].second->treeNode = createIndexSwapJobNode(adjList[firstIndex].second->treeNode, indexVar);
                adjList[firstIndex].second->indexVar1 = indexVar;
                adjList[firstIndex].second->indexVar2 = "";
            }
            
            if (edgeData2->indexVar1 != indexVar && edgeData2->indexVar2 != indexVar) {
                adjList[i].second->treeNode = createIndexSwapJobNode(adjList[i].second->treeNode, indexVar);
            }
            
            adjList[firstIndex].second->treeNode = 
                createJoinJobNode(indexVar, adjList[firstIndex].second->treeNode, adjList[i].second->treeNode, i != lastJoinIdx);
            double sigma = 0.000001;
            if (adjList[firstIndex].second->estimateCardinality < 1000 && adjList[i].second->estimateCardinality < 1000) {
                sigma = 0.001;
            } else if (adjList[firstIndex].second->estimateCardinality < 1000 || adjList[i].second->estimateCardinality < 1000) {
                sigma = 0.0001;
            }
            adjList[firstIndex].second->estimateCardinality *= ceil(adjList[i].second->estimateCardinality * sigma);
            for (auto v: adjList[i].second->vars) {
                adjList[firstIndex].second->vars.insert(v);
            }
        }
    }
  
    for (int i = adjList.size() - 1; i > firstIndex; --i) {
        if (!adjList[i].second->hasVariable(joinVar)) continue;
        mergeEdge(vertex, adjList[firstIndex].first, adjList[i].first);
    }
  
  adjList[firstIndex].second->indexVar1 = indexVar;
  adjList[firstIndex].second->indexVar2 = "";
  
  assert(adjList[firstIndex].second->treeNode != nullptr);
  return adjList[firstIndex].second->treeNode;
}