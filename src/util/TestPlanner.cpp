#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

struct PlanTreeNode {
    std::string op;
    PlanTreeNode *parent { nullptr };
    PlanTreeNode *child1 { nullptr };
    PlanTreeNode *child2 { nullptr };
    std::string debugName;
    size_t columnToSwap;
};

PlanTreeNode* createUploadJobNode(const std::string &tripleStr/* TODO: other upload information */) {
    PlanTreeNode *node = new PlanTreeNode();
    node->op = "UPLOAD";
    node->debugName = tripleStr;
    return node;
}

PlanTreeNode* createJoinJobNode(const std::string &joinVar, PlanTreeNode *child1, PlanTreeNode *child2/* TODO: other upload information */) {
    PlanTreeNode *node = new PlanTreeNode();
    node->op = "JOIN";
    node->child1 = child1;
    node->child2 = child2;
    node->debugName = joinVar;
    return node;
}

PlanTreeNode* createIndexSwapJobNode(PlanTreeNode *child, std::string swapTo/* TODO: other upload information */) {
    PlanTreeNode *node = new PlanTreeNode();
    node->op = "INDEX-SWAP";
    node->child1 = child;
    node->debugName = swapTo;
    return node;
}

void printBT(const std::string& prefix, const PlanTreeNode* node, bool isLeft) {
    if (node != nullptr) {
        std::cout << prefix;

        std::cout << (isLeft ? "├──" : "└──" );

        // print the value of the node
        std::cout << node->op << "  " << node->debugName << std::endl;

        // enter the next tree level - left and right branch
        printBT( prefix + (isLeft ? "│   " : "    "), node->child1, true);
        printBT( prefix + (isLeft ? "│   " : "    "), node->child2, false);
    }
}

void printBT(const PlanTreeNode* node) {
    printBT("", node, false);    
}

void postorderTraversal(PlanTreeNode* root) {
    if (root == nullptr) return;

    if (root->child1 != nullptr) postorderTraversal(root->child1);
    if (root->child2 != nullptr) postorderTraversal(root->child2);
    std::cout << root->op << " (" << root->debugName << ")\n";
}

struct GraphEdge {
    GraphEdge(std::string &id, const std::string &s, const std::string &o) {
        initialize(id.c_str(), s, o);
    }
    GraphEdge(const char* id, const std::string &s, const std::string &o) {
        initialize(id, s, o);
    }
    void initialize(const char* id, const std::string &s, const std::string &o) {
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
        estimateCardinality = (vars.size() == 2) ? 40000 : 900;
    }
    bool hasVariable(const std::string v) {
        return vars.count(v) > 0;
    }
    std::string id;
    std::string indexVar1;
    std::string indexVar2;
    std::set<std::string> vars;
    std::string tmpTriple;
    // TODO: triple
    size_t estimateCardinality;
    PlanTreeNode *treeNode;
};

bool compareEdge(pair<string, GraphEdge*> a, pair<string, GraphEdge*> b) {
    if (a.second->estimateCardinality == b.second->estimateCardinality) {
        return a.second->estimateCardinality < b.second->estimateCardinality;
    }
    return a.second->estimateCardinality < b.second->estimateCardinality;
}

class LabeledGraph {
private:
    std::unordered_map<string, vector<pair<string, GraphEdge*>>> adjacencyList;
    std::vector<GraphEdge*> edgeDataList;
public:
    ~LabeledGraph() {
        /*for (GraphEdge* edgeData: edgeDataList) {
            delete edgeData;
        }*/
    }
    void addEdge(string source, string destination, GraphEdge* edgeData) {
        adjacencyList[source].push_back(make_pair(destination, edgeData));
        adjacencyList[destination].push_back(make_pair(source, edgeData));

        edgeDataList.push_back(edgeData);
    }

    bool edgeExist(const std::string v1, const std::string v2) {
        for (auto edge: adjacencyList[v1]) {
            if (edge.first == v2) return true;
        }
        return false;
    }
    
    void mergeEdge(const std::string joinVar, const std::string end1, const std::string end2) {
        std::cout << "MERGE EDGE (" << joinVar << "," << end2 << ") to (" << joinVar << "," << end1 << ")\n";
        // Merge edge (joinVar, end2) to (joinVar, end1)
        // XXX: assume no cycle in graph
        for (auto edge : adjacencyList[end2]) {
            if (!edgeExist(end1, edge.first)) {
                std::cout << " ADD EDGE " << end1 << " <-> " << edge.first << '\n';
                addEdge(end1, edge.first, edge.second);
            }
            for (auto it = adjacencyList[edge.first].rbegin(); it != adjacencyList[edge.first].rend(); ++it) {
                if (it->first == end2) {
                    std::cout << " REMOVE EDGE " << edge.first << " -> " << end2 << '\n';
                    adjacencyList[edge.first].erase(--(it.base()));
                    break;
                }
            }
        }
        
        std::cout << " REMOVE EDGE " << end2 << " -> *\n";
        adjacencyList.erase(end2); // remove this edge
    }

    unsigned long long sumOfCardinality(string nodeName) {
        unsigned long long sum = 0;
        for (auto edge: adjacencyList[nodeName]) {
            sum += edge.second->estimateCardinality;
        }
        return sum;
    }
    
    std::tuple<size_t, size_t, std::string> sameMaxDegAndTotalCard(std::vector< std::pair<std::string, GraphEdge*> > &links) {
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
        
        std::vector<std::tuple<size_t, size_t, std::string>> varsCountAndCard;
        for (auto v: varCount) {
            varsCountAndCard.push_back(std::make_tuple(v.second, varEstCard[v.first], v.first));
        }
        
        sort(varsCountAndCard.begin(), varsCountAndCard.end(), [](const std::tuple<size_t, size_t, std::string>& a, const std::tuple<size_t, size_t, std::string>& b) {
            if (std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) > std::get<0>(b);
            } else {
                return std::get<1>(a) < std::get<1>(b);
            }
        });
        
        return varsCountAndCard[0];
    }

    std::pair<std::string, std::string> findBestStarNode() {
        std::string bestNodeName, bestNodeJoinVar;
        size_t maxDegree = 0, maxDegCardSum = 0;
        for (auto node: adjacencyList) {
            
            std::tuple<size_t, size_t, std::string> varDegCard = sameMaxDegAndTotalCard(node.second);
            std::string nodeName = node.first;
            std::string joinVar = std::get<2>(varDegCard);
            size_t degree = std::get<0>(varDegCard);
            size_t totalEstCard = std::get<1>(varDegCard);
            // std::cout << nodeName << "," << joinVar << "," << degree << "," << totalEstCard << '\n';
            
            if (degree > maxDegree || (degree == maxDegree && totalEstCard < maxDegCardSum)) {
                maxDegree = degree;
                bestNodeName = nodeName;
                maxDegCardSum = totalEstCard;
                bestNodeJoinVar = joinVar;
            }
        }
        return std::make_pair(bestNodeName, bestNodeJoinVar);
    }

    PlanTreeNode* generateJoinPlanForNode(const std::string vertex, const std::string joinVar) {
        auto &adjList = adjacencyList[vertex];
        std::sort(adjList.begin(), adjList.end(), compareEdge);
        
        size_t firstIndex = 0;
        while (!adjList[firstIndex].second->hasVariable(joinVar)) ++firstIndex;

        std::string indexVar = joinVar;
        
        for (size_t i = firstIndex; i < adjList.size(); ++i) {
            if (!adjList[i].second->hasVariable(joinVar)) continue;
            // data still on main memory, upload to GPU memory
            if (!adjList[i].second->treeNode) {
                adjList[i].second->treeNode = createUploadJobNode(adjList[i].second->tmpTriple);
            }

            if (i > firstIndex) {
                GraphEdge *edgeData1 = adjList[firstIndex].second;
                GraphEdge *edgeData2 = adjList[i].second;
                
                if (edgeData1->indexVar1 != indexVar && edgeData1->indexVar2 != indexVar) {
                    adjList[firstIndex].second->treeNode = createIndexSwapJobNode(adjList[firstIndex].second->treeNode, indexVar);
                    adjList[firstIndex].second->indexVar1 = indexVar;
                    adjList[firstIndex].second->indexVar2 = "";
                }
                
                if (edgeData2->indexVar1 != indexVar && edgeData2->indexVar2 != indexVar)
                    adjList[i].second->treeNode = createIndexSwapJobNode(adjList[i].second->treeNode, indexVar);
                
                adjList[firstIndex].second->treeNode = 
                    createJoinJobNode(indexVar, adjList[firstIndex].second->treeNode, adjList[i].second->treeNode);
                adjList[firstIndex].second->estimateCardinality *= ceil(adjList[i].second->estimateCardinality * 0.000001);
                for (auto v: adjList[i].second->vars) {
                    adjList[firstIndex].second->vars.insert(v);
                }
            }
        }
        
        for (int i = adjList.size() - 1; i > firstIndex; --i) {
            if (!adjList[i].second->hasVariable(joinVar)) continue;
            std::cout << "Merge i = " << i << " of " << adjList.size() << '\n';
            if (adjList.size() == 0) break;
            mergeEdge(vertex, adjList[firstIndex].first, adjList[i].first);
        }
        
        adjList[firstIndex].second->indexVar1 = indexVar;
        adjList[firstIndex].second->indexVar2 = "";
        
        assert(adjList[firstIndex].second->treeNode != nullptr);
        return adjList[firstIndex].second->treeNode;
    }

    PlanTreeNode* generateQueryPlan() {
        PlanTreeNode* root = nullptr;
        while (adjacencyList.size() > 2) {
            auto starCenter = findBestStarNode();
            std::cout << "Star : " << starCenter.first << " , join with " << starCenter.second << '\n';
            root = generateJoinPlanForNode(starCenter.first, starCenter.second);
            // printBT(root);
            // std::cout << '\n';

            // print();
            // int i;
            // std::cin >> i;
            
            // break;
        }
        
        // adjacencyList[0]
        printBT(root);
        std::cout << '\n';
        postorderTraversal(root);
        std::cout << '\n';
        
        return root;
    }

    void print() {
        std::cout << "Labeled Graph\n";
        for (auto srcNode: adjacencyList) {
            std::cout << '\t' << srcNode.first << " => ";
            for (auto destNode: srcNode.second) {
                std::cout << destNode.first << '(' << destNode.second->id << ") ";
            }
            std::cout << '\n';
        }
    }
};


class QueryGraph: public LabeledGraph {
public:
    QueryGraph(/* TriplePattern List */) {};

private:

};



void testL2() {
    std::cout << "==================== TEST L2 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v1", "7", new GraphEdge("1000", "?v1", "7"));
    graph.addEdge("?v2", "?v1", new GraphEdge("1002", "?v2", "?v1"));
    graph.addEdge("?v2", "99", new GraphEdge("1004", "?v2", "99"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST L2 =====================\n\n";
}

void testS2() {
    std::cout << "==================== TEST S2 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v2", new GraphEdge("1001", "?v0", "?v2"));
    graph.addEdge("?v0", "?v3", new GraphEdge("1003", "?v0", "?v3"));
    graph.addEdge("?v0", "?v4", new GraphEdge("1005", "?v0", "?v4"));
    graph.addEdge("?v0", "?v1", new GraphEdge("1007", "?v0", "?v1"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST S2 =====================\n\n";
}

void testS5() {
    std::cout << "==================== TEST S5 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v2", new GraphEdge("1001", "?v0", "?v2"));
    graph.addEdge("?v0", "?v3", new GraphEdge("1003", "?v0", "?v3"));
    graph.addEdge("?v0", "15", new GraphEdge("1005", "?v0", "15"));
    graph.addEdge("?v0", "26", new GraphEdge("1007", "?v0", "26"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST S5 =====================\n\n";
}

void testF1() {
    std::cout << "==================== TEST F1 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v2", new GraphEdge("1001", "?v0", "?v2"));
    graph.addEdge("?v0", "55", new GraphEdge("1003", "?v0", "55"));
    graph.addEdge("?v3", "?v4", new GraphEdge("1005", "?v3", "?v4"));
    graph.addEdge("?v3", "?v0", new GraphEdge("1007", "?v3", "?v0"));
    graph.addEdge("?v3", "?v5", new GraphEdge("1009", "?v3", "?v5"));
    graph.addEdge("?v3", "2", new GraphEdge("1011", "?v3", "2"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST F1 =====================\n\n";
}

void testF3() {
    std::cout << "==================== TEST F3 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v2", new GraphEdge("1001", "?v0", "?v2"));
    graph.addEdge("?v0", "?v1", new GraphEdge("1003", "?v0", "?v1"));
    graph.addEdge("?v0", "?v3", new GraphEdge("1005", "?v0", "?v3"));
    graph.addEdge("?v5", "?v0", new GraphEdge("1007", "?v5", "?v0"));
    graph.addEdge("?v4", "?v5", new GraphEdge("1009", "?v4", "?v5"));
    graph.addEdge("?v5", "?v6", new GraphEdge("1011", "?v5", "?v6"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST F3 =====================\n\n";
}

void testF4() {
    std::cout << "==================== TEST F4 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v1", new GraphEdge("1001", "?v0", "?v1"));
    graph.addEdge("?v0", "?v4", new GraphEdge("1002", "?v0", "?v4"));
    graph.addEdge("?v0", "?v8", new GraphEdge("1003", "?v0", "?v8"));
    graph.addEdge("?v2", "?v0", new GraphEdge("1004", "?v2", "?v0"));
    graph.addEdge("?v0", "3", new GraphEdge("1005", "?v0", "3"));
    graph.addEdge("?v1", "?v5", new GraphEdge("1006", "?v1", "?v5"));
    graph.addEdge("?v1", "?v6", new GraphEdge("1007", "?v1", "?v6"));
    graph.addEdge("?v1", "?v7", new GraphEdge("1008", "?v1", "?v7"));
    graph.addEdge("?v1", "999", new GraphEdge("1009", "?v1", "999"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST F4 =====================\n\n";
}

void testF5() {
    std::cout << "==================== TEST F5 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v3", new GraphEdge("1001", "?v0", "?v3"));
    graph.addEdge("?v0", "?v1", new GraphEdge("1003", "?v0", "?v1"));
    graph.addEdge("?v0", "?v4", new GraphEdge("1004", "?v0", "?v4"));
    graph.addEdge("2", "?v0", new GraphEdge("1005", "2", "?v0"));
    graph.addEdge("?v1", "?v5", new GraphEdge("1006", "?v1", "?v5"));
    graph.addEdge("?v1", "?v6", new GraphEdge("1007", "?v1", "?v6"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST F5 =====================\n\n";
}

void testC2() {
    std::cout << "==================== TEST C2 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v1", new GraphEdge("2000", "?v0", "?v1"));
    graph.addEdge("?v0", "?v2", new GraphEdge("2002", "?v0", "?v2"));
    graph.addEdge("?v2", "100", new GraphEdge("2004", "?v2", "100"));
    graph.addEdge("?v2", "?v3", new GraphEdge("2005", "?v2", "?v3"));
    graph.addEdge("?v4", "?v5", new GraphEdge("2011", "?v4", "?v5"));
    graph.addEdge("?v4", "?v6", new GraphEdge("2010", "?v4", "?v6"));
    graph.addEdge("?v4", "?v7", new GraphEdge("2009", "?v4", "?v7"));
    graph.addEdge("?v7", "?v3", new GraphEdge("2008", "?v7", "?v3"));
    graph.addEdge("?v3", "?v8", new GraphEdge("2006", "?v3", "?v8"));
    graph.addEdge("?v8", "?v9", new GraphEdge("2007", "?v8", "?v9"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST C2 =====================\n\n";
}

void testC3() {
    std::cout << "==================== TEST C3 ======================\n";
    LabeledGraph graph;
    graph.addEdge("?v0", "?v1", new GraphEdge("2000", "?v0", "?v1"));
    graph.addEdge("?v0", "?v2", new GraphEdge("2002", "?v0", "?v2"));
    graph.addEdge("?v0", "?v3", new GraphEdge("2004", "?v0", "?v3"));
    graph.addEdge("?v0", "?v4", new GraphEdge("2006", "?v0", "?v4"));
    graph.addEdge("?v0", "?v5", new GraphEdge("2009", "?v0", "?v5"));
    graph.addEdge("?v0", "?v6", new GraphEdge("2010", "?v0", "?v6"));
    graph.print();
    graph.generateQueryPlan();
    std::cout << "Graph after generate plan\n";
    graph.print();
    std::cout << "================== END TEST C3 =====================\n\n";
}

int main() {

    // testL2();
    // testS2();
    // testS5();
    // testF1();
    // testF3();
    testF4();
    // testF5();
    // testC2();
    // testC3();

    return 0;
}
