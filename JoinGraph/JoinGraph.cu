#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include "JoinGraph.h"
#include "VariableNode.h"
#include "IrNode.h"

JoinGraph::JoinGraph(SparqlQuery *sparqlQuery) {
    std::set<std::string> joinVariables;

    size_t patternSize = sparqlQuery->getPatternNum();
    for (size_t i = 0; i < patternSize; i++) {
        TriplePattern *tp = sparqlQuery->getPatternPtr(i);
        TriplePatternNode* tpn = new TriplePatternNode(tp);

        if (tp->subjectIsVariable()) addNewNode(tp->getSubject(), tpn);
        if (tp->predicateIsVariable()) addNewNode(tp->getPredicate(), tpn);
        if (tp->objectIsVariable()) addNewNode(tp->getObject(), tpn);
    }

    variables = sparqlQuery->getVariables();
    for (unsigned i = 0; i < variables.size(); i++) {
        unsigned long long ull = 1;
        varBitDict[variables[i]] = (ull << i);
    }
}

void JoinGraph::createLinkToVarNode(GraphNode *fromGn, const std::string &variable) {
    if (gNodeDict.count(fromGn) == 0) {
        adjList.push_back(std::list<GraphNode*>());
        removeMask.push_back(false);
        gNodeDict[fromGn] = adjList.size() - 1;
    }
    VariableNode *vn = new VariableNode(variable);
    adjList[gNodeDict[fromGn]].push_back(vn);
}

void JoinGraph::addNewNode(std::string variable, GraphNode *gn) {
    // Add triple pattern node to variable node
    if (varNodeDict.count(variable) == 0) {
        adjList.push_back(std::list<GraphNode*>());
        removeMask.push_back(false);
        varNodeDict[variable] = adjList.size() - 1;
        nodeVarDict[adjList.size() - 1] = variable;
    }
    adjList[varNodeDict[variable]].push_back(gn);

    createLinkToVarNode(gn, variable);
}

void JoinGraph::replaceNode(GraphNode *gn, GraphNode *replaceGn) {
    size_t idx = gNodeDict[gn];

    for (auto vit = adjList[idx].begin(); vit != adjList[idx].end(); ++vit) {
        auto graphNode = *vit;
        VariableNode *vn = dynamic_cast<VariableNode *>(graphNode);
        assert(vn != nullptr);

        size_t vidx = varNodeDict[vn->getVariable()];
        adjList[vidx].remove(gn);

        auto it = std::find(adjList[vidx].begin(), adjList[vidx].end(), replaceGn);
        if (it == adjList[vidx].end()) {
            adjList[vidx].push_back(replaceGn);
            createLinkToVarNode(replaceGn, vn->getVariable());
        }
    }
}

size_t JoinGraph::maxDegreeVarNodeIndex() const {
    size_t maxIdx = 0;
    for (auto vn: varNodeDict) {
        auto variable = vn.first;
        auto index = vn.second;
        if (adjList[index].size() > adjList[maxIdx].size()) {
            maxIdx = index;
        }
    }
    return maxIdx;
}

ExecutionPlanTree* JoinGraph::createPlan() {
    ExecutionPlanTree *tree = new ExecutionPlanTree();

    // TODO: remove variable node with 1 degree
    
    /*while (true) {
        size_t maxDegIdx = maxDegreeVarNodeIndex();
        std::string variable = nodeVarDict[maxDegIdx];
        int degree = static_cast<int>(adjList[maxDegIdx].size());

        if (degree <= 1) break;

        std::cout << "maxDegIdx : " << maxDegIdx << "\n";
        std::cout << "VAR : " << variable << " WITH DEGREE " << degree << "\n";

        GraphNode *n1, *n2;
        std::list<GraphNode*> &nodeList = adjList[maxDegIdx];

        assert(degree >= 2);

        // Random pick variable nodes
        auto firstIt = nodeList.begin();
        std::advance(firstIt, rand() % degree);
        n1 = *firstIt;
        nodeList.erase(firstIt);
        auto secondIt = nodeList.begin();
        std::advance(secondIt, rand() % (degree - 1));
        n2 = *secondIt;
        nodeList.erase(secondIt);

        TriplePatternNode* tpn1 = dynamic_cast<TriplePatternNode *>(n1);
        IrNode* irn1 = dynamic_cast<IrNode *>(n1);
        TriplePatternNode* tpn2 = dynamic_cast<TriplePatternNode *>(n2);
        IrNode* irn2 = dynamic_cast<IrNode *>(n2);
        size_t opNum1, opNum2;
        if (tpn1 != nullptr) {
            auto index = getUsedIndex(tpn1->getTriplePattern(), variable);
            opNum1 = tree->addUploadOperation(index, tpn1->getTriplePattern());
        } else if (irn1 != nullptr) {
            opNum1 = irn1->operatorId;
        }

        if (tpn2 != nullptr) {
            auto index = getUsedIndex(tpn2->getTriplePattern(), variable);
            opNum2 = tree->addUploadOperation(index, tpn2->getTriplePattern());
        } else if (irn2 != nullptr) {
            opNum2 = irn2->operatorId;
        }

        // std::cout << "JOIN " << opNum1 << ":" << opNum2 << "\n";
        size_t newOpNum = tree->addJoinOperation(opNum1, opNum2, variable);
        IrNode *newIrNode = new IrNode(newOpNum, opNum1, opNum2);
        // std::cout << "  GET " << newOpNum << "\n";

        replaceNode(n1, newIrNode);
        replaceNode(n2, newIrNode);
    }*/
    
    int lastIdx = -1;
    while (true) {
        size_t maxDegIdx = (lastIdx < 0)? maxDegreeVarNodeIndex() : lastIdx;
        std::string variable = nodeVarDict[maxDegIdx];
        int degree = static_cast<int>(adjList[maxDegIdx].size());

        if (adjList.size() <= 1) break;
        
        if (degree <= 1) {
            if (lastIdx < 0) {
                break;
            } else {
                lastIdx = -1;
                continue;
            }
        }

        lastIdx = maxDegIdx;

        GraphNode *n1, *n2;
        std::list<GraphNode*> &nodeList = adjList[maxDegIdx];

        assert(degree >= 2);

        // Random pick variable nodes
        auto firstIt = nodeList.begin();
        std::advance(firstIt, rand() % degree);
        n1 = *firstIt;
        nodeList.erase(firstIt);
        auto secondIt = nodeList.begin();
        std::advance(secondIt, rand() % (degree - 1));
        n2 = *secondIt;
        nodeList.erase(secondIt);

        TriplePatternNode* tpn1 = dynamic_cast<TriplePatternNode *>(n1);
        IrNode* irn1 = dynamic_cast<IrNode *>(n1);
        TriplePatternNode* tpn2 = dynamic_cast<TriplePatternNode *>(n2);
        IrNode* irn2 = dynamic_cast<IrNode *>(n2);
        size_t opNum1, opNum2;
        if (tpn1 != nullptr) {
            auto index = getUsedIndex(tpn1->getTriplePattern(), variable);
            opNum1 = tree->addUploadOperation(index, tpn1->getTriplePattern());
        } else if (irn1 != nullptr) {
            opNum1 = irn1->operatorId;
        }

        if (tpn2 != nullptr) {
            auto index = getUsedIndex(tpn2->getTriplePattern(), variable);
            opNum2 = tree->addUploadOperation(index, tpn2->getTriplePattern());
        } else if (irn2 != nullptr) {
            opNum2 = irn2->operatorId;
        }

        // std::cout << "JOIN " << opNum1 << ":" << opNum2 << "\n";
        size_t newOpNum = tree->addJoinOperation(opNum1, opNum2, variable);
        IrNode *newIrNode = new IrNode(newOpNum, opNum1, opNum2);
        // std::cout << "  GET " << newOpNum << "\n";

        replaceNode(n1, newIrNode);
        replaceNode(n2, newIrNode);
    }
    
    return tree;
}

ExecutionPlanTree* JoinGraph::createPlanDP() {

}

void JoinGraph::searchBinaryPlanDP() {
    std::vector<bool> varMask(variables.size(), false);
    /*for (unsigned i = 0; i < variables.size(); i++) {

    }*/
    searchBinaryPlanDPRecursive(varMask);
}

void JoinGraph::searchBinaryPlanDPRecursive(std::vector<bool> &varMask) {

    if (std::all_of(varMask.begin(), varMask.end(), [](bool b){ return b; })) {
        // Complete plan tree

        return;
    }

    for (unsigned i = 0; i < variables.size(); i++) {

    }
}

#define SUBJECT_BIT    4
#define PREDICATE_BIT  2
#define OBJECT_BIT     1
std::string JoinGraph::getUsedIndex(TriplePattern *tp, std::string &joinVar) {
    unsigned tpBit = 0;
    if (tp->subjectIsVariable()) tpBit += SUBJECT_BIT;
    if (tp->predicateIsVariable()) tpBit += PREDICATE_BIT;
    if (tp->objectIsVariable()) tpBit += OBJECT_BIT;

    switch (tpBit) {
        case (SUBJECT_BIT | PREDICATE_BIT | OBJECT_BIT):
            assert(false);
        case (SUBJECT_BIT):
            return "OPS"; // return "OSP";
        case (PREDICATE_BIT):
            return "OSP"; // return "SOP";
        case (OBJECT_BIT):
            return "PSO"; // return "SPO";
        case (SUBJECT_BIT | PREDICATE_BIT):
            return (joinVar == tp->getPredicate())? "OPS" : "OSP";
        case (SUBJECT_BIT | OBJECT_BIT):
            return (joinVar == tp->getObject())? "POS" : "PSO";
        case (PREDICATE_BIT | OBJECT_BIT):
            return (joinVar == tp->getObject())? "SOP" : "SPO";
    }

    return "";
}
