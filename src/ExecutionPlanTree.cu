#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include "ExecutionPlanTree.h"

ExecutionPlanTree::~ExecutionPlanTree() {
    for (auto node : this->nodeList) {
        delete node;
    }
}

size_t ExecutionPlanTree::addUploadOperation(std::string index, TriplePattern *tp) {
    ExecPlanTreeNode *tn = new ExecPlanTreeNode();
    tn->planOp = UPLOAD2;
    tn->order = ++lastOrder;
    tn->parent = nullptr;
    tn->index = index;
    tn->tp = tp;
    nodeList.push_back(tn);

    return tn->order;
}

size_t ExecutionPlanTree::addJoinOperation(size_t opId1, size_t opId2, std::string joinVar) {
    ExecPlanTreeNode *tn = new ExecPlanTreeNode();
    tn->planOp = JOIN2;
    tn->order = ++lastOrder;
    tn->parent = nullptr;
    tn->tp = nullptr;
    tn->joinVariable = joinVar;

    assert(nodeList[opId1 - 1]->parent == nullptr);
    nodeList[opId1 - 1]->parent = tn;
    tn->children.push_back(nodeList[opId1 - 1]);

    assert(nodeList[opId2 - 1]->parent == nullptr);
    nodeList[opId2 - 1]->parent = tn;
    tn->children.push_back(nodeList[opId2 - 1]);

    nodeList.push_back(tn);

    return tn->order;
}

void ExecutionPlanTree::printTree() const {

}

void ExecutionPlanTree::printSequentialOrder() const {
    for (auto node : nodeList) {
        node->print();
    }
}

std::string ExecutionPlanTree::graphvizNodeDescription(ExecPlanTreeNode * node) {
    std::stringstream ss;
    std::string name = "";
    if (node->planOp == UPLOAD2) {
        ss << "upload" << node->order << "[shape=none, label=< <b>tp<sub>1</sub>(" << node->order << ")</b><br/>" 
           << node->tp->toString() << "<br/>|tp<sub>" << node->tp->getId() << "</sub>|="
           << node->resultSize << "<br/><font point-size=\"10\">"
           << std::setprecision(2) << node->nanosecTime/1E6 << " ms.</font> >]";
    } else {
        ss << "join" << node->order << " [shape=none, label=< <b>join<sub>" << node->joinVariable 
           << "</sub>(" << node->order << ")</b><br/>Result=" << node->resultSize << "<br/><font point-size=\"10\">"
           << std::setprecision(2) << node->nanosecTime/1E6 << " ms.</font> >]";
    }

    return ss.str();
}

void ExecutionPlanTree::writeGraphvizTreeFile(std::string fileName) {
    std::ofstream gvfile(fileName, std::ios::out);

    // dot -Tpng <fileName>.gv -o plan.png

    gvfile << "digraph \"EXECUTION PLAN\" {\n";
    // Node list
    for (auto node : nodeList) {
        gvfile << graphvizNodeDescription(node) << "\n";
    }
    // Edge list
    for (auto node : nodeList) {
        if (node->parent != nullptr) {
            gvfile << ((node->parent->planOp == UPLOAD2)? "upload" : "join")
                   << node->parent->order << " -> "
                   << ((node->planOp == UPLOAD2)? "upload" : "join")
                   << node->order << " [dir=none]\n";
        }
    }
    gvfile << "}\n";

    gvfile.close();
}

std::vector<ExecPlanTreeNode *> ExecutionPlanTree::getNodeList() const {
    return nodeList;
}
