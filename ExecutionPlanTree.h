#ifndef EXECUTIONPLANTREE_H
#define EXECUTIONPLANTREE_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "TriplePattern.h"

enum PlanOperation { UPLOAD, JOIN };

struct ExecPlanTreeNode {
    PlanOperation planOp;
    size_t order;
    ExecPlanTreeNode *parent;
    std::string index;                        // For UPLOAD only
    TriplePattern *tp;                        // For UPLOAD only
    std::vector<ExecPlanTreeNode *> children; // For JOIN only
    std::string joinVariable;                 // For JOIN only

    int resultSize = 0;
    double nanosecTime = 0.0;

    void print() {
        std::cout << "(OP " << std::setw(3) << order << ") : ";
        if (planOp == UPLOAD) {
            std::cout << "UPLOAD (" << tp->getSubject() << "," << tp->getPredicate() << ","
                      << tp->getObject() << ") with index \"" << index << "\"\n";
        } else {
            std::cout << "JOIN (" << children[0]->order << ") and ("
                      << children[1]->order << ") with " << joinVariable << "\n";
        }
    }
};

class ExecutionPlanTree {
public:
    ~ExecutionPlanTree();

    size_t addUploadOperation(std::string index, TriplePattern *tp);
    size_t addJoinOperation(size_t opId1, size_t opId2, std::string joinVar);

    void printTree() const;
    void printSequentialOrder() const;
    void writeGraphvizTreeFile(std::string fileName);
    std::vector<ExecPlanTreeNode *> getNodeList() const;
private:
    size_t lastOrder = 0;
    std::vector<ExecPlanTreeNode *> nodeList;

    std::string graphvizNodeDescription(ExecPlanTreeNode * node);
};

#endif
