
#ifndef PLANTREENODE_H
#define PLANTREENODE_H

#include <string>
#include <iostream>
#include "TriplePattern.h"

enum PlanOperation { UPLOAD, JOIN, INDEXSWAP };

struct PlanTreeNode {
    PlanOperation op;
    PlanTreeNode *child1 { nullptr };
    PlanTreeNode *child2 { nullptr };
    std::string debugName = "";
    
    bool reuseVar = false;           // For JOIN only
    TriplePattern *tp;               // For UPLOAD only
    std::string var;                 // join var For JOIN, swap var for INDEXSWAP
};

#endif