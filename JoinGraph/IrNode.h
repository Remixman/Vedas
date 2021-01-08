#ifndef IRNODE_H
#define IRNODE_H

#include <vector>
#include "GraphNode.h"

class IrNode: public GraphNode {
public:
    IrNode(size_t operatorId, size_t childId1, size_t childId2);
    std::vector<size_t> children;
    size_t operatorId;
private:
};

#endif
