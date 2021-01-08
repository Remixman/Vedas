#include "IrNode.h"

IrNode::IrNode(size_t operatorId, size_t childId1, size_t childId2) {
    this->operatorId = operatorId;
    this->children.push_back(childId1);
    this->children.push_back(childId2);
}
