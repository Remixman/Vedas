#include "VariableNode.h"

VariableNode::VariableNode(std::string variable) {
    this->variable = variable;
}

std::string VariableNode::getVariable() const {
    return this->variable;
}
