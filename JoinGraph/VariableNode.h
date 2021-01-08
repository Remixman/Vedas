#ifndef VARIABLENODE_H
#define VARIABLENODE_H

#include <string>
#include "GraphNode.h"

class VariableNode : public GraphNode {
public:
    VariableNode(std::string variable);
    std::string getVariable() const;
private:
    std::string variable;
};

#endif
