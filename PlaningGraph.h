#ifndef PLANINGGRAPH_H
#define PLANINGGRAPH_H

#include <string>
#include <unordered_map>
#include "TriplePattern.h"

class PlaningGraphNode
{
public:
    PlaningGraphNode(TriplePattern *pattern);
    PlaningGraphNode(std::string &var, size_t var_count);
    std::string getName() const;
    std::string getJoinedVariableName() const;
    bool isDataNode() const;
    bool isJoinNode() const;
private:
    std::string name;
    std::string join_variable;
    size_t join_variable_count;

    TriplePattern *pattern;

    bool isData;
};

class PlaningGraph
{
public:
    PlaningGraph();
    void addJoinVariableNode(std::string var, size_t var_count);
    void addSelectNode(TriplePattern *pattern);
    bool canPartition();
private:
    std::unordered_map<std::string, size_t> node_map;
    std::unordered_map<std::string, size_t> join_var_map;
    std::vector<std::vector<PlaningGraphNode>> adj_list;
    std::vector<PlaningGraphNode> node_vector;
};

#endif // PLANINGGRAPH_H
