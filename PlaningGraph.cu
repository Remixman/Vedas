#include "PlaningGraph.h"

PlaningGraphNode::PlaningGraphNode(TriplePattern *pattern) {
    this->pattern = pattern;
    this->name = pattern->getName();
}

PlaningGraphNode::PlaningGraphNode(std::string &var, size_t var_count) {
    this->pattern = nullptr;
    this->join_variable = var;
    this->join_variable_count = var_count;
    this->name = "(var)" + var;
}

std::string PlaningGraphNode::getName() const {
    return name;
}

std::string PlaningGraphNode::getJoinedVariableName() const {
    return join_variable;
}

bool PlaningGraphNode::isDataNode() const {
    return isData;
}

bool PlaningGraphNode::isJoinNode() const {
    return !isDataNode();
}


PlaningGraph::PlaningGraph() {

}

void PlaningGraph::addJoinVariableNode(std::string var, size_t var_count) {
    PlaningGraphNode node(var, var_count);

    node_vector.push_back(node);

    adj_list.push_back(std::vector<PlaningGraphNode>());
    node_map[node.getName()] = adj_list.size() - 1;
    join_var_map[node.getJoinedVariableName()] = adj_list.size() - 1;
}

void PlaningGraph::addSelectNode(TriplePattern *pattern) {
    PlaningGraphNode node(pattern);

    node_vector.push_back(node);

    adj_list.push_back(std::vector<PlaningGraphNode>());
    size_t idx = adj_list.size() - 1;
    node_map[node.getName()] = idx;

    if (pattern->subjectIsVariable() && join_var_map.count(pattern->getSubject()) > 1) {
        size_t var_idx = join_var_map[pattern->getSubject()];
//        adj_list[var_idx].push_back(idx);
//        adj_list[idx].push_back(var_idx);
    }
    if (pattern->predicateIsVariable() && join_var_map.count(pattern->getPredicate()) > 1) {
        size_t var_idx = join_var_map[pattern->getPredicate()];
//        plan_adj_list[var_idx].push_back(idx);
//        plan_adj_list[idx].push_back(var_idx);
    }
    if (pattern->objectIsVariable() && join_var_map.count(pattern->getObject()) > 1) {
        size_t var_idx = join_var_map[pattern->getObject()];
//        plan_adj_list[var_idx].push_back(idx);
//        plan_adj_list[idx].push_back(var_idx);
    }
}
