#include "PlanNode.h"

PlanNode::PlanNode(std::string join_variable) {
    this->join_variable = join_variable
}

PlanNode::PlanNode(SelectQueryJob *select_job) {
    this->select_job = select_job;
}

SelectQueryJob *PlanNode::getSelectJob() {
    return select_job;
}

std::string PlanNode::getJoinVariable() {
    return join_variable;
}

bool PlanNode::isJoinPlanNode() const {
    return this->join_variable != "";
}
