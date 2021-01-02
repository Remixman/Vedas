#ifndef PLANNODE_H
#define PLANNODE_H

#include <string>
#include "SelectQueryJob.h"

class PlanNode
{
public:
    PlanNode(std::string join_variable);
    PlanNode(SelectQueryJob *select_job);
    SelectQueryJob *getSelectJob();
    std::string getJoinVariable();
    bool isJoinPlanNode() const;
private:
    std::string join_variable;
    SelectQueryJob *select_job { nullptr };
};

#endif // PLANNODE_H
