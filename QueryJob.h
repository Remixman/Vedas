#ifndef QUERYJOB_H
#define QUERYJOB_H

#include "IR.h"
#include "ExecutionPlanTree.h"

class QueryJob
{
public:
    QueryJob();
    virtual ~QueryJob() = default;
    virtual IR* getIR() = 0;
    virtual void startJob() = 0;
    virtual void print() const = 0;
    ExecPlanTreeNode *planTreeNode = nullptr;
protected:
    IR* intermediateResult { nullptr };
};

#endif // QUERYJOB_H
