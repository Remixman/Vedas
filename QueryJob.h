#ifndef QUERYJOB_H
#define QUERYJOB_H

#include "IR.h"

class QueryJob
{
public:
    QueryJob();
    virtual ~QueryJob() = default;
    virtual IR* getIR() = 0;
    virtual void startJob() = 0;
    virtual void print() const = 0;
protected:
    IR* intermediateResult { nullptr };
};

#endif // QUERYJOB_H