#ifndef INDEXSWAPJOB_H
#define INDEXSWAPJOB_H

#include <moderngpu/context.hxx>
#include "EmptyIntervalDict.h"
#include "QueryJob.h"
#include "IR.h"
#include "IndexIR.h"
#include "FullRelationIR.h"

class IndexSwapJob : public QueryJob
{
public:
    IndexSwapJob(QueryJob *beforeJob, std::string swapVar, mgpu::standard_context_t* context, EmptyIntervalDict *ei_dict);
    ~IndexSwapJob() override;
    IR* getIR() override;
    int startJob() override;
    void print() const override;
    std::string jobTypeName() const override;
private:
    mgpu::standard_context_t* context;
    std::string swapVar;
    QueryJob *beforeJob { nullptr };
    EmptyIntervalDict *ei_dict{ nullptr };
};

#endif // INDEXSWAPJOB_H
