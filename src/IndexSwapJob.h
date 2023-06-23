#ifndef INDEXSWAPJOB_H
#define INDEXSWAPJOB_H

#include <moderngpu/context.hxx>
#include "EmptyIntervalDict.h"
#include "QueryJob.h"
#include "IR.h"
#include "FullRelationIR.h"

class IndexSwapJob : public QueryJob
{
public:
    IndexSwapJob(QueryJob *beforeJob, std::string swapVar, mgpu::standard_context_t* context, 
                std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound, EmptyIntervalDict *ei_dict);
    ~IndexSwapJob() override;
    IR* getIR() override;
    int startJob(int gpuId) override;
    void print() const override;
    std::string jobTypeName() const override;
private:
    mgpu::standard_context_t* context;
    std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound{ nullptr };
    std::string swapVar;
    QueryJob *beforeJob { nullptr };
    EmptyIntervalDict *ei_dict{ nullptr };
};

#endif // INDEXSWAPJOB_H
