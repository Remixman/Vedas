#ifndef SWAPINDEXJOB_H
#define SWAPINDEXJOB_H

#include "QueryJob.h"
#include "IR.h"
#include "IndexIR.h"
#include "FullRelationIR.h"

class SwapIndexJob : public QueryJob
{
public:
    SwapIndexJob();

    IR* swap(FullRelationIR *ir, size_t swap_column);
};

#endif // SWAPINDEXJOB_H
