#ifndef TRANSFERJOB_H
#define TRANSFERJOB_H

#include "QueryJob.h"
#include "IR.h"
#include "FullRelationIR.h"

class TransferJob : public QueryJob
{
public:
    TransferJob(size_t src_thread_id, size_t dest_thread_id, int src_gpu_id, int dest_gpu_id, QueryJob *job);
    ~TransferJob() override;
    IR* getIR() override;
    int startJob(int gpuId) override;
    void print() const override;
    std::string jobTypeName() const override;
private:
    QueryJob *job;
    size_t src_thread_id, dest_thread_id;
    int src_gpu_id, dest_gpu_id;
    std::vector<int> gpu_ids;
};

#endif // TRANSFERJOB_H
