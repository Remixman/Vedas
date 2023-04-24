#ifndef TRANSFERJOB_H
#define TRANSFERJOB_H

#include "QueryJob.h"
#include "IR.h"
#include "FullRelationIR.h"

class TransferJob : public QueryJob
{
public:
    TransferJob(size_t src_thread_id, size_t dest_thread_id, QueryJob *job);
    ~TransferJob() override;
    IR* getIR() override;
    int startJob() override;
    void print() const override;
    std::string jobTypeName() const override;
    void setGpuIds(const std::vector<int>& gpu_ids);
private:
    QueryJob *job;
    size_t src_thread_id, dest_thread_id;
    std::vector<int> gpu_ids;
};

/* This job will send message to destination thread */
class MessageJob : public QueryJob
{
public:
    MessageJob(int dest_thread_id): dest_thread_id(dest_thread_id) {};
    int getDestinationThreadId() const { return dest_thread_id; };
    // FIXME: remove unnecessary overrides
    IR* getIR() { return nullptr; };
    int startJob() { return 0; };
    void print() const {};
    std::string jobTypeName() const { return "Message Job"; };
private:
    int dest_thread_id;
};

class WaitMessageJob : public QueryJob
{
public:
    WaitMessageJob(int src_thread_id): src_thread_id(src_thread_id) {};
    int getSourceThreadId() const { return src_thread_id; };
    // FIXME: remove unnecessary overrides
    IR* getIR() { return nullptr; };
    int startJob() { return 0; };
    void print() const {};
    std::string jobTypeName() const { return "Wait Message Job"; };
private:
    int src_thread_id;
};

#endif // TRANSFERJOB_H
