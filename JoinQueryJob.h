#ifndef JOINQUERYJOB_H
#define JOINQUERYJOB_H

#include <moderngpu/context.hxx>
#include "QueryJob.h"
#include "IR.h"
#include "IndexIR.h"
#include "FullRelationIR.h"

class JoinQueryJob : public QueryJob
{
public:
    JoinQueryJob(QueryJob *leftJob, QueryJob *rightJob, std::string joinVariable, mgpu::standard_context_t* context,
                 std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound);
    ~JoinQueryJob() override;
    IR* getIR() override;
    void startJob() override;

    void filterJoinedIndexVars(TYPEID *new_idx_vars, TYPEID *orig_idx_vars, int2* joined_idx, size_t joined_idx_size);
    // original size should be >= new size
    void calculateJoinIndexOffsetSize(TYPEID *original_offset, size_t offset_num, TYPEID *new_offset,
                                      int2* joined_idx, size_t joined_idx_size, size_t orig_relation_size, bool left);
    size_t filterJoinedRelation(TYPEID *orig_idx_offst_ptr, TYPEID *orig_idx_size_ptr, TYPEID_DEVICE_VEC* orig_relation,
                                size_t orig_relation_size, size_t new_relation_size,
                                int2* joined_idx, size_t joined_idx_size, bool left);

    mgpu::mem_t<int2> innerJoinMulti(TYPEID* a, int a_count, TYPEID* b, int b_count);

    IR* join(FullRelationIR *lir, FullRelationIR *rir);
    IR* join(FullRelationIR *lir, IndexIR *rir);
    IR* join(IndexIR *lir, IndexIR *rir);
    IR* multiJoinIndexedIr(std::vector<IndexIR*> &irs);
    void print() const override;
    void setQueryVarCounter(std::map<std::string, size_t> *query_variable_counter);
    unsigned getLeftIRSize() const;
    unsigned getRightIRSize() const;
    std::string getJoinVariable() const;
private:
    std::string joinVariable;
    mgpu::standard_context_t* context;
    QueryJob *leftJob { nullptr };
    QueryJob *rightJob { nullptr };

    std::map<std::string, size_t> *query_variable_counter;
    std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound{ nullptr };
};

#endif // JOINQUERYJOB_H
