#ifndef JOINQUERYJOB_H
#define JOINQUERYJOB_H

#include <moderngpu/context.hxx>
#include "EmptyIntervalDict.h"
#include "QueryJob.h"
#include "IR.h"
#include "FullRelationIR.h"

class JoinQueryJob : public QueryJob
{
public:
    JoinQueryJob(QueryJob *leftJob, QueryJob *rightJob, std::string joinVariable, mgpu::standard_context_t* context,
                 std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound, EmptyIntervalDict *ei_dict, bool lastJoinForVar = true);
    ~JoinQueryJob() override;
    IR* getIR() override;
    int startJob(int gpuId) override;
    std::string jobTypeName() const override;

    void filterJoinedIndexVars(TYPEID *new_idx_vars, TYPEID *orig_idx_vars, int2* joined_idx, size_t joined_idx_size);
    // original size should be >= new size
    void calculateJoinIndexOffsetSize(TYPEID *original_offset, size_t offset_num, TYPEID *new_offset,
                                      int2* joined_idx, size_t joined_idx_size, size_t orig_relation_size, bool left);
    size_t filterJoinedRelation(TYPEID *orig_idx_offst_ptr, TYPEID *orig_idx_size_ptr, TYPEID_DEVICE_VEC* orig_relation,
                                size_t orig_relation_size, size_t new_relation_size,
                                int2* joined_idx, size_t joined_idx_size, bool left);

    mgpu::mem_t<int2> innerJoinMulti(TYPEID* a, int a_count, TYPEID* b, int b_count);

    void setOperands(QueryJob *leftJob, QueryJob *rightJob);
    IR* join(FullRelationIR *lir, FullRelationIR *rir);
    void print() const override;
    void setQueryVarCounter(std::map<std::string, size_t> *query_variable_counter);
    unsigned getLeftIRSize() const;
    unsigned getRightIRSize() const;
    std::string getJoinVariable() const;
    void mergeColumns(DTYPEID* irMergeRelation, FullRelationIR *ir, size_t col0, size_t col1);
    std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound{ nullptr };
    EmptyIntervalDict *ei_dict{ nullptr };
private:
    std::string joinVariable;
    mgpu::standard_context_t* context;
    QueryJob *leftJob { nullptr };
    QueryJob *rightJob { nullptr };
    bool lastJoinForVar;

    std::map<std::string, size_t> *query_variable_counter;
};

#endif // JOINQUERYJOB_H
