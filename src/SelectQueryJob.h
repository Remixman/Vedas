#ifndef SELECTQUERYJOB_H
#define SELECTQUERYJOB_H

#include <string>
#include <map>
#include "vedas.h"
#include "IR.h"
#include "QueryJob.h"
#include "EmptyIntervalDict.h"

class SelectQueryJob : public QueryJob
{
public:
    SelectQueryJob(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                   TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                   TYPEID_HOST_VEC *l2_data,
                   std::string v1, std::string v2, TYPEID id1, TYPEID_HOST_VEC *data, TYPEID_DEVICE_VEC *device_data,
                   std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound, EmptyIntervalDict *ei_dict,
                   bool *is_predicates, bool is_second_var_used, mgpu::standard_context_t* context);
    SelectQueryJob(TYPEID_HOST_VEC *l1_index_values, TYPEID_HOST_VEC *l1_index_offsets,
                   TYPEID_HOST_VEC *l2_index_values, TYPEID_HOST_VEC *l2_index_offsets,
                   std::string v1, TYPEID id1, TYPEID id2, TYPEID_HOST_VEC *data, TYPEID_DEVICE_VEC *device_data,
                   std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound, EmptyIntervalDict *ei_dict,
                   bool *is_predicates, mgpu::standard_context_t* context);
    ~SelectQueryJob() override;
    int startJob() override;
    IR* getIR() override;
    TYPEID_HOST_VEC *getL1IndexValues() const;
    TYPEID_HOST_VEC *getL1IndexOffsets() const;
    TYPEID_HOST_VEC *getL2IndexValues() const;
    TYPEID_HOST_VEC *getL2IndexOffsets() const;
    std::string getVariable(size_t i) const;
    size_t getVariableNum() const;
    TYPEID getId(size_t i) const;
    void print() const override;
    std::string jobTypeName() const override;
private:
    std::string variables[3];
    size_t variable_num;
    TYPEID dataid[2];
    bool is_predicates[2];
    bool is_second_var_used;
    TYPEID_HOST_VEC *l1_index_values{ nullptr };
    TYPEID_HOST_VEC *l1_index_offsets{ nullptr };
    TYPEID_HOST_VEC *l2_index_values{ nullptr };
    TYPEID_HOST_VEC *l2_index_offsets{ nullptr };
    TYPEID_HOST_VEC *l2_data{ nullptr };
    TYPEID_HOST_VEC *data{ nullptr };
    TYPEID_DEVICE_VEC *device_data{ nullptr };
    std::map< std::string, std::pair<TYPEID, TYPEID> > *variables_bound{ nullptr };
    EmptyIntervalDict *ei_dict{ nullptr };

    mgpu::standard_context_t* context;

    size_t findLowerBoundDataOffset(TYPEID val, TYPEID_HOST_VEC::iterator start, TYPEID_HOST_VEC::iterator end, 
        TYPEID_HOST_VEC::iterator raw_start);
    size_t findUpperBoundDataOffset(TYPEID val, TYPEID_HOST_VEC::iterator start, TYPEID_HOST_VEC::iterator end,
        TYPEID_HOST_VEC::iterator raw_start);
};

#endif // SELECTQUERYJOB_H
