#ifndef INDEXIR_H
#define INDEXIR_H

#include <vector>
#include <string>
#include <unordered_map>
#include "vedas.h"
#include "IR.h"
#include "FullRelationIR.h"

class IndexIR : public IR
{
public:
    IndexIR(size_t columnNum, size_t indexSize, mgpu::standard_context_t* context);
    IndexIR(size_t columnNum, size_t indexSize, std::vector<size_t> relationSizes, mgpu::standard_context_t* context);
    void removeDuplicate();
    void resizeRelation(size_t i, size_t new_size);
    void resizeRelations(std::vector<size_t> relationSizes);
    void setIndexVariable(std::string idx_name, bool is_predicate);
    void setIndex(std::string &idx_name, bool is_predicate, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void setIndexOffset(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void setIndexOffset(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit);
    void setIndexOffsetAndNormalize(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    size_t getColumnNum() const;
    size_t getColumnId(std::string var) const;
    size_t getRelationSize(size_t i) const;
    void getRelationPointers(TYPEID** relations);
    void setRelationData(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void setRelationData(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit);
    void setHeader(size_t i, std::string h, bool is_predicate);
    std::string getHeader(size_t i) const;
    bool getIsPredicate(size_t i) const;
    std::string getIndexVariable() const;
    bool getIndexIsPredicate() const;
    size_t getIndexSize() const;
    TYPEID* getIndexVarsRawPointer();
    TYPEID* getIndexOffsetsRawPtr(size_t i);
    TYPEID_DEVICE_VEC* getIndexVars();
    TYPEID_DEVICE_VEC* getIndexOffset(size_t i);
    TYPEID_DEVICE_VEC* getRelationData(size_t i);
    FullRelationIR* toFullRelationIR();
    void adjustIndexOffset(TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void print() const;

    void calculateOffsetSizeArray(TYPEID_DEVICE_VEC *offsets, TYPEID_DEVICE_VEC *sizes, size_t relation_size);
    //void setRelation(TYPEID *relation, size_t relation_num, TYPEID *offset, TYPEID *vars, size_t sizes_num);
    void setRelationRecursive(FullRelationIR *fir, size_t i, size_t pre_k, size_t k, std::vector<TYPEID_HOST_VEC> &acc_size_matrix,
                              std::vector<TYPEID_HOST_VEC> &index_offst_matrix,
                              size_t relation_offset, size_t curr_relation_num, size_t prev_relation_num);
    void setRelation(FullRelationIR *fir, size_t i, std::vector<TYPEID_HOST_VEC> &acc_size_matrix,
                     std::vector<TYPEID_HOST_VEC> &index_offst_matrix, size_t relation_offst,
                     size_t curr_relation_num, TYPEID *offset, TYPEID *vars, size_t var_size);
private:
    std::string index_variable;
    TYPEID_DEVICE_VEC indexes_vars;
    bool index_is_predicate;

    std::vector<std::string> variables;   // Ex: ?y
    std::vector<bool> is_predicates;
    std::vector<TYPEID_DEVICE_VEC> indexes_offsets; // XXX: index to data (maybe multiple row)
    std::vector<TYPEID_DEVICE_VEC> datas;  // XXX: relation data (maybe multiple row)

    std::unordered_map<std::string, size_t> var_column_map;

    mgpu::standard_context_t* context;
};

#endif // INDEXIR_H
