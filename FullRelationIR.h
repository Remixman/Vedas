#ifndef FULLRELATIONIR_H
#define FULLRELATIONIR_H

#include <vector>
#include <unordered_map>
#include "vedas.h"
#include "IR.h"

class IndexIR; // To prevent circular definition

class FullRelationIR : public IR
{
public:
    FullRelationIR();
    FullRelationIR(size_t columnNum, size_t relationSize);
    ~FullRelationIR();
    void removeDuplicate();
    TYPEID* getRelationRawPointer(size_t i);
    TYPEID_DEVICE_VEC* getRelation(size_t i);
    void getRelationPointers(TYPEID** relations);
    size_t getColumnNum() const;
    size_t getColumnId(std::string var) const;
    size_t getRelationSize(size_t i) const;
    void setHeader(size_t i, std::string h, bool is_predicate);
    std::string getHeader(size_t i) const;
    bool getIsPredicate(size_t i) const;
    void setRelation(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void setRelation(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit);
    void sortByColumn(size_t i);
    void swapColumn(size_t i ,size_t j);
    IndexIR* toIndexIR(std::string idx_var);
    void print() const;
    void print(REVERSE_DICTTYPE *r_so_map, REVERSE_DICTTYPE *r_p_map) const;
    size_t size() const;
private:
    std::vector<TYPEID_DEVICE_VEC*> relation;
    std::vector<bool> is_predicates;
    std::vector<std::string> headers;
    std::unordered_map<std::string, size_t> var_column_map;
};

#endif // FULLRELATIONIR_H
