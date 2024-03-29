#ifndef FULLRELATIONIR_H
#define FULLRELATIONIR_H

#include <vector>
#include <unordered_map>
#include <fstream>
#include "vedas.h"
#include "IR.h"

class FullRelationIR : public IR
{
public:
    FullRelationIR();
    FullRelationIR(size_t columnNum, size_t relationSize);
    ~FullRelationIR();
    void sort();
    void removeDuplicate();
    TYPEID* getRelationRawPointer(size_t i);
    TYPEID_DEVICE_VEC* getRelation(size_t i);
    void getRelationPointers(TYPEID** relations);
    size_t getColumnNum() const;
    size_t getColumnId(std::string var) const;
    size_t getRelationSize(size_t i) const;
    void setHeader(size_t i, std::string h, bool is_predicate);
    std::string getHeader(size_t i) const;
    std::string getHeaders(std::string delimitor) const;
    bool getIsPredicate(size_t i) const;
    void setRelation(size_t i, size_t offset, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void setRelation(size_t i, TYPEID_HOST_VEC::iterator bit, TYPEID_HOST_VEC::iterator eit);
    void setRelation(size_t i, size_t offset, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit);
    void setRelation(size_t i, TYPEID_DEVICE_VEC::iterator bit, TYPEID_DEVICE_VEC::iterator eit);
    void sortByFirstColumn();
    void swapColumn(size_t i ,size_t j);
    void removeColumn(size_t i);
    void removeColumn(size_t i, std::string &maintain_var);
    void movePeer(size_t src_device_id, size_t dest_device_id);
    void print() const;
    void print(bool full_version, std::ostream& out) const;
    void print(REVERSE_DICTTYPE *r_so_map, REVERSE_DICTTYPE *r_p_map, REVERSE_DICTTYPE *r_l_map) const;
    size_t size() const;
private:
    std::vector<TYPEID_DEVICE_VEC*> relation;
    std::vector<bool> is_predicates;
    std::vector<std::string> headers;
    std::vector<char> schemas;
    std::unordered_map<std::string, size_t> var_column_map;
};

#endif // FULLRELATIONIR_H
