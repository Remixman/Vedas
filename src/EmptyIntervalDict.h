#ifndef EMPTYINTERVALDICT_H
#define EMPTYINTERVALDICT_H

#include "vedas.h"

class EmptyIntervalDict
{
public:
    VAR_BOUND getBound(const std::string& var) const;
    void updateBound(const std::string& var, size_t lb1, size_t ub1, size_t lb2, size_t ub2);
    void getUploadRelationData(const std::string &var,
                            TYPEID_HOST_VEC::iterator start, TYPEID_HOST_VEC::iterator end, 
                            TYPEID_HOST_VEC::iterator dataBegin, size_t &relationSize,
                            std::vector<std::pair<size_t, size_t>> &uploadIntervals,
                            std::vector<size_t> &offsets);
    size_t emptySize(const std::string &var);
    void print() const;
private:
    bool hasBoundFor(const std::string& var) const;
    std::tuple<size_t, size_t> boundIntersect(size_t lba, size_t uba, size_t lbb, size_t ubb);
    std::unordered_map<std::string, VAR_BOUND> dict;
};

#endif // EMPTYINTERVALDICT_H
