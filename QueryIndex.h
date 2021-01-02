#ifndef QUERYINDEX_H
#define QUERYINDEX_H

#include "vedas.h"

class QueryIndex
{
public:
    QueryIndex();
    void setL1Index(TYPEID_HOST_VEC *l1_values, TYPEID_HOST_VEC *l1_offsets);
    void setL2Index(TYPEID_HOST_VEC *l2_values, TYPEID_HOST_VEC *l2_offsets);
    TYPEID_HOST_VEC *getL1IndexValues() const;
    TYPEID_HOST_VEC *getL1IndexOffsets() const;
    TYPEID_HOST_VEC *getL2IndexValues() const;
    TYPEID_HOST_VEC *getL2IndexOffsets() const;
private:
    TYPEID_HOST_VEC *l1_values{ nullptr };
    TYPEID_HOST_VEC *l1_offsets{ nullptr };
    TYPEID_HOST_VEC *l2_values{ nullptr };
    TYPEID_HOST_VEC *l2_offsets{ nullptr };
};

#endif // QUERYINDEX_H
