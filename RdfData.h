#ifndef RDFDATA_H
#define RDFDATA_H

#include <vector>
#include "vedas.h"

class RdfData
{
public:
    RdfData(std::vector<TYPEID> subjects, std::vector<TYPEID> predicates, std::vector<TYPEID> objects);
    size_t size() const;
    TYPEID_HOST_VEC& getSubject();
    TYPEID_HOST_VEC& getPredicate();
    TYPEID_HOST_VEC& getObject();
private:
    TYPEID_HOST_VEC subjects;
    TYPEID_HOST_VEC predicates;
    TYPEID_HOST_VEC objects;
};

#endif // RDFDATA_H
