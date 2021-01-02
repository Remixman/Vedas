#ifndef LINKEDDATA_H
#define LINKEDDATA_H

#include <vector>
#include <unordered_map>
#include "vedas.h"

class LinkedData
{
public:
    LinkedData(size_t n);
    void addLink(TYPEID i, TYPEID j, TYPEID pred);
    void reassignIdByBfs(std::unordered_map<TYPEID, TYPEID> &reassign_map);
private:
    size_t n;
    std::vector<std::vector<std::pair<TYPEID, TYPEID>>> links;
};

#endif // LINKEDDATA_H
