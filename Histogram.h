#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <string>
#include <vector>
#include "vedas.h"

class Histogram
{
public:
    Histogram(std::string name, TYPEID_HOST_VEC idData);

    unsigned cardinalityEstimate(TYPEID start, TYPEID end);
private:
    std::string name;
    std::vector<TYPEID> lower_bounds, upper_bounds;
    std::vector<unsigned> frequencies;

    void initEqWidth(TYPEID_HOST_VEC idData, unsigned width);
    void initEqDepth(TYPEID_HOST_VEC idData, unsigned depth);
};

#endif // HISTOGRAM_H
