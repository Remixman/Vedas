#include "Histogram.h"
#include <cmath>

Histogram::Histogram(std::string name, TYPEID_HOST_VEC idData) {
    this->name = name;
}

unsigned Histogram::cardinalityEstimate(TYPEID start, TYPEID end) {
    return 0;
}

void Histogram::initEqWidth(TYPEID_HOST_VEC idData, unsigned width) {

    if (idData.size() == 0) return;

    // XXX: assume that idData is already sorted
    TYPEID startId, endId;
    startId = idData[0];
    endId = startId + (width * ceil(idData.back() / 1.0 * width));

    // lower_bounds.clear();
    // upper_bounds.clear();
    // frequencies.clear();
    for (TYPEID id = startId; id < endId; id += width) {
        lower_bounds.push_back(id);
        upper_bounds.push_back(id + width);
        frequencies.push_back(0);
    }

    int i = 0;
    for (auto it = idData.begin(); it != idData.end(); ++it) {
        if (*it < upper_bounds[i]) frequencies[i]++;
        else i++;
    }
}

void Histogram::initEqDepth(TYPEID_HOST_VEC idData, unsigned depth) {

}
