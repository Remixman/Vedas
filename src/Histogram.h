#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <string>
#include <vector>
#include "vedas.h"
#include "util/SegmentTree.h"

class Histogram
{
public:
    Histogram(const TYPEID_HOST_VEC& idData, unsigned width, unsigned depth);
    Histogram(const char *eqWidthFileName, const char *eqDepthFileName);
    ~Histogram();

    unsigned cardinalityEstimate(TYPEID start, TYPEID end, unsigned histType);
    TYPEID getCutId(TYPEID start, TYPEID end, unsigned histType);
    void loadEqWidth(const char *fname); // Read histogram from file and initialize
    void loadEqDepth(const char *fname); // Read histogram from file and initialize
    void writeEqWidth(const char *fname);
    void writeEqDepth(const char *fname);
    unsigned getWidth() const;
    unsigned getDepth() const;
private:
    unsigned width;
    unsigned depth;
    SegmentTree *st_eq_width = nullptr, *st_eq_depth = nullptr;
    std::vector<TYPEID> eqw_lowers, eqd_lowers;
    std::vector<TYPEID> eqw_uppers, eqd_uppers;
    std::vector<int> eqw_freq, eqd_freq;
    std::vector<int> eqw_acc_freq, eqd_acc_freq;

    void initEqWidth(const TYPEID_HOST_VEC& idData, unsigned width);
    void initEqDepth(const TYPEID_HOST_VEC& idData, unsigned depth);
    void write(const char *fname, int histType, unsigned widthOrDepth);
};

#endif // HISTOGRAM_H
