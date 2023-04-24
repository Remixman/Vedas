#include "Histogram.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <cmath>
#include <thrust/binary_search.h>

Histogram::Histogram(const TYPEID_HOST_VEC& idData, unsigned width, unsigned depth) {
    this->width = width;
    this->depth = depth;

    this->initEqWidth(idData, width);
    this->initEqDepth(idData, depth);
}

Histogram::Histogram(const char *eqWidthFileName, const char *eqDepthFileName) {
    loadEqWidth(eqWidthFileName);
    loadEqDepth(eqDepthFileName);
}

Histogram::~Histogram() {
    if (st_eq_width != nullptr) delete st_eq_width;
    if (st_eq_depth != nullptr) delete st_eq_depth;
}

unsigned Histogram::cardinalityEstimate(TYPEID start, TYPEID end, unsigned histType) {
    std::vector<TYPEID> &lowers = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_lowers : eqd_lowers;
    std::vector<TYPEID> &uppers = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_uppers : eqd_uppers;
    std::vector<int> &freq = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_freq : eqd_freq;
    std::vector<int> &acc_freq = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_acc_freq : eqd_acc_freq;

    int total_freq = 0;
    auto lw1 = std::lower_bound(lowers.begin(), lowers.end(), start);
    auto lw2 = std::lower_bound(lowers.begin(), lowers.end(), end);

    auto idx1 = lw1 - lowers.begin();
    auto idx2 = lw2 - lowers.begin();

    total_freq += (uppers[idx1] - start) / (1.0 * (uppers[idx1] - lowers[idx1])) * freq[idx1];
    total_freq += acc_freq[idx2 - 1] - acc_freq[idx1];
    total_freq += (end - uppers[idx2]) / (1.0 * (uppers[idx2] - lowers[idx2])) * freq[idx2];

    return static_cast<unsigned>(total_freq);
}

TYPEID Histogram::getCutId(TYPEID start, TYPEID end, unsigned histType) {
    if (start == end) return start;

    std::vector<TYPEID> &lowers = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_lowers : eqd_lowers;
    // std::vector<TYPEID> &uppers = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_uppers : eqd_uppers;

    auto lw1 = std::lower_bound(lowers.begin(), lowers.end(), start);
    auto lw2 = std::lower_bound(lowers.begin(), lowers.end(), end);
    if (lw1 == lw2) return (end + start) / 2;

    TYPEID cutId;
    if (histType == HISTOGRAM_EQUAL_WIDTH) {
        // std::cout << "[HISTOGRAM_EQUAL_WIDTH] ";
        // lw += 1;
        
        /*int startii = std::distance(lw1, eqw_lowers.begin());
        int endii = std::distance(lw2, eqw_lowers.begin());
        for (auto ii = startii; ii <= endii; ++ii) {
            std::cout << "(" << eqw_lowers[ii] << "," << eqw_uppers[ii] << ") = " << eqw_freq[ii] << '\n';
        }*/
        
        std::cout << "RMQ(" << lw1 - eqw_lowers.begin() << ',' << lw2 - eqw_lowers.begin() << ")\n";
        auto minValIdx = st_eq_width->rmq(lw1 - eqw_lowers.begin(), lw2 - eqw_lowers.begin());
        auto minFreq = eqw_freq[minValIdx];
        std::cout << "Min Freq = " << minFreq << "(from range " << eqw_lowers[minValIdx] << '-' << eqw_uppers[minValIdx] <<  ")\n";
        cutId = (eqw_uppers[minValIdx] + eqw_lowers[minValIdx]) / 2;
    } else {
        // std::cout << "[HISTOGRAM_EQUAL_DEPTH] ";
        // lw += 1;
        auto minInvValIdx = st_eq_depth->rmq(lw1 - eqw_lowers.begin(), lw2 - eqw_lowers.begin());
        auto minFreq = eqd_freq[minInvValIdx];
        std::cout << "Min Freq = " << minFreq << '\n';
        cutId = (eqw_uppers[minInvValIdx] + eqw_lowers[minInvValIdx]) / 2;
    }
    std::cout << "Cut point from " << start << " to " << end << " is " << cutId << '\n';
    return cutId;
}

void Histogram::write(const char *fname, int histType, unsigned widthOrDepth) {
    std::vector<TYPEID> &lowers = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_lowers : eqd_lowers;
    std::vector<TYPEID> &uppers = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_uppers : eqd_uppers;
    std::vector<int> &freqs = (histType == HISTOGRAM_EQUAL_WIDTH) ? eqw_freq : eqd_freq;

    std::ofstream out;
    out.open(fname, std::ios::out);
    out << widthOrDepth << '\n';
    for (size_t i = 0; i < freqs.size(); ++i)
        out << lowers[i] << ' ' << uppers[i] << ' ' << freqs[i] << '\n';
    out.close();
}

void Histogram::writeEqWidth(const char *fname) {
    write(fname, HISTOGRAM_EQUAL_WIDTH, width);
}

void Histogram::writeEqDepth(const char *fname) {
    write(fname, HISTOGRAM_EQUAL_DEPTH, depth);
}

unsigned Histogram::getWidth() const {
    return width;
}

unsigned Histogram::getDepth() const {
    return depth;
}

void Histogram::loadEqWidth(const char *fname) {
    std::ifstream in(fname, std::ios::in);
    TYPEID lower, upper;
    int freq;

    in >> width;
    while (in >> lower >> upper >> freq) {
        eqw_lowers.push_back(lower);
        eqw_uppers.push_back(upper);
        eqw_freq.push_back(freq);
    }
    in.close();

    st_eq_width = new SegmentTree(eqw_freq);

    /*eqw_acc_freq.resize(eqw_freq.size());
    for (size_t i = 0; i < eqw_freq.size(); ++i) {
        if (i == 0) eqw_acc_freq[i] = eqw_freq[i];
        else eqw_acc_freq[i] = eqw_acc_freq[i-1] + eqw_freq[i];
    }

    // Create inverse frequency vector and segment tree
    int totalCount = eqw_acc_freq.back();
    std::vector<int> inv_freq;
    for (size_t i = 0; i < eqw_freq.size(); ++i) {
        inv_freq.push_back(totalCount - eqw_freq[i]);
    }

    st_eq_width = new SegmentTree(inv_freq);*/
}

void Histogram::loadEqDepth(const char *fname) {
    std::ifstream in(fname, std::ios::in);
    TYPEID lower, upper;
    int max_depth = 0;
    int freq;

    in >> depth;
    while (in >> lower >> upper >> freq) {
        eqd_lowers.push_back(lower);
        eqd_uppers.push_back(upper);
        eqd_freq.push_back(freq);
        if (upper - lower > max_depth) max_depth = upper - lower;
    }
    in.close();

    st_eq_depth = new SegmentTree(eqd_freq);

    // Store inverse of depth
    /*max_depth = 1000000;
    std::vector<int> inv_depth;
    for (size_t i = 0; i < eqd_freq.size(); ++i) inv_depth.push_back(max_depth - eqd_freq[i]);
    st_eq_depth = new SegmentTree(inv_depth);*/
}

void Histogram::initEqWidth(const TYPEID_HOST_VEC& idData, unsigned width) {

    if (idData.size() == 0) return;

    // XXX: assume that idData is already sorted
    TYPEID startId = idData[0];
    auto lastIrIt = thrust::lower_bound(thrust::host, idData.begin(), idData.end(), LITERAL_START_ID - 1);
    TYPEID endId = idData[thrust::distance(idData.begin(), lastIrIt) - 1];

    for (TYPEID id = startId; id < endId; id += width) {
        eqw_lowers.push_back(id);
        TYPEID lastId = id + width - 1;
        if (lastId > endId) lastId = endId;
        eqw_uppers.push_back(lastId);
    }

    int rangeIdx = 0;
    eqw_freq.resize(eqw_lowers.size(), 0);
    for (auto it = idData.begin(); it != idData.end(); ++it) {
        if (*it >= LITERAL_START_ID) break;
        while (*it > eqw_uppers[rangeIdx]) {
            rangeIdx++;
        }
        eqw_freq[rangeIdx] += 1;
    }

    // int acc = (eqw_acc_freq.size() == 0) ? 0 : eqw_acc_freq.back();
    // eqw_acc_freq.push_back(acc + eqw_freq.back());

    // st_eq_width = new SegmentTree(eqw_freq);
}

void Histogram::initEqDepth(const TYPEID_HOST_VEC& idData, unsigned depth) {
    unsigned startIdx = 0, dataIdx = 0;

    auto lastIrIt = thrust::lower_bound(thrust::host, idData.begin(), idData.end(), LITERAL_START_ID - 1);
    size_t woLiteralSize = thrust::distance(idData.begin(), lastIrIt);

    std::vector<int> inv_freq;
    while (dataIdx < woLiteralSize) {
        startIdx = dataIdx;
        TYPEID firstId = idData[startIdx];
        dataIdx += depth;
        if (dataIdx >= woLiteralSize) dataIdx = woLiteralSize - 1;
        TYPEID lastId = idData[dataIdx];
        while (dataIdx < woLiteralSize - 1 && idData[dataIdx] == idData[dataIdx + 1]) {
            dataIdx++;
        }
        unsigned freq = dataIdx - startIdx + 1;

        eqd_lowers.push_back(firstId);
        eqd_uppers.push_back(lastId);
        eqd_freq.push_back(freq);
        inv_freq.push_back(-freq);
        dataIdx++;

        //int acc = (eqd_acc_freq.size() == 0) ? 0 : eqd_acc_freq.back();
        //eqd_acc_freq.push_back(acc + eqd_freq.back());
    }

    // st_eq_depth = new SegmentTree(inv_freq);
}
