#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "EmptyIntervalDict.h"

bool EmptyIntervalDict::hasBoundFor(const std::string& var) const {
    return dict.count(var) > 0;
}

VAR_BOUND EmptyIntervalDict::getBound(const std::string& var) const {
    return dict.at(var);
}

size_t EmptyIntervalDict::emptySize(const std::string &var) {
    auto &intervalTuple = dict[var];

    size_t lb1 = std::get<0>(intervalTuple);
    size_t ub1 = std::get<1>(intervalTuple);
    size_t lb2 = std::get<2>(intervalTuple);
    size_t ub2 = std::get<3>(intervalTuple);
    
    return (ub1 - lb1) + (ub2 - lb2);
}

int rangeDistance2(std::tuple<size_t, size_t> &r1, std::tuple<size_t, size_t> &r2) {
    // Assume that r1 is before r2
    return VAR_LB(r2) - VAR_UB(r1);
}

void EmptyIntervalDict::updateBound(const std::string& var, size_t lb1, size_t ub1, size_t lb2, size_t ub2) {
    if (!hasBoundFor(var)) {
        dict[var] = std::make_tuple(lb1, ub1, lb2, ub2);
        return;
    }

#ifdef VERBOSE_DEBUG
    std::cout << "\tBefore update : " << DOUBLE_BOUND_STRING(dict[var]) << '\n';
#endif

    const VAR_BOUND& old_bound = dict.at(var);
    std::vector<std::pair<size_t, size_t>> intervalList;
    
    intervalList.push_back( std::make_pair(VAR_LB(old_bound), VAR_UB(old_bound)) );
    intervalList.push_back( std::make_pair(VAR_LB2(old_bound), VAR_UB2(old_bound)) );
    intervalList.push_back( std::make_pair(lb1, ub1) );
    intervalList.push_back( std::make_pair(lb2, ub2) );
    std::sort(intervalList.begin(), intervalList.end());

    for (size_t p = intervalList.size() - 1; p >= 1; --p) {
        auto& thisInterval = intervalList[p];
        auto& prevInterval = intervalList[p-1];

        if (thisInterval.first <= prevInterval.second) {
            // Overlap, merge interval
            prevInterval.second = thisInterval.second;
            intervalList.erase(intervalList.begin() + p);
        } else {
            // Not overlap, ignore
        }
    }

    switch (intervalList.size()) {
        case 0:
            dict[var] = std::make_tuple(0, 0, 0, 0);
            break;
        case 1:
            dict[var] = std::make_tuple(0, 0, intervalList[0].first, intervalList[0].second);
            break;
        case 2:
            dict[var] = std::make_tuple(intervalList[0].first, intervalList[0].second, intervalList[1].first, intervalList[1].second);
            break;
        default: // more than 2
        {
            size_t maxDist = 0, secondMaxDist = 0;
            size_t maxDistIdx = 0, secondMaxDistIdx = 0;
            for (size_t i = 0; i < intervalList.size(); ++i) {
                auto distance = intervalList[i].second - intervalList[i].first;
                if (distance > maxDist) {
                    secondMaxDist = maxDist;
                    secondMaxDistIdx = maxDistIdx;
                    maxDist = distance;
                    maxDistIdx = i;
                } else if (distance > secondMaxDist) {
                    secondMaxDist = distance;
                    secondMaxDistIdx = i;
                }
            }
            if (maxDistIdx > secondMaxDistIdx) {
                dict[var] = std::make_tuple(intervalList[secondMaxDistIdx].first, intervalList[secondMaxDistIdx].second, 
                                            intervalList[maxDistIdx].first, intervalList[maxDistIdx].second);
            } else {
                dict[var] = std::make_tuple(intervalList[maxDistIdx].first, intervalList[maxDistIdx].second,
                                            intervalList[secondMaxDistIdx].first, intervalList[secondMaxDistIdx].second);
            }
        }
    }

#ifdef VERBOSE_DEBUG
    std::cout << "\tAfter update : " << DOUBLE_BOUND_STRING(dict[var]) << '\n';
#endif
    
 
}

void EmptyIntervalDict::getUploadRelationData(const std::string &var,
                                            TYPEID_HOST_VEC::iterator start, TYPEID_HOST_VEC::iterator end, 
                                            TYPEID_HOST_VEC::iterator dataBegin, size_t &relationSize,
                                            std::vector<std::pair<size_t, size_t>> &uploadIntervals,
                                            std::vector<size_t> &offsets) {
    uploadIntervals.clear(); offsets.clear();
    auto &intervalTuple = dict[var];

    size_t lb1 = std::get<0>(intervalTuple);
    size_t ub1 = std::get<1>(intervalTuple);
    size_t lb2 = std::get<2>(intervalTuple);
    size_t ub2 = std::get<3>(intervalTuple);
    
    // std::cout << "EID for " << var << " (" << lb1 << "," << ub1 << ") and (" << lb2 << "," << ub2 << ")\n";

    assert(lb1 == 0 || ub1 == 0 || lb1 != ub1);
    assert(lb2 == 0 || ub2 == 0 || lb2 != ub2);

    if (ub2 == 0 && ub1 == 0) {
        size_t startOffst = thrust::distance(dataBegin, start);
        size_t endOffst = thrust::distance(dataBegin, end);

        uploadIntervals.push_back(std::make_pair(startOffst, endOffst));
        offsets.push_back(0);
        relationSize = thrust::distance(start, end);
    } else if (ub1 == 0) {
        auto offset1 = thrust::lower_bound(thrust::host, start, end, lb2);
        auto offset2 = thrust::upper_bound(thrust::host, start, end, ub2);

        size_t startOffst = thrust::distance(dataBegin, start);
        size_t offset1Offst = thrust::distance(dataBegin, offset1);
        size_t offset2Offst = thrust::distance(dataBegin, offset2);
        size_t endOffst = thrust::distance(dataBegin, end);

        uploadIntervals.push_back(std::make_pair(startOffst, offset1Offst));
        offsets.push_back(0);
        uploadIntervals.push_back(std::make_pair(offset2Offst, endOffst));
        offsets.push_back(thrust::distance(start, offset1));
        relationSize = thrust::distance(start, offset1) + thrust::distance(offset2, end);
    } else {
        auto offset1 = thrust::lower_bound(thrust::host, start, end, lb1);
        auto offset2 = thrust::upper_bound(thrust::host, start, end, ub1);
        auto offset3 = thrust::lower_bound(thrust::host, start, end, lb2);
        auto offset4 = thrust::upper_bound(thrust::host, start, end, ub2);

        size_t startOffst = thrust::distance(dataBegin, start);
        size_t offset1Offst = thrust::distance(dataBegin, offset1);
        size_t offset2Offst = thrust::distance(dataBegin, offset2);
        size_t offset3Offst = thrust::distance(dataBegin, offset3);
        size_t offset4Offst = thrust::distance(dataBegin, offset4);
        size_t endOffst = thrust::distance(dataBegin, end);

        uploadIntervals.push_back(std::make_pair(startOffst, offset1Offst));
        offsets.push_back(0);
        if (offset2 != offset3) {
            uploadIntervals.push_back(std::make_pair(offset2Offst, offset3Offst));
            offsets.push_back(thrust::distance(start, offset1));
        }
        uploadIntervals.push_back(std::make_pair(offset4Offst, endOffst));
        offsets.push_back(thrust::distance(start, offset1) + thrust::distance(offset2, offset3));
        relationSize = thrust::distance(start, offset1) + thrust::distance(offset2, offset3) + thrust::distance(offset4, end);
    }
}

void EmptyIntervalDict::print() const {
    for (auto element: dict) {
        std::cout << '[' << element.first << "] -> " << DOUBLE_BOUND_STRING(element.second) << '\n';
    }
}

std::tuple<size_t, size_t> EmptyIntervalDict::boundIntersect(size_t lba, size_t uba, size_t lbb, size_t ubb) {
    if (lbb > uba || lba > ubb) return std::make_tuple(0, 0);
    return std::make_tuple(std::max(lba, lbb), std::min(uba, ubb));
}
