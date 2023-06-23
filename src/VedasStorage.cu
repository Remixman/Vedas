#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/scan.h>     // inclusize_scan, exclusive_scan
#include <thrust/tuple.h>
#include <thrust/reduce.h>   // reduce, reduce_by_key
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include "VedasStorage.h"
#include "Histogram.h"

VedasStorage::VedasStorage(RdfData &rdfData, bool preload = false, bool fullIndex = false)
{
    this->preload = preload;
    this->fullIndex = fullIndex;
    size_t rdfSize = rdfData.size();

    TYPEID_DEVICE_VEC d_subjs = rdfData.getSubject();
    TYPEID_DEVICE_VEC d_preds = rdfData.getPredicate();
    TYPEID_DEVICE_VEC d_objs  = rdfData.getObject();

    this->ps_data.resize(rdfSize);
    this->po_data.resize(rdfSize);
    this->pso_data.resize(rdfSize);
    this->pos_data.resize(rdfSize);

    // PSO & POS Index
    this->createIndexPS(d_subjs, d_preds, d_objs, rdfSize, p_idx_values, p_idx_offsets);
    thrust::copy(d_subjs.begin(), d_subjs.end(), ps_data.begin());
    thrust::copy(d_objs.begin(), d_objs.end(), pso_data.begin());
    this->createIndexPSO(d_subjs, d_preds, d_objs, rdfSize, p_idx_values, p_idx_offsets, ps_s_idx_values, ps_idx_offsets);
    this->createIndexPO(d_subjs, d_preds, d_objs, rdfSize, p_idx_values, p_idx_offsets);
    thrust::copy(d_objs.begin(), d_objs.end(), po_data.begin());
    thrust::copy(d_subjs.begin(), d_subjs.end(), pos_data.begin());
    this->createIndexPOS(d_subjs, d_preds, d_objs, rdfSize, p_idx_values, p_idx_offsets, po_o_idx_values, po_idx_offsets);


    if (fullIndex) {
        this->os_data.resize(rdfSize);
        this->spo_data.resize(rdfSize);
        this->sop_data.resize(rdfSize);
        this->ops_data.resize(rdfSize);
        this->osp_data.resize(rdfSize);

        // SPO & SOP Index
        this->createIndexSPO(d_subjs, d_preds, d_objs, rdfSize, s_idx_values, s_idx_offsets, sp_p_idx_values, sp_idx_offsets);
        thrust::copy(d_objs.begin(), d_objs.end(), spo_data.begin());
        this->createIndexSOP(d_subjs, d_preds, d_objs, rdfSize, s_idx_values, s_idx_offsets, so_o_idx_values, so_idx_offsets);
        thrust::copy(d_preds.begin(), d_preds.end(), sop_data.begin());

        // OSP & OPS Index
        this->createIndexOS(d_subjs, d_preds, d_objs, rdfSize, o_idx_values, o_idx_offsets);
        thrust::copy(d_subjs.begin(), d_subjs.end(), os_data.begin());
        thrust::copy(d_preds.begin(), d_preds.end(), osp_data.begin());
        this->createIndexOPS(d_subjs, d_preds, d_objs, rdfSize, o_idx_values, o_idx_offsets, op_p_idx_values, op_idx_offsets);
        thrust::copy(d_subjs.begin(), d_subjs.end(), ops_data.begin());
    }

    uploadData(); // Upload index and data to GPU
}

// note: functor inherits from unary_function
struct diffSquare : public thrust::unary_function<double, double>
{
  __host__ __device__
  double operator()(double x) const
  {
    return x * x;
  }
};

double sd(thrust::host_vector<double> &data, double avg) {
    double sum = 0.0;
    for (int i = 0; i < data.size(); i++) {
        double x = data[i] - avg;
        sum += (x * x);
    }
    return sum / data.size();
}

VedasStorage::VedasStorage(const char *fname, bool preload = false, bool fullIndex = false) {
    this->preload = preload;
    this->fullIndex = fullIndex;
    this->open(fname);

    uploadData(); // Upload index and data to GPU
}

PredicateIndexStat VedasStorage::psStat() const {
    // std::cout << "[Predicate Indices]\n";
    PredicateIndexStat stat;

    size_t uniquePredicateNumber = p_idx_values.size();
    thrust::host_vector<double> psDists(uniquePredicateNumber);

    for (size_t i = 0; i < p_idx_values.size(); i++) {
        // std::cout << "Predicate ID : " << p_idx_values[i] << "\n";
        // Show PS Distribution
        auto offset = p_idx_offsets[i];
        auto next_offset = (p_idx_values.size() - 1 != i)? p_idx_offsets[i+1] : ps_data.size();
        auto minId = ps_data[offset];
        auto maxId = ps_data[next_offset-1];
        auto n = next_offset - offset;
        psDists[i] = (maxId - minId) / (1.0*n);
        // std::cout << "\t[PS] Range = " << maxId - minId << " , N = " << n << " , Distribution = " << psDists[i] << "\n";
    }

    auto psDistSum = thrust::reduce(psDists.begin(), psDists.end(), 0.0, thrust::plus<double>());
    stat.dist_mean = psDistSum / uniquePredicateNumber;
    stat.dist_sd = sd(psDists, stat.dist_mean);

    return stat;
}

PredicateIndexStat VedasStorage::poStat() const {
    PredicateIndexStat stat;

    size_t uniquePredicateNumber = p_idx_values.size();
    thrust::host_vector<double> poDists(uniquePredicateNumber);

    for (size_t i = 0; i < p_idx_values.size(); i++) {
        auto offset = p_idx_offsets[i];
        auto next_offset = (p_idx_values.size() - 1 != i)? p_idx_offsets[i+1] : po_data.size();
        auto minId = po_data[offset];
        auto maxId = po_data[next_offset-1];
        auto n = next_offset - offset;
        poDists[i] = (maxId - minId) / (1.0*n);
    }

    auto poDistSum = thrust::reduce(poDists.begin(), poDists.end(), 0.0, thrust::plus<double>());
    stat.dist_mean = poDistSum / uniquePredicateNumber;
    stat.dist_sd = sd(poDists, stat.dist_mean);

    return stat;
}

double VedasStorage::psBoundaryCompactness() const {
    size_t predNum = p_idx_values.size();

    double nSum = 0;
    for (int i = 0; i < predNum; ++i) {
        for (int j = 1; j < predNum; ++j) {
            if (i == j) continue;

            auto i_offset = p_idx_offsets[i];
            auto next_i_offset = (p_idx_values.size() - 1 != i)? p_idx_offsets[i+1] : ps_data.size();
            auto minIdI = ps_data[i_offset];
            auto maxIdI = ps_data[next_i_offset-1];

            auto j_offset = p_idx_offsets[j];
            auto next_j_offset = (p_idx_values.size() - 1 != j)? p_idx_offsets[j+1] : ps_data.size();
            auto minIdJ = ps_data[j_offset];
            auto maxIdJ = ps_data[next_j_offset-1];

            auto maxIdOfMin = std::max(minIdI, minIdJ);
            auto minIdOfMax = std::min(maxIdI, maxIdJ);

            auto bitI = thrust::lower_bound(thrust::host, ps_data.begin() + i_offset, ps_data.begin() + next_i_offset, maxIdOfMin);
            auto eitI = thrust::upper_bound(thrust::host, ps_data.begin() + i_offset, ps_data.begin() + next_i_offset, minIdOfMax);
            size_t iRow = thrust::distance(eitI, bitI);

            auto bitJ = thrust::lower_bound(thrust::host, ps_data.begin() + j_offset, ps_data.begin() + next_j_offset, maxIdOfMin);
            auto eitJ = thrust::upper_bound(thrust::host, ps_data.begin() + j_offset, ps_data.begin() + next_j_offset, minIdOfMax);
            size_t jRow = thrust::distance(eitJ, bitJ);

            nSum += (iRow + jRow) / 2.0;
        }
    }

    return nSum / predNum;
}

double VedasStorage::poBoundaryCompactness() const {
    size_t predNum = p_idx_values.size();

    double nSum = 0;
    for (int i = 0; i < predNum; ++i) {
        for (int j = 1; j < predNum; ++j) {
            if (i == j) continue;

            auto i_offset = p_idx_offsets[i];
            auto next_i_offset = (p_idx_values.size() - 1 != i)? p_idx_offsets[i+1] : po_data.size();
            auto minIdI = po_data[i_offset];
            auto maxIdI = po_data[next_i_offset-1];

            auto j_offset = p_idx_offsets[j];
            auto next_j_offset = (p_idx_values.size() - 1 != j)? p_idx_offsets[j+1] : po_data.size();
            auto minIdJ = po_data[j_offset];
            auto maxIdJ = po_data[next_j_offset-1];

            auto maxIdOfMin = std::max(minIdI, minIdJ);
            auto minIdOfMax = std::min(maxIdI, maxIdJ);

            auto bitI = thrust::lower_bound(thrust::host, po_data.begin() + i_offset, po_data.begin() + next_i_offset, maxIdOfMin);
            auto eitI = thrust::upper_bound(thrust::host, po_data.begin() + i_offset, po_data.begin() + next_i_offset, minIdOfMax);
            size_t iRow = thrust::distance(eitI, bitI);

            auto bitJ = thrust::lower_bound(thrust::host, po_data.begin() + j_offset, po_data.begin() + next_j_offset, maxIdOfMin);
            auto eitJ = thrust::upper_bound(thrust::host, po_data.begin() + j_offset, po_data.begin() + next_j_offset, minIdOfMax);
            size_t jRow = thrust::distance(eitJ, bitJ);

            nSum += (iRow + jRow) / 2.0;
        }
    }

    return nSum / predNum;
}

size_t VedasStorage::getSubjectIndexSize() const { return this->s_idx_values.size(); }
size_t VedasStorage::getSubjectPredicateIndexSize() const { return this->sp_idx_offsets.size(); }
size_t VedasStorage::getSubjectObjectIndexSize() const { return this->so_idx_offsets.size(); }
size_t VedasStorage::getPredicateIndexSize() const { return this->p_idx_values.size(); }
size_t VedasStorage::getObjectIndexSize() const { return this->o_idx_values.size(); }

size_t VedasStorage::getObjectPredicateIndexSize() const { return this->op_idx_offsets.size(); }
size_t VedasStorage::getTripleSize() const { return this->pso_data.size(); }

TYPEID_HOST_VEC *VedasStorage::getSubjectIndexValues() { return &(this->s_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getSubjectIndexOffsets() { return &(this->s_idx_offsets); }
TYPEID_HOST_VEC *VedasStorage::getSubjectPredicateIndexValues() { return &(this->sp_p_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getSubjectPredicateIndexOffsets() { return &(this->sp_idx_offsets); }
TYPEID_HOST_VEC *VedasStorage::getSubjectObjectIndexValues() { return &(this->so_o_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getSubjectObjectIndexOffsets() { return &(this->so_idx_offsets); }

TYPEID_HOST_VEC *VedasStorage::getPredicateIndexValues() { return &(this->p_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getPredicateIndexOffsets() { return &(this->p_idx_offsets); }
TYPEID_HOST_VEC *VedasStorage::getPredicateSubjectIndexValues() { return &(this->ps_s_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getPredicateSubjectIndexOffsets() { return &(this->ps_idx_offsets); }
TYPEID_HOST_VEC *VedasStorage::getPredicateObjectIndexValues() { return &(this->po_o_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getPredicateObjectIndexOffsets() { return &(this->po_idx_offsets); }

TYPEID_HOST_VEC *VedasStorage::getObjectIndexValues() { return &(this->o_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getObjectIndexOffsets() { return &(this->o_idx_offsets); }


TYPEID_HOST_VEC *VedasStorage::getObjectPredicateIndexValues() { return &(this->op_p_idx_values); }
TYPEID_HOST_VEC *VedasStorage::getObjectPredicateIndexOffsets() { return &(this->op_idx_offsets); }

TYPEID_HOST_VEC *VedasStorage::getPSdata() { return &(ps_data); }
TYPEID_HOST_VEC *VedasStorage::getPOdata() { return &(po_data); }
TYPEID_HOST_VEC *VedasStorage::getOSdata() { return &(os_data); }

TYPEID_HOST_VEC *VedasStorage::getSPOdata() { return &(spo_data); }
TYPEID_HOST_VEC *VedasStorage::getSOPdata() { return &(sop_data); }
TYPEID_HOST_VEC *VedasStorage::getPSOdata() { return &(pso_data); }
TYPEID_HOST_VEC *VedasStorage::getPOSdata() { return &(pos_data); }
TYPEID_HOST_VEC *VedasStorage::getOPSdata() { return &(ops_data); }
TYPEID_HOST_VEC *VedasStorage::getOSPdata() { return &(osp_data); }

TYPEID_DEVICE_VEC *VedasStorage::getDeviceSPOdata() { return &(d_spo_data); }
TYPEID_DEVICE_VEC *VedasStorage::getDeviceSOPdata() { return &(d_sop_data); }
TYPEID_DEVICE_VEC *VedasStorage::getDevicePSOdata() { return &(d_pso_data); }
TYPEID_DEVICE_VEC *VedasStorage::getDevicePOSdata() { return &(d_pos_data); }
TYPEID_DEVICE_VEC *VedasStorage::getDeviceOPSdata() { return &(d_ops_data); }
TYPEID_DEVICE_VEC *VedasStorage::getDeviceOSPdata() { return &(d_osp_data); }

void VedasStorage::createIndex1Level(const char *index_name, TYPEID_DEVICE_VEC &v1, TYPEID_DEVICE_VEC &v2, TYPEID_DEVICE_VEC &v3,
                                     size_t n, TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets) {
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin(), v3.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(v1.end(), v2.end(), v3.end()));

    thrust::sort(zip_begin, zip_end);

    // Sort by v1 and create 1st level index. Skip step if it's already exist
    if (values.size() == 0) {
        TYPEID_DEVICE_VEC d_values(n), d_offsets(n);
        auto end_iterator_pair = thrust::reduce_by_key(v1.begin(), v1.end(), thrust::make_constant_iterator(1), d_values.begin(), d_offsets.begin());
        thrust::exclusive_scan(d_offsets.begin(), end_iterator_pair.second, d_offsets.begin());
        unsigned value_size = thrust::distance(d_offsets.begin(), end_iterator_pair.second);

        // Copy 1st level index value and offset to host
        values.resize(value_size); offsets.resize(value_size);
        thrust::copy(d_values.begin(), end_iterator_pair.first, values.begin());
        thrust::copy(d_offsets.begin(), end_iterator_pair.second, offsets.begin());
    }
}

void VedasStorage::createIndexPS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects,
                                 size_t n, TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets) {
    this->createIndex1Level("PS", predicate, subjects, objects, n, values, offsets);
}

void VedasStorage::createIndexPO(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects,
                                 size_t n, TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets) {
    this->createIndex1Level("PO", predicate, objects, subjects, n, values, offsets);
}

void VedasStorage::createIndexOS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects,
                                 size_t n, TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets) {
    this->createIndex1Level("OS", objects, subjects, predicate, n, values, offsets);
}

void VedasStorage::createIndex(const char *index_name,
                               TYPEID_DEVICE_VEC &v1, TYPEID_DEVICE_VEC &v2, TYPEID_DEVICE_VEC &v3 /* Input and Output */,
                               size_t n,
                               TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, /*TYPEID_HOST_VEC &value2_1,*/
                               TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin(), v3.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(v1.end(), v2.end(), v3.end()));

    thrust::sort(zip_begin, zip_end);

    if (index_name[0] == 'O') writeHistogram("object", v1);
    else if (index_name[0] == 'S') writeHistogram("subject", v1);

    if (values.size() == 0) {
        // Create 1st level index
        // TYPEID_DEVICE_VEC d_values(n);                // 1st level index values (device)
        // thrust::device_vector<unsigned> d_offsets(n); // 1st level index offsets (device)
        // auto end_iterator_pair = thrust::reduce_by_key(v1.begin(), v1.end(), thrust::make_constant_iterator(1), d_values.begin(), d_offsets.begin());
        // thrust::exclusive_scan(d_offsets.begin(), end_iterator_pair.second, d_offsets.begin());
        // unsigned value_size = thrust::distance(d_offsets.begin(), end_iterator_pair.second);

        // values.resize(value_size);                    // 1st level index values (host)
        // offsets.resize(value_size);                   // 1st level index offsets (host)
        // thrust::copy(d_values.begin(), end_iterator_pair.first, values.begin());
        // thrust::copy(d_offsets.begin(), end_iterator_pair.second, offsets.begin());

        // // Count second data per one first data
        // TYPEID_DEVICE_VEC d_2nd_col_per_1st_col(value_size);
        // thrust::transform(d_offsets.begin() + 1, d_offsets.end(), d_offsets.begin(), d_2nd_col_per_1st_col,
        //                   thrust::minus<TYPEID>());
        // d_2nd_col_per_1st_col[value_size - 1] = d_values2_1.size() - d_offsets.back();
        // TYPEID_HOST_VEC 2nd_col_per_1st_col(value_size);
        // thrust::copy(d_2nd_col_per_1st_col.begin(), d_2nd_col_per_1st_col.end(), 2nd_col_per_1st_col.begin());

        // // TODO: only test
        // std::string filename = index_name + ".txt";
        // std::ofstream diff; diff.open(filename, std::ios::out);
        // diff << 2nd_col_per_1st_col.size() << "\n";
        // for (auto d: 2nd_col_per_1st_col) diff << d << "\n";
        // diff.close();
    }

    // Create 2nd level index
    TYPEID_DEVICE_VEC d_values2_1(n);
    TYPEID_DEVICE_VEC d_values2_2(n);
    thrust::device_vector<unsigned> d_offsets2(n);  // 2nd level index offsets
    auto zip2_begin = thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin()));
    auto zip2_end = thrust::make_zip_iterator(thrust::make_tuple(v1.end(), v2.end()));
    auto zip_value_begin = thrust::make_zip_iterator(thrust::make_tuple(d_values2_1.begin(), d_values2_2.begin()));
    auto end_iterator_pair2 = thrust::reduce_by_key(zip2_begin, zip2_end , thrust::make_constant_iterator(1), zip_value_begin, d_offsets2.begin());
    thrust::exclusive_scan(d_offsets2.begin(), end_iterator_pair2.second, d_offsets2.begin());
    unsigned value2_size = thrust::distance(d_offsets2.begin(), end_iterator_pair2.second);

    value2_2.resize(value2_size);
    offsets2.resize(value2_size);
    thrust::copy(d_values2_2.begin(), d_values2_2.begin() + value2_size, value2_2.begin());
    thrust::copy(d_offsets2.begin(), end_iterator_pair2.second, offsets2.begin());

    // TODO: Create aggreagation indexed
}

void VedasStorage::createIndexSPO(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                  TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    this->createIndex("SPO", subjects, predicate, objects, n, values, offsets, value2_2, offsets2);
}

void VedasStorage::createIndexSOP(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                  TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    this->createIndex("SOP", subjects, objects, predicate, n, values, offsets, value2_2, offsets2);
}

void VedasStorage::createIndexPSO(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                  TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    this->createIndex("PSO", predicate, subjects, objects, n, values, offsets, value2_2, offsets2);
}

void VedasStorage::createIndexPOS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                  TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    this->createIndex("POS", predicate, objects, subjects, n, values, offsets, value2_2, offsets2);
}

void VedasStorage::createIndexOSP(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                  TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    this->createIndex("OSP", objects, subjects, predicate, n, values, offsets, value2_2, offsets2);
}

void VedasStorage::createIndexOPS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                  TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2) {
    this->createIndex("OPS", objects, predicate, subjects, n, values, offsets, value2_2, offsets2);
}

bool VedasStorage::isPreload() const { return this->preload; }

void VedasStorage::printIndex(char c1, char c2, char c3,
                              const TYPEID_HOST_VEC &l1IdxVals, const TYPEID_HOST_VEC &l1IdxOfssts,
                              const TYPEID_HOST_VEC &l1l2IdxVals, const TYPEID_HOST_VEC &l1l2IdxOfssts,
                              const TYPEID_HOST_VEC &data) const {
    std::cout << "\n";
    std::cout << "|============================= "<<c1<<c2<<c3<<" Index ==========================|\n";
    std::cout << "|======== "<<c1<<" Idx ========|======= "<<c1<<c2<<" Idx ========|===== "<<c3<<" Data =====|\n";
    std::cout << "|=== Val ===|== Offst ==|=== Val ===|== Offst ==|                  |\n";
    std::cout << "|===========|===========|===========|===========|==================|\n";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "|";
        if (i < l1IdxVals.size()) std::cout << std::setw(10) << l1IdxVals[i] << " |";
        else std::cout << "           |";
        if (i < l1IdxOfssts.size()) std::cout << std::setw(10) << l1IdxOfssts[i] << " |";
        else std::cout << "           |";
        if (i < l1l2IdxVals.size()) std::cout << std::setw(10) << l1l2IdxVals[i] << " |";
        else std::cout << "           |";
        if (i < l1l2IdxOfssts.size()) std::cout << std::setw(10) << l1l2IdxOfssts[i] << " |";
        else std::cout << "           |";
        std::cout << std::setw(17) << data[i] << " |\n";
    }
    std::cout << "|==================================================================|\n";
}

void VedasStorage::printSPOIndex() const {
    printIndex('S', 'P', 'O', s_idx_values, s_idx_offsets, sp_p_idx_values, sp_idx_offsets, spo_data);
}

void VedasStorage::printSOPIndex() const {
    printIndex('S', 'O', 'P', s_idx_values, s_idx_offsets, so_o_idx_values, so_idx_offsets, sop_data);
}

void VedasStorage::printOPSIndex() const {
    printIndex('O', 'P', 'S', o_idx_values, o_idx_offsets, op_p_idx_values, op_idx_offsets, ops_data);
}

void VedasStorage::uploadData() {
    if (!preload) return;

    size_t data_size = pso_data.size();
    d_pso_data.resize(data_size);
    d_pos_data.resize(data_size);
    
    thrust::copy(pso_data.begin(), pso_data.end(), d_pso_data.begin());
    thrust::copy(pos_data.begin(), pos_data.end(), d_pos_data.begin());


    if (fullIndex) {
        d_spo_data.resize(data_size);
        d_sop_data.resize(data_size);
        d_ops_data.resize(data_size);
        d_osp_data.resize(data_size);
    
        thrust::copy(spo_data.begin(), spo_data.end(), d_spo_data.begin());
        thrust::copy(sop_data.begin(), sop_data.end(), d_sop_data.begin());
        thrust::copy(ops_data.begin(), ops_data.end(), d_ops_data.begin());
        thrust::copy(osp_data.begin(), osp_data.end(), d_osp_data.begin());
    }
}

void readArray(std::ifstream &in, TYPEID_HOST_VEC &arr) {
    size_t arr_size;
    in.read((char*)&arr_size, sizeof(size_t));
    arr.resize(arr_size);
    in.read((char*)arr.data(), arr_size * sizeof(TYPEID));
}

void VedasStorage::open(const char *fname) {
    std::ifstream in;
    in.open(fname, std::ios::binary);

    readArray(in, p_idx_values); readArray(in, p_idx_offsets);
    readArray(in, ps_s_idx_values); readArray(in, ps_idx_offsets);
    readArray(in, po_o_idx_values); readArray(in, po_idx_offsets);
    readArray(in, ps_data); readArray(in, po_data);
    readArray(in, pso_data); readArray(in, pos_data);
    

    if (fullIndex) {
        readArray(in, s_idx_values); readArray(in, s_idx_offsets);
        readArray(in, sp_p_idx_values); readArray(in, sp_idx_offsets);
        readArray(in, so_o_idx_values); readArray(in, so_idx_offsets);

        readArray(in, o_idx_values); readArray(in, o_idx_offsets);
        readArray(in, os_data);
        readArray(in, op_p_idx_values); readArray(in, op_idx_offsets);

        readArray(in, spo_data); readArray(in, sop_data);
        readArray(in, ops_data); readArray(in, osp_data);
    }

    in.close();
}

void writeArray(std::ofstream &out, const TYPEID_HOST_VEC &arr) {
    size_t array_size = arr.size();
    out.write((char*)&array_size, sizeof(size_t));
    out.write((char*)arr.data(), array_size * sizeof(TYPEID));
}

void VedasStorage::write(const char *fname) {
    std::ofstream out;
    out.open(fname, std::ios::binary);

    writeArray(out, p_idx_values); writeArray(out, p_idx_offsets);
    writeArray(out, ps_s_idx_values); writeArray(out, ps_idx_offsets);
    writeArray(out, po_o_idx_values); writeArray(out, po_idx_offsets);
    writeArray(out, ps_data); writeArray(out, po_data);
    writeArray(out, pso_data); writeArray(out, pos_data);
    
    if (fullIndex) {
        writeArray(out, s_idx_values); writeArray(out, s_idx_offsets);
        writeArray(out, sp_p_idx_values); writeArray(out, sp_idx_offsets);
        writeArray(out, so_o_idx_values); writeArray(out, so_idx_offsets);
        
        writeArray(out, o_idx_values); writeArray(out, o_idx_offsets);
        writeArray(out, os_data);
        writeArray(out, op_p_idx_values); writeArray(out, op_idx_offsets);

        writeArray(out, spo_data); writeArray(out, sop_data);
        writeArray(out, ops_data); writeArray(out, osp_data);
    }

    out.close();
}

void VedasStorage::writeHistogram(const char *termType, TYPEID_DEVICE_VEC v) {
    TYPEID_HOST_VEC hv = v;
    // for (int w = 1E4; w <= 1E6; w *= 10) {
    for (int w = 1E2; w <= 1E4; w *= 10) {
        Histogram hist(hv, w, w);
        std::string eqwName = "hist-" + std::string(termType) + "-equal-width-" + std::to_string(w) + ".txt";
        std::string eqdName = "hist-" + std::string(termType) + "-equal-depth-" + std::to_string(w) + ".txt";
        hist.writeEqWidth(eqwName.c_str());
        hist.writeEqDepth(eqdName.c_str());
    }

    int defaultEqWidthSize = 1E2;
    int defaultEqDepthSize = 1E3;
    Histogram hist(hv, defaultEqWidthSize, defaultEqDepthSize);
    std::string eqwName = "hist-" + std::string(termType) + "-equal-width.txt";
    std::string eqdName = "hist-" + std::string(termType) + "-equal-depth.txt";
    hist.writeEqWidth(eqwName.c_str());
    hist.writeEqDepth(eqdName.c_str());
}
