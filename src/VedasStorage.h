#ifndef VEDASSTORAGE_H
#define VEDASSTORAGE_H

#include "vedas.h"
#include "DataMetaInfo.h"
#include "RdfData.h"

class VedasStorage
{
public:
    VedasStorage(RdfData &rdfData, bool preload, bool fullIndex);  // Use from VedasBuild
    VedasStorage(const char *fname, bool preload, bool fullIndex); // Use from VedasQuery/VedasBench

    size_t getSubjectIndexSize() const;
    size_t getSubjectPredicateIndexSize() const;
    size_t getSubjectObjectIndexSize() const;
    size_t getPredicateIndexSize() const;
    size_t getObjectIndexSize() const;
    size_t getObjectSubjectIndexSize() const;
    size_t getObjectPredicateIndexSize() const;
    size_t getTripleSize() const;

    TYPEID_HOST_VEC *getSubjectIndexValues();
    TYPEID_HOST_VEC *getSubjectIndexOffsets();
    TYPEID_HOST_VEC *getSubjectPredicateIndexValues();
    TYPEID_HOST_VEC *getSubjectPredicateIndexOffsets();
    TYPEID_HOST_VEC *getSubjectObjectIndexValues();
    TYPEID_HOST_VEC *getSubjectObjectIndexOffsets();

    TYPEID_HOST_VEC *getPredicateIndexValues();
    TYPEID_HOST_VEC *getPredicateIndexOffsets();
    TYPEID_HOST_VEC *getPredicateSubjectIndexValues();
    TYPEID_HOST_VEC *getPredicateSubjectIndexOffsets();
    TYPEID_HOST_VEC *getPredicateObjectIndexValues();
    TYPEID_HOST_VEC *getPredicateObjectIndexOffsets();

    TYPEID_HOST_VEC *getObjectIndexValues();
    TYPEID_HOST_VEC *getObjectIndexOffsets();

    TYPEID_HOST_VEC *getObjectPredicateIndexValues();
    TYPEID_HOST_VEC *getObjectPredicateIndexOffsets();

    TYPEID_HOST_VEC *getPSdata();
    TYPEID_HOST_VEC *getPOdata();
    TYPEID_HOST_VEC *getOSdata();

    TYPEID_HOST_VEC *getSPOdata();
    TYPEID_HOST_VEC *getSOPdata();
    TYPEID_HOST_VEC *getPSOdata();
    TYPEID_HOST_VEC *getPOSdata();
    TYPEID_HOST_VEC *getOPSdata();
    TYPEID_HOST_VEC *getOSPdata();

    TYPEID_DEVICE_VEC *getDeviceSPOdata();
    TYPEID_DEVICE_VEC *getDeviceSOPdata();
    TYPEID_DEVICE_VEC *getDevicePSOdata();
    TYPEID_DEVICE_VEC *getDevicePOSdata();
    TYPEID_DEVICE_VEC *getDeviceOPSdata();
    TYPEID_DEVICE_VEC *getDeviceOSPdata();

    PredicateIndexStat psStat() const;
    PredicateIndexStat poStat() const;
    double psBoundaryCompactness() const;
    double poBoundaryCompactness() const;

    bool isPreload() const;

    void printSPOIndex() const;
    void printSOPIndex() const;
    void printOPSIndex() const;

    void open(const char *fname);
    void write(const char *fname);
    void writeHistogram(const char *termType, TYPEID_DEVICE_VEC v);
private:
    bool preload = false;
    bool fullIndex = false;
    DataMetaInfo metaInfo;

    TYPEID_HOST_VEC s_idx_values, s_idx_offsets;
    TYPEID_HOST_VEC sp_p_idx_values, sp_idx_offsets;
    TYPEID_HOST_VEC so_o_idx_values, so_idx_offsets;

    TYPEID_HOST_VEC p_idx_values, p_idx_offsets;
    TYPEID_HOST_VEC ps_s_idx_values, ps_idx_offsets;
    TYPEID_HOST_VEC po_o_idx_values, po_idx_offsets;
    TYPEID_HOST_VEC ps_data, po_data;

    TYPEID_HOST_VEC o_idx_values, o_idx_offsets;
    TYPEID_HOST_VEC os_data;
    TYPEID_HOST_VEC op_p_idx_values, op_idx_offsets;

    TYPEID_HOST_VEC spo_data, sop_data;
    TYPEID_HOST_VEC pso_data, pos_data;
    TYPEID_HOST_VEC ops_data, osp_data;

    // For store on GPU
    TYPEID_DEVICE_VEC d_spo_data, d_sop_data;
    TYPEID_DEVICE_VEC d_pso_data, d_pos_data;
    TYPEID_DEVICE_VEC d_ops_data, d_osp_data;

    void uploadData();
    void printIndex(char c1, char c2, char c3,
                    const TYPEID_HOST_VEC &l1IdxVals, const TYPEID_HOST_VEC &l1IdxOfssts,
                    const TYPEID_HOST_VEC &l1l2IdxVals, const TYPEID_HOST_VEC &l1l2IdxOfssts,
                    const TYPEID_HOST_VEC &data) const;

    void createIndex1Level(const char *index_name, TYPEID_DEVICE_VEC &v1, TYPEID_DEVICE_VEC &v2, TYPEID_DEVICE_VEC &v3, size_t n,
                           TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets);
    void createIndexPS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                       TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets);
    void createIndexPO(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                       TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets);
    void createIndexOS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                       TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets);

    void createIndex(const char *index_name,
                     TYPEID_DEVICE_VEC &v1, TYPEID_DEVICE_VEC &v2, TYPEID_DEVICE_VEC &v3 /* Input and Output */, size_t n,
                     TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
    void createIndexSPO(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                      TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
    void createIndexSOP(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                      TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
    void createIndexPSO(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                      TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
    void createIndexPOS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                      TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
    void createIndexOSP(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                      TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
    void createIndexOPS(TYPEID_DEVICE_VEC &subjects, TYPEID_DEVICE_VEC &predicate, TYPEID_DEVICE_VEC &objects, size_t n,
                      TYPEID_HOST_VEC &values, TYPEID_HOST_VEC &offsets, TYPEID_HOST_VEC &value2_2, TYPEID_HOST_VEC &offsets2);
};

#endif // VEDASSTORAGE_H
