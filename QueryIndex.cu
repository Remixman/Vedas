#include <stdio.h>
#include <stdlib.h>
#include "QueryIndex.h"

QueryIndex::QueryIndex() {

}

void QueryIndex::setL1Index(TYPEID_HOST_VEC *l1_values, TYPEID_HOST_VEC *l1_offsets) {
    this->l1_values = l1_values;
    this->l1_offsets = l1_offsets;
}

void QueryIndex::setL2Index(TYPEID_HOST_VEC *l2_values, TYPEID_HOST_VEC *l2_offsets) {
    this->l2_values = l2_values;
    this->l2_offsets = l2_offsets;
}

TYPEID_HOST_VEC *QueryIndex::getL1IndexValues() const {
    return this->l1_values;
}

TYPEID_HOST_VEC *QueryIndex::getL1IndexOffsets() const {
    return this->l1_offsets;
}

TYPEID_HOST_VEC *QueryIndex::getL2IndexValues() const {
    return this->l2_values;
}

TYPEID_HOST_VEC *QueryIndex::getL2IndexOffsets() const {
    return this->l2_offsets;
}
