#include "RdfData.h"
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

RdfData::RdfData(TYPEID_HOST_VEC &subjects, TYPEID_HOST_VEC &predicates, TYPEID_HOST_VEC &objects)
{
    size_t n = subjects.size();
    this->subjects = subjects;
    this->predicates = predicates;
    this->objects = objects;
}

size_t RdfData::size() const {
    return this->subjects.size();
}

TYPEID_HOST_VEC& RdfData::getSubject() { return this->subjects; }
TYPEID_HOST_VEC& RdfData::getPredicate() { return this->predicates; }
TYPEID_HOST_VEC& RdfData::getObject() { return this->objects; }

void RdfData::write(const char *fname) {
    std::ofstream out;
    out.open(fname, std::ios::out);

    size_t len = this->size();
    for (size_t i = 0; i < len; i++) {
        out << this->subjects[i] << ' '
            << this->predicates[i] << ' '
            << this->objects[i] << '\n';
    }

    out.close();
}
