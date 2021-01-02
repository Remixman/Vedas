#include "RdfData.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

RdfData::RdfData(std::vector<TYPEID> subjects, std::vector<TYPEID> predicates, std::vector<TYPEID> objects)
{
    size_t n = subjects.size();

    this->subjects.resize(n);
    this->predicates.resize(n);
    this->objects.resize(n);

    thrust::copy(subjects.begin(), subjects.end(), this->subjects.begin());
    thrust::copy(predicates.begin(), predicates.end(), this->predicates.begin());
    thrust::copy(objects.begin(), objects.end(), this->objects.begin());
}

size_t RdfData::size() const {
    return this->subjects.size();
}

TYPEID_HOST_VEC& RdfData::getSubject() { return this->subjects; }
TYPEID_HOST_VEC& RdfData::getPredicate() { return this->predicates; }
TYPEID_HOST_VEC& RdfData::getObject() { return this->objects; }
