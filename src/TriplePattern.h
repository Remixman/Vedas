#ifndef TRIPLEPATTERN_H
#define TRIPLEPATTERN_H

#include <string>
#include "vedas.h"

class TriplePattern {
  public:
    TriplePattern(size_t id, std::string &name, std::string &s, std::string &p, std::string &o, DICTTYPE &so_dict, DICTTYPE &p_dict, DICTTYPE &l_dict);
    size_t getId() const;
    std::string getName() const;
    TYPEID getSubjectId() const;
    TYPEID getPredicateId() const;
    TYPEID getObjectId() const;
    std::string getSubject() const;
    std::string getPredicate() const;
    std::string getObject() const;
    size_t getVariableNum() const;
    size_t getVariableBitmap() const;
    bool subjectIsVariable() const;
    bool predicateIsVariable() const;
    bool objectIsVariable() const;
    bool hasVariable(std::string v) const;
    std::string hasCommonVariable(const TriplePattern &tp);
    void print() const;
    std::string toString() const;

    // For optimization
    size_t estimate_rows;
  private:
    size_t id;
    std::string name;
    std::string subject, predicate, object;
    TYPEID subject_id, predicate_id, object_id;
    bool isVar[3];
};

#endif
