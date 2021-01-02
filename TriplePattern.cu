#include <iostream>
#include <string>
#include <cassert>
#include "vedas.h"
#include "TriplePattern.h"

using namespace std;

TriplePattern::TriplePattern(std::string name, std::string s, std::string p, std::string o,
                             DICTTYPE &so_dict, DICTTYPE &p_dict) {
  this->isVar[0] = (s.size() > 0 && s[0] == '?');
  this->isVar[1] = (p.size() > 0 && p[0] == '?');
  this->isVar[2] = (o.size() > 0 && o[0] == '?');

  assert(this->isVar[0] || (so_dict.count(s) > 0));
  assert(this->isVar[1] || (p_dict.count(p) > 0));
  assert(this->isVar[2] || (so_dict.count(o) > 0));

  if (this->isVar[0]) {
      subject = s; subject_id = 0;
  } else {
      subject_id = so_dict[s];
  }

  if (this->isVar[1]) {
      predicate = p; predicate_id = 0;
  } else {
      predicate_id = p_dict[p];
  }

  if (this->isVar[2]) {
      object = o; object_id = 0;
  } else {
      object_id = so_dict[o];
  }

  this->name = name;
}

std::string TriplePattern::getName() const { return this->name; }
TYPEID TriplePattern::getSubjectId() const { return this->subject_id; }
TYPEID TriplePattern::getPredicateId() const { return this->predicate_id; }
TYPEID TriplePattern::getObjectId() const { return this->object_id; }
string TriplePattern::getSubject() const { return this->subject; }
string TriplePattern::getPredicate() const { return this->predicate; }
string TriplePattern::getObject() const { return this->object; }
size_t TriplePattern::getVariableNum() const {
    return static_cast<size_t>(this->isVar[0]) +
            static_cast<size_t>(this->isVar[1]) +
            static_cast<size_t>(this->isVar[2]);
}
size_t TriplePattern::getVariableBitmap() const {
    return (this->isVar[0] * 4) + (this->isVar[1] * 2) + (this->isVar[2]);
}
bool TriplePattern::subjectIsVariable() const { return this->isVar[0]; }
bool TriplePattern::predicateIsVariable() const { return this->isVar[1]; }
bool TriplePattern::objectIsVariable() const { return this->isVar[2]; }

bool TriplePattern::hasVariable(std::string v) const {
    if (subjectIsVariable() && getSubject() == v) return true;
    if (predicateIsVariable() && getPredicate() == v) return true;
    if (objectIsVariable() && getObject() == v) return true;
    return false;
}

void TriplePattern::print() const {
    if (isVar[0]) std::cout << subject << "[?] ";
    else std::cout << subject_id << " ";
    if (isVar[1]) std::cout << predicate << "[?] ";
    else std::cout << predicate_id << " ";
    if (isVar[2]) std::cout << object << "[?]";
    else std::cout << object_id;

    std::cout << "\n";
}
