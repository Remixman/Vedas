#include "TriplePatternNode.h"

TriplePatternNode::TriplePatternNode(TriplePattern *tp) {
    this->tp = tp;
}

TriplePattern *TriplePatternNode::getTriplePattern() const {
    return this->tp;
}

std::string TriplePatternNode::toString() const {
    return "Triple Pattern";
}
