#ifndef TRIPLEPATTERNNODE_H
#define TRIPLEPATTERNNODE_H

#include <string>
#include "../TriplePattern.h"
#include "GraphNode.h"

class TriplePatternNode : public GraphNode {
public:
  TriplePatternNode(TriplePattern *tp);
  TriplePattern *getTriplePattern() const;
  std::string toString() const;
private:
  TriplePattern *tp;
};

#endif
