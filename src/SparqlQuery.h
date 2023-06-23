#ifndef SPARQLQUERY_H
#define SPARQLQUERY_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include "vedas.h"
#include "TriplePattern.h"

class SparqlQuery {
public:
    // SparqlQuery(std::string &q, DICTTYPE &so_map, DICTTYPE &p_map);
    SparqlQuery(const char *q, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map);
    // SparqlQuery(std::vector<TriplePattern> &patterns, std::vector<std::string> &variables);
    std::vector<TriplePattern> getPatterns();
    TriplePattern getPattern(size_t i);
    TriplePattern *getPatternPtr(size_t i);
    std::vector<TriplePattern> * getPatternsPtr();
    std::map<std::string, int> &getVarCountMap();
    size_t getPatternNum() const;
    std::vector<std::string> getVariables() const;
    std::set<std::string> getSelectedVariables() const;
    bool isStarShaped() const;
    bool isStarBasedShaped() const;
    bool isLinearShaped() const;
    std::string getStarCenterVariable() const;
    void splitStarQuery(int expectGpuNum, std::vector<std::vector<size_t>>& groups);
    void print() const;
private:
    std::vector<TriplePattern> patterns;
    std::vector<std::string> variables;
    std::set<std::string> selected_variables;
    std::map<std::string, int> vars_count;
    int maxVarCount = 0;
    bool linearShaped = false;
    bool starShaped = false;
    bool starBasedShaped = false;
    std::string starCenterVar = "";
};

#endif // SPARQLQUERY_H
