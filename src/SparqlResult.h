#ifndef SPARQLRESULT_H
#define SPARQLRESULT_H

#include <string>
#include <vector>
#include "vedas.h"
#include "FullRelationIR.h"

class SparqlResult
{
public:
    SparqlResult();
    void setResult(IR *resultIR);
    FullRelationIR *getResultIR();
    std::vector<std::string> getHeaderVariables();
    std::vector<std::vector<TYPEID>> get();
    std::vector<TYPEID> get(size_t i);
    void printResult(REVERSE_DICTTYPE &so_dict, REVERSE_DICTTYPE &p_dict, REVERSE_DICTTYPE &l_dict) const;
private:
    FullRelationIR *resultIR;
    std::vector<std::vector<TYPEID>> results;
    std::vector<std::string> header;
};

#endif // SPARQLRESULT_H
