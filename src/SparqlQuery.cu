#include <iostream>
#include <raptor2/raptor2.h>
#include <rasqal/rasqal.h>
#include "vedas.h"
#include "SparqlQuery.h"


// SparqlQuery::SparqlQuery(std::string &q, DICTTYPE &so_map, DICTTYPE &p_map) {
//     this(q.c_str(), so_map, p_map);
// }

void toVedasString(rasqal_literal *l, std::string &s) {
    switch (l->type) {
      case RASQAL_LITERAL_VARIABLE:
        s = "?" + std::string(reinterpret_cast<const char *>(l->value.variable->name));
        break;
      case RASQAL_LITERAL_URI:
        s = "<" + std::string(reinterpret_cast<const char *>(raptor_uri_as_string(l->value.uri))) + ">";
        break;
      case RASQAL_LITERAL_STRING:
      case RASQAL_LITERAL_UDT:
        s = "\"" + std::string(reinterpret_cast<const char *>(l->string)) + "\"";
        break;
      default:
        std::cout << "ERROR !!!";
    }
  }

SparqlQuery::SparqlQuery(const char *q, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map) {
    raptor_world *raptor_world = raptor_new_world();
    rasqal_world *world = rasqal_new_world();
    const unsigned char *uri = reinterpret_cast<const unsigned char *>("http://example.org/");
    raptor_uri *base_uri = raptor_new_uri(raptor_world, uri);
    rasqal_query *rq = rasqal_new_query(world, "sparql", nullptr);

    const unsigned char *query_string = reinterpret_cast<const unsigned char *>(q);
    rasqal_query_prepare(rq, query_string, base_uri);

    // Variables
    raptor_sequence *var_seq = rasqal_query_get_all_variable_sequence(rq);
    int var_num = raptor_sequence_size(var_seq);
    for (int i = 0; i < var_num; ++i) {
      rasqal_variable* v = rasqal_query_get_variable(rq, i);
      this->variables.push_back(reinterpret_cast<const char *>(v->name));
      this->variables.back() = "?" + this->variables.back();
    }

    // Graph pattern
    rasqal_graph_pattern *gp = rasqal_query_get_query_graph_pattern(rq);
    int triple_idx = 0;
    while (true) {
        rasqal_triple *triple = rasqal_graph_pattern_get_triple(gp, triple_idx);
        if (!triple) {
            break;
        }

        std::string subject, predicate, object;
        toVedasString(triple->subject, subject);
        toVedasString(triple->predicate, predicate);
        toVedasString(triple->object, object);

        std::string query_name = "q" + std::to_string(triple_idx);
        TriplePattern tp(triple_idx, query_name, subject, predicate, object, so_map, p_map, l_map);
        
        if (tp.subjectIsVariable()) vars_count[tp.getSubject()] += 1;
        if (tp.predicateIsVariable()) vars_count[tp.getPredicate()] += 1;
        if (tp.objectIsVariable()) vars_count[tp.getObject()] += 1;

        if (tp.getVariableNum() == 1) boundedNum++;
        
        this->patterns.push_back(tp);

        triple_idx++;
    }

    raptor_sequence *selected_var_seq = rasqal_query_get_bound_variable_sequence(rq);
    for (int i = 0; i < raptor_sequence_size(selected_var_seq); ++i) {
        rasqal_variable* v = (rasqal_variable*)raptor_sequence_get_at(selected_var_seq, i);
        std::string vs(reinterpret_cast<const char *>(v->name)); vs = "?" + vs;
        this->selected_variables.insert(vs);
    }
    
    // Check graph shape
    std::multiset<int, std::greater<int> > degrees;
    std::string varWithMaxCount = "";
    for (std::pair<std::string, int> vc: vars_count) {
        degrees.insert(vc.second);
        if (vc.second >= 3) {
            this->star_centers.push_back(vc.first);
        }
        if (vc.second > this->maxVarCount) {
            this->maxVarCount = vc.second;
            varWithMaxCount = vc.first;
        }
    }
    // maxVarCount is degree. If one of them has degree N, this query is star-shaped
    if (this->maxVarCount <= 2) {
        this->linearShaped = true;
    } else {
        if (this->maxVarCount == patterns.size()) {
            this->starShaped = true;
            this->starCenterVar = varWithMaxCount;
        }
        
        size_t gte3 = 0, gte2TotalDeg = 0;
        for (auto d: degrees) if (d >= 3) { gte3++; gte2TotalDeg += d; }
        if (patterns.size() >= 4 && this->maxVarCount >= 3 && gte3 == 1 && !this->starShaped) {
            this->starBasedShaped = true;
            this->starCenterVar = varWithMaxCount;
        }
        
        if (patterns.size() >= 5 && this->maxVarCount >= 3 && !this->starShaped && !this->starBasedShaped) {
            if (gte2TotalDeg - (gte3 - 1) == patterns.size()) {
                this->snowflakeShaped = true;
                this->starCount = gte3;
            }
        }
    }
    

    rasqal_free_query(rq);
    raptor_free_uri(base_uri);
    rasqal_free_world(world);
}

// SparqlQuery::SparqlQuery(std::vector<TriplePattern> &patterns, std::vector<std::string> &variables) {
//     this->variables = variables;
//     this->patterns = patterns;
// }

bool compareTriplaPatternCardinality(TriplePattern &tp1, TriplePattern &tp2) {
    return tp1.estimate_rows < tp2.estimate_rows;
}

void SparqlQuery::splitStarQuery(int expectGpuNum, std::vector<std::vector<size_t>>& groups) {
    std::vector<TriplePattern> triplePatterns = patterns;
    std::sort(triplePatterns.begin(), triplePatterns.end(), compareTriplaPatternCardinality);
    
    groups.resize(expectGpuNum);
    for (int i = 0; i < triplePatterns.size(); ++i) {
        groups[i % expectGpuNum].push_back(triplePatterns[i].getId());
        if (boundedNum == 1 && triplePatterns[i].estimate_rows < 1000) {
            groups[(i % expectGpuNum) + 1].push_back(triplePatterns[i].getId());
        }
    }
}

void SparqlQuery::splitSnowflakeQuery(int expectGpuNum, std::vector<std::vector<size_t>>& groups, std::string& groupJoinVar) {
    groups.resize(expectGpuNum);
    std::vector<std::vector<int>> g(star_centers.size());

    int g1Card = 0, g2Card = 0;
    std::string &c1 = star_centers[0];
    std::string &c2 = star_centers[1];
    size_t commonTripleIdx = 0;
    for (size_t i = 0; i < patterns.size(); ++i) {
        if (patterns[i].hasVariable(c1)) {
            if (patterns[i].hasVariable(c2)) {
                commonTripleIdx = i;
            } else {
                groups[0].push_back(i);
                g1Card += patterns[i].estimate_rows;
            }
        } else if (patterns[i].hasVariable(c2)) {
            groups[1].push_back(i);
            g2Card += patterns[i].estimate_rows;
        }
    }
    
    if (groups[0].size() < groups[1].size()) {
        groups[0].push_back(commonTripleIdx);
        groupJoinVar = c2;
    } else if (groups[1].size() < groups[0].size()) {
        groups[1].push_back(commonTripleIdx);
        groupJoinVar = c1;
    } else if (g1Card < g2Card) {
        groups[0].push_back(commonTripleIdx);
        groupJoinVar = c2;
    } else {
        groups[1].push_back(commonTripleIdx);
        groupJoinVar = c1;
    }
    
    std::vector<TriplePattern> bgp[expectGpuNum];
    for (int g = 0; g < 2; ++g) {
        for (int i = 0; i < groups[g].size(); ++i) bgp[g].push_back(patterns[groups[g][i]]);
        std::sort(bgp[g].begin(), bgp[g].end(), compareTriplaPatternCardinality);
        groups[g].resize(0);
        for (int i = 0; i < bgp[g].size(); ++i) groups[g].push_back(bgp[g][i].getId());        
        // std::cout << "Group " << (g+1) << " : ";
        // for (int i = 0; i < groups[g].size(); ++i) std::cout << groups[g][i] << ' '; 
        // std::cout << "\n";
    }
}

std::vector<TriplePattern> SparqlQuery::getPatterns() { return this->patterns; }
std::vector<TriplePattern> * SparqlQuery::getPatternsPtr() { return &(this->patterns); }
TriplePattern SparqlQuery::getPattern(size_t i) { return this->patterns[i]; }
TriplePattern *SparqlQuery::getPatternPtr(size_t i) { return &(this->patterns[i]); }
size_t SparqlQuery::getPatternNum() const { return this->patterns.size(); }
std::map<std::string, int> & SparqlQuery::getVarCountMap() { return this->vars_count; }
bool SparqlQuery::isStarShaped() const { return this->starShaped; }
bool SparqlQuery::isStarBasedShaped() const { return this->starBasedShaped; }
bool SparqlQuery::isLinearShaped() const { return this->linearShaped; }
bool SparqlQuery::isSnowflakeShaped() const { return this->snowflakeShaped; }
std::string SparqlQuery::getStarCenterVariable() const { return this->starCenterVar; }
std::vector<std::string> SparqlQuery::getStarCenters() const { return this->star_centers; }

std::vector<std::string> SparqlQuery::getVariables() const {
    return this->variables;
}

std::set<std::string> SparqlQuery::getSelectedVariables() const {
    return this->selected_variables;
}

void SparqlQuery::printShape() const {
    if (linearShaped) std::cout << "LINEAR-SHAPED\n";
    if (starShaped) std::cout << "STAR-SHAPED\n";
    if (starBasedShaped) std::cout << "MULTI-CHAIN STAR-SHAPED\n";
    if (snowflakeShaped) std::cout << "SNOWFLAKE-SHAPED\n";
    if (!linearShaped && !starShaped && !starBasedShaped && !snowflakeShaped) std::cout << "OTHER SHAPE\n";
}

void SparqlQuery::print() const {
    std::cout << "======================= SPARQL Query ========================\n";
    std::cout << " Variables : ";
    for (size_t i = 0; i < variables.size(); ++i)
        std::cout << variables[i] << ",";
    std::cout << "\n";
    std::cout << " Triple Patterns : \n";
    for (size_t i = 0; i < patterns.size(); ++i) {
        std::cout << "    ";
        patterns[i].print();
    }
}
