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

SparqlQuery::SparqlQuery(const char *q, DICTTYPE &so_map, DICTTYPE &p_map) {
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
#ifdef DEBUG
      std::cout << "PARSED VARIABLE ( " << reinterpret_cast<const char *>(v->name) << " )\n";
#endif
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
#ifdef DEBUG
        std::cout << "PARSED PATTERN ( " << subject << " , " << predicate << " , " << object << " )\n";
#endif
        std::string query_name = "q" + std::to_string(triple_idx);
        TriplePattern pattern(triple_idx, query_name, subject, predicate, object, so_map, p_map);
        this->patterns.push_back(pattern);

        triple_idx++;
    }

    raptor_sequence *selected_var_seq = rasqal_query_get_bound_variable_sequence(rq);
    for (int i = 0; i < raptor_sequence_size(selected_var_seq); ++i) {
        rasqal_variable* v = (rasqal_variable*)raptor_sequence_get_at(selected_var_seq, i);
        std::string vs(reinterpret_cast<const char *>(v->name)); vs = "?" + vs;
        this->selected_variables.insert(vs);
#ifdef DEBUG
        std::cout << "SELECT VARIABLE : " << vs << "\n";
#endif
    }

    rasqal_free_query(rq);
    raptor_free_uri(base_uri);
    rasqal_free_world(world);
}

SparqlQuery::SparqlQuery(std::vector<TriplePattern> &patterns, std::vector<std::string> &variables) {
    this->variables = variables;
    this->patterns = patterns;
}

std::vector<TriplePattern> SparqlQuery::getPatterns() { return this->patterns; }
TriplePattern SparqlQuery::getPattern(size_t i) { return this->patterns[i]; }
TriplePattern *SparqlQuery::getPatternPtr(size_t i) { return &(this->patterns[i]); }
size_t SparqlQuery::getPatternNum() const { return this->patterns.size(); }

std::vector<std::string> SparqlQuery::getVariables() const {
    return this->variables;
}

std::set<std::string> SparqlQuery::getSelectedVariables() const {
    return this->selected_variables;
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
