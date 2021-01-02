#include <iostream>
#include <cstdio>
#include <vector>
#include <raptor2/raptor2.h>
#include <rasqal/rasqal.h>

static int pattern_walker(rasqal_query *query, rasqal_graph_pattern *gp, void *user_data) {
  std::cout << "TEST\n";
}

std::string toVedasString(rasqal_literal *l) {
  switch (l->type) {
    case RASQAL_LITERAL_VARIABLE:
      return std::string("?") + std::string(reinterpret_cast<const char *>(l->value.variable->name));
    case RASQAL_LITERAL_URI:
      return std::string(reinterpret_cast<const char *>(raptor_uri_as_string(l->value.uri)));
    case RASQAL_LITERAL_STRING:
    case RASQAL_LITERAL_UDT:
      return std::string(reinterpret_cast<const char *>(l->string));
    default:
      std::cout << "ERROR !!!";
      return std::string("");
  }
}

int main() {
    raptor_world *raptor_world = raptor_new_world();
    rasqal_world *world = rasqal_new_world();
    rasqal_query_results *results;
    const unsigned char *uri = reinterpret_cast<const unsigned char *>("http://example.org/foo");
    raptor_uri *base_uri = raptor_new_uri(raptor_world, uri);
    rasqal_query *rq = rasqal_new_query(world, "sparql", nullptr);

    const unsigned char *query_string = reinterpret_cast<const unsigned char *>("SELECT ?s ?p ?o WHERE { ?s <aa> ?o . ?s ?p <bb> . }");
    rasqal_query_prepare(rq, query_string, base_uri);

    // Variables
    std::vector<std::string> variables;
    raptor_sequence *var_seq = rasqal_query_get_all_variable_sequence(rq);
    int var_num = raptor_sequence_size(var_seq);
    for (int i = 0; i < var_num; ++i) {
      rasqal_variable* v = rasqal_query_get_variable(rq, i);
      variables.push_back(reinterpret_cast<const char *>(v->name));
    }
    for (std::string s : variables) std::cout << s << "\n";

    // Graph pattern
    rasqal_graph_pattern *gp = rasqal_query_get_query_graph_pattern(rq);
    int triple_idx = 0;
    while (true) {
      rasqal_triple *triple = rasqal_graph_pattern_get_triple(gp, triple_idx);
      if (!triple) break;


      std::cout << toVedasString(triple->subject) << "\n";
      std::cout << toVedasString(triple->predicate) << "\n";
      std::cout << toVedasString(triple->object) << "\n";
      // std::cout << raptor_term_to_string(triple->object) << "\n";

      // rasqal_triple_print(triple, stdout);
      // fputc('\n', stdout);

      triple_idx++;
    }
    
    rasqal_free_query(rq);
    raptor_free_uri(base_uri);
    rasqal_free_world(world);
  
    return 0;
}