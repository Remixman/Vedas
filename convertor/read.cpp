#include <iostream>
#include <cstdio>
#include <raptor2/raptor2.h>

static void
print_triple(void* user_data, raptor_statement* triple)
{
  // std::cout << "Subject(" << triple->subject << ")\n";
  // raptor_statement_print_as_ntriples(triple, stdout);
  // fputc('\n', stdout);
  // fputc('\n', stdout);
  // fputc('\n', stdout);
  // fputc('\n', stdout);

  std::cout << raptor_term_to_string(triple->object) << "\n";
}

int main() {
    raptor_world *world = nullptr;
    raptor_parser *rdf_parser = nullptr;
    unsigned char *uri_string;
    raptor_uri *uri, *base_uri;

    world = raptor_new_world();
    rdf_parser = raptor_new_parser(world, "ntriples");

    raptor_parser_set_statement_handler(rdf_parser, nullptr, print_triple);

    uri_string = raptor_uri_filename_to_uri_string("data/test.nt");
    uri = raptor_new_uri(world, uri_string);
    base_uri = raptor_uri_copy(uri);
    
    raptor_parser_parse_file(rdf_parser, uri, base_uri);

    raptor_free_parser(rdf_parser);

    raptor_free_uri(base_uri);
    raptor_free_uri(uri);
    raptor_free_memory(uri_string);

    raptor_free_world(world);

    return 0;
}