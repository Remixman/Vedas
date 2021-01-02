#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <raptor2/raptor2.h>
#include "vedas.h"
#include "LinkedData.h"
#include "RdfData.h"
#include "VedasStorage.h"

using namespace std;

static ofstream cleaned_out;

static void
print_triple(void* user_data, raptor_statement* triple)
{
    try {
        std::string s_str(reinterpret_cast<char*>(raptor_term_to_string(triple->subject)));
        std::string p_str(reinterpret_cast<char*>(raptor_term_to_string(triple->predicate)));
        std::string o_str(reinterpret_cast<char*>(raptor_term_to_string(triple->object)));
        
        // replace \n to empty string
        std::replace(s_str.begin(), s_str.end(), '\n', ' ');
        std::replace(p_str.begin(), p_str.end(), '\n', ' ');
        std::replace(o_str.begin(), o_str.end(), '\n', ' ');
        
        std::string unicodeStart = std::string("\\u");
        if (o_str.find(unicodeStart)==std::string::npos && s_str.find(unicodeStart)==std::string::npos) {
            cleaned_out << s_str << " " << p_str << " " << o_str << " .\n";
        }
    } catch (...) {
        // Ignore
    }
}

void load_nt(const char *f);

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Usage : " << argv[0] << " <N-Triples file path> <Target N-Triples file path>\n";
        return -1;
    }

    cleaned_out.open(argv[2], std::fstream::out);
    load_nt(argv[1]);
    cleaned_out.close();

    std::cout << "Finished!!\n";
    return 0;
}

void load_nt(const char *f) {

    raptor_world *world = nullptr;
    raptor_parser *rdf_parser = nullptr;
    unsigned char *uri_string;
    raptor_uri *uri, *base_uri;

    world = raptor_new_world();
    rdf_parser = raptor_new_parser(world, "ntriples");

    raptor_parser_set_statement_handler(rdf_parser, nullptr, print_triple);

    uri_string = raptor_uri_filename_to_uri_string(f);
    uri = raptor_new_uri(world, uri_string);
    base_uri = raptor_uri_copy(uri);

    raptor_parser_parse_file(rdf_parser, uri, base_uri);

    raptor_free_parser(rdf_parser);

    raptor_free_uri(base_uri);
    raptor_free_uri(uri);
    raptor_free_memory(uri_string);

    raptor_free_world(world);
}
