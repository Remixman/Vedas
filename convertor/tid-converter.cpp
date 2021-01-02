#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <utility>
#include <raptor2/raptor2.h>

// export LD_LIBRARY_PATH=/home/remixman/raptor/lib:$LD_LIBRARY_PATH
// g++ -std=c++11 -fopenmp tid-converter.cpp -lraptor2 -o tid-converter
// http://librdf.org/raptor/api/
// split --numeric-suffixes -l 10000000 yago.nt yago2s

static std::string triple_file_ext = ".tidx";
static std::string triple_hash_ext = ".tidh";

class tripleid_hash {
public:
  tripleid_hash() {
    counter = 1;
  }
  void add_triple(std::string &subject, std::string &predicate, std::string &object) {
    clean_term(subject); clean_term(predicate); clean_term(object);
    
    add_term(subject);
    add_term(predicate);
    add_term(object);

    unsigned subject_id = get_id_from_term(subject);
    unsigned predicate_id = get_id_from_term(predicate);
    unsigned object_id = get_id_from_term(object);
    
    // std::cout << "ADD (" << subject_id << "," << predicate_id << "," << object_id << ")\n";
    triples.push_back( std::make_tuple(subject_id, predicate_id, object_id) );
  }
  void add_term(std::string &term) {
    if (hash_table.count(term) == 0) {
      hash_table[term] = counter;
      counter += 1;
    }
  }
  unsigned get_id_from_term(std::string &term) {
    return hash_table[term];
  }
  void clean_term(std::string &term) {
    term.erase(std::remove(term.begin(), term.end(), '\n'), term.end());
  }
  void print_table(std::ostream &os) {
    for (auto it = hash_table.begin(); it != hash_table.end(); ++it) {
      os << it->second << " " << it->first << "\n";
    }
  }
  void write_triple_file(std::string &original_filename) {
    std::string new_name = original_filename + triple_file_ext;
    #pragma omp critical
    std::cout << "WRITE " << triples.size() << " TRIPLES TO FILE " << new_name << "\n";
    std::ofstream outf(new_name, std::ios::out);    
    for (auto it = triples.begin(); it != triples.end(); ++it) {
      outf << std::get<0>(*it) << " " << std::get<1>(*it) << " " << std::get<2>(*it) << "\n";
    }
    outf.close();    
  }
  void write_hash_table_file(std::string &original_filename) {
    std::string new_name = original_filename + triple_hash_ext;
    #pragma omp critical
    std::cout << "WRITE " << hash_table.size() << " HASH RECORDS TO FILE " << new_name << "\n";
    std::ofstream outf(new_name, std::ios::out);
    print_table(outf);
    outf.close();
  }
private: 
  std::map<std::string, unsigned> hash_table;
  std::vector<std::tuple<unsigned, unsigned, unsigned>> triples;
  unsigned counter;
};

static tripleid_hash th;

static void
hash_triple(void* user_data, raptor_statement* triple) 
{
  std::string subject((const char *)raptor_term_to_string(triple->subject));
  std::string predicate((const char *)raptor_term_to_string(triple->predicate));
  std::string object((const char *)raptor_term_to_string(triple->object));

  th.add_triple(subject, predicate, object);
}

int main(int argc, char *argv[]) {

  /*if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <n-triples file name>\n";
    return -1;
  }*/

  raptor_world* world = raptor_new_world();

  // raptor_parser* rdf_parser = raptor_new_parser(world, "ntriple");
  raptor_parser* rdf_parser = raptor_new_parser(world, "turtle");
  
  /*std::string ntfiles[16] = {
    "/media/noo/hd2/DATA/yago/yago2s00",
    "/media/noo/hd2/DATA/yago/yago2s01",
    "/media/noo/hd2/DATA/yago/yago2s02",
    "/media/noo/hd2/DATA/yago/yago2s03",
    "/media/noo/hd2/DATA/yago/yago2s04",
    "/media/noo/hd2/DATA/yago/yago2s05",
    "/media/noo/hd2/DATA/yago/yago2s06",
    "/media/noo/hd2/DATA/yago/yago2s07",
    "/media/noo/hd2/DATA/yago/yago2s08",
    "/media/noo/hd2/DATA/yago/yago2s09",
    "/media/noo/hd2/DATA/yago/yago2s10",
    "/media/noo/hd2/DATA/yago/yago2s11",
    "/media/noo/hd2/DATA/yago/yago2s12",
    "/media/noo/hd2/DATA/yago/yago2s13",
    "/media/noo/hd2/DATA/yago/yago2s14",
    "/media/noo/hd2/DATA/yago/yago2s15"
  };*/
  std::string ntfiles[1] = {
    "/media/noo/hd2/DATA/yago3/yagoTaxonomy.ttl"
  };
  
  for (auto s : ntfiles) {
    std::cout << "PARSE FILE : " << s.c_str() << "\n";
    unsigned char *uri_string = raptor_uri_filename_to_uri_string(s.c_str());
    raptor_uri* nt_uri = raptor_new_uri(world, uri_string);
    raptor_uri* base_uri = raptor_uri_copy(nt_uri);
    raptor_serializer* rdf_serializer;

    raptor_parser_set_statement_handler(rdf_parser, NULL, hash_triple);
    raptor_parser_parse_file(rdf_parser, nt_uri, NULL);
    
    raptor_free_uri(nt_uri);
  }

  // std::string fname = "/media/noo/hd2/DATA/yago/yago2s.nt";
  std::string fname = "/media/noo/hd2/DATA/yago3/yagoTaxonomy.ttl";
#pragma omp parallel sections
{
  #pragma omp section
  th.write_triple_file(fname);

  #pragma omp section
  th.write_hash_table_file(fname);
}

  raptor_free_parser(rdf_parser);
  raptor_free_world(world);

  return 0;
}