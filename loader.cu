#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <raptor2/raptor2.h>
#include "vedas.h"

using namespace std;

// http://librdf.org/raptor/api/tutorial-parser-example.html
static vector<TYPEID> *s_p, *p_p, *o_p;
static DICTTYPE *so_map_p, *p_map_p, *l_map_p;
static REVERSE_DICTTYPE *r_so_map_p, *r_p_map_p, *r_l_map_p;
static TYPEID so_count = 1, p_count = 1, l_count = 5E7;

static void
triple_statement_handler(void* user_data, raptor_statement* triple)
{
    std::string s_str(reinterpret_cast<char*>(raptor_term_to_string(triple->subject)));
    std::string p_str(reinterpret_cast<char*>(raptor_term_to_string(triple->predicate)));
    std::string o_str(reinterpret_cast<char*>(raptor_term_to_string(triple->object)));

    assert(triple->subject->type != RAPTOR_TERM_TYPE_LITERAL);

    if (so_map_p->count(s_str) == 0) {
        (*so_map_p)[s_str] = so_count++;
        (*r_so_map_p)[so_count - 1] = s_str;
    }

    bool enable_literal_dict = false;
#define LITERAL_DICT
#ifdef LITERAL_DICT
    enable_literal_dict = true;
#endif
    if (triple->object->type == RAPTOR_TERM_TYPE_LITERAL && enable_literal_dict) {
        if (l_map_p->count(o_str) == 0) {
            (*l_map_p)[o_str] = l_count++;
            (*r_l_map_p)[l_count - 1] = o_str;
        }
        o_p->push_back( (*l_map_p)[o_str] );
    } else {
        if (so_map_p->count(o_str) == 0) {
            (*so_map_p)[o_str] = so_count++;
            (*r_so_map_p)[so_count - 1] = o_str;
        }
        o_p->push_back( (*so_map_p)[o_str] );
    }

    if (p_map_p->count(p_str) == 0) {
        (*p_map_p)[p_str] = p_count++;
        (*r_p_map_p)[p_count - 1] = p_str;
    }

    s_p->push_back( (*so_map_p)[s_str] );
    p_p->push_back( (*p_map_p)[p_str] );
}

void load_rdf(const char *f, vector<TYPEID> &s, vector<TYPEID> &p, vector<TYPEID> &o,
               DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
               REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map) {
    s_p = &s;
    p_p = &p;
    o_p = &o;
    so_map_p = &so_map;
    p_map_p = &p_map;
    l_map_p = &l_map;
    r_so_map_p = &r_so_map;
    r_p_map_p = &r_p_map;
    r_l_map_p = &r_l_map;

    raptor_world *world = nullptr;
    raptor_parser *rdf_parser = nullptr;
    unsigned char *uri_string;
    raptor_uri *uri, *base_uri;

    world = raptor_new_world();
    rdf_parser = raptor_new_parser(world, "ntriples");

    raptor_parser_set_statement_handler(rdf_parser, nullptr, triple_statement_handler);

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

void load_dummy_rdf(vector<TYPEID> &s, vector<TYPEID> &p, vector<TYPEID> &o,
                    DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    const int N = 23;
    std::string array[N][3] = {
        { "1",  "6", "1"  },
        { "1",  "6", "14" },
        { "1",  "6", "17" },
        { "1",  "6", "21" },
        { "1",  "7", "20" },
        { "1",  "8", "19" },
        { "1",  "9", "21" },
        { "2",  "7", "19" },
        { "2",  "8", "16" },
        { "2",  "9",  "9" },
        { "3",  "4",  "5" },
        { "3",  "5",  "3" },
        { "3",  "6",  "4" },
        { "3",  "6",  "6" },
        { "3",  "6", "20" },
        { "3",  "7",  "9" },
        { "3",  "7", "20" },
        { "3",  "7", "25" },
        { "4",  "7", "10" },
        { "5", "20", "12" },
        { "6",  "7", "15" },
        { "6", "25", "18" },
        { "6", "25", "22" }
    };

    string s_str, p_str, o_str;
    TYPEID so_count = 1, p_count = 1;
    for (size_t i = 0; i < N; i++) {
        s_str = array[i][0]; p_str = array[i][1]; o_str = array[i][2];
        if (so_map.count(s_str) == 0) {
            so_map[s_str] = so_count++;
            r_so_map[so_count - 1] = s_str;
        }
        if (so_map.count(o_str) == 0) {
            so_map[o_str] = so_count++;
            r_so_map[so_count - 1] = o_str;
        }
        if (p_map.count(p_str) == 0) {
            p_map[p_str] = p_count++;
            r_p_map[p_count - 1] = p_str;
        }

        s.push_back( so_map[s_str] );
        p.push_back( p_map[p_str] );
        o.push_back( so_map[o_str] );
    }
}
