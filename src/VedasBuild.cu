#include <algorithm>
#include <random>
#include <iostream>
#include <set>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <thrust/sort.h>
#include "InputParser.h"
#include "vedas.h"
#include "LinkedData.h"
#include "RdfData.h"
#include "VedasStorage.h"

using namespace std;

void save_dict(const char *fname, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map,
                REVERSE_DICTTYPE &r_l_map);
void reassign_dict_random(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                          DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);
void reassign_dict_bfs(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                       DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
                       REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map);
void reassign_dict_with_dim_reduction(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                    DICTTYPE &so_map, DICTTYPE &p_map,
                    REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);
void reassign_spread(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                     DICTTYPE &so_map, DICTTYPE &p_map,
                     REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);

#define REASSIGN_NONE     0
#define REASSIGN_RAND     1
#define REASSIGN_BFS      2
#define REASSIGN_PCA      3
#define REASSIGN_NMF      4
#define REASSIGN_SD       5   // Sorted with degree number

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Usage : " << argv[0] << " <Database Name> <N-Triples file path>\n";
        return -1;
    }

    InputParser input_parser(argc, argv);
    int reassign_method = REASSIGN_NONE;
    std::string reassign_method_option = input_parser.getCmdOption("-rasm");
    if (reassign_method_option.size() > 0) {
        if (reassign_method_option == "rand")     reassign_method = REASSIGN_RAND;
        else if (reassign_method_option == "bfs") reassign_method = REASSIGN_BFS;
        else if (reassign_method_option == "pca") reassign_method = REASSIGN_PCA;
        else if (reassign_method_option == "nmf") reassign_method = REASSIGN_NMF;
    }
    bool fullIndex = input_parser.cmdOptionExists("-full-index");
    bool literalDict = input_parser.cmdOptionExists("-ldict");
    bool spreadData = input_parser.cmdOptionExists("-spread");

    TYPEID_HOST_VEC subjects, predicates, objects;
    DICTTYPE so_map;   // Subject and Object IRI map
    DICTTYPE p_map;    // Predicate map
    DICTTYPE l_map;    // Literal map
    REVERSE_DICTTYPE r_so_map, r_p_map, r_l_map;  // Reverse map

    auto load_start = std::chrono::high_resolution_clock::now();
    load_rdf(argv[2], subjects, predicates, objects,
           so_map, p_map, l_map, r_so_map, r_p_map, r_l_map, literalDict);
    auto load_end = std::chrono::high_resolution_clock::now();

    set<string> freq_pred_set;

    std::cout << "Subject/object dict size : " << so_map.size() << "\n";
    std::cout << "Predicate dict size : " << p_map.size() << "\n";
    std::cout << "Literal dict size : " << l_map.size() << "\n";

    auto reassign_start = std::chrono::high_resolution_clock::now();

    switch (reassign_method) {
        case REASSIGN_RAND:
            std::cout << "Reassign with Random!\n";
            reassign_dict_random(subjects, predicates, objects, so_map, p_map,
                                 r_so_map, r_p_map);
            break;
        case REASSIGN_BFS:
            std::cout << "Reassign with BFS!\n";
            reassign_dict_bfs(subjects, predicates, objects, so_map, p_map, l_map,
                              r_so_map, r_p_map, r_l_map);
            break;
        case REASSIGN_PCA:
            std::cout << "Reassign with PCA!\n";
            reassign_dict_with_dim_reduction(subjects, predicates, objects, so_map, p_map,
                                             r_so_map, r_p_map);
            break;
    }

    auto reassign_end = std::chrono::high_resolution_clock::now();
    
    if (spreadData) {
        std::cout << "Spread out data ID\n";
        reassign_spread(subjects, predicates, objects, so_map, p_map, r_so_map, r_p_map);
    }

    std::string dict_path = argv[1]; dict_path += ".vdd";
    std::string data_path = argv[1]; data_path += ".vds";

    save_dict(dict_path.c_str(), r_so_map, r_p_map, r_l_map);

    auto copy_start = std::chrono::high_resolution_clock::now();
    RdfData rdfData(subjects, predicates, objects);
    auto copy_end = std::chrono::high_resolution_clock::now();

    auto indexing_start = std::chrono::high_resolution_clock::now();
    VedasStorage *vedasStorage = new VedasStorage(rdfData, false, fullIndex);
    vedasStorage->write(data_path.c_str());
    auto indexing_end = std::chrono::high_resolution_clock::now();

    std::cout << "PSO Boundary Compactness : " << vedasStorage->psBoundaryCompactness() << "\n";
    std::cout << "POS Boundary Compactness : " << vedasStorage->poBoundaryCompactness() << "\n";

    /* DEBUG */
    // std::string text_data_path = argv[1]; text_data_path += ".txt";
    // rdfData.write(text_data_path.c_str());

    /* Save graph file for visualize */
    ofstream node_out("./tools/nodes.txt");
    for (TYPEID node = 1; node <= subjects.size(); ++node) node_out << node << "\n";
    node_out.close();
    ofstream edge_out("./tools/edges.txt");
    for (TYPEID i = 1; i <= subjects.size(); ++i) edge_out << subjects[i] << " " << objects[i] << "\n";
    edge_out.close();

    std::cout << "Build Finished!\n";
    cout << "Loading time  : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(load_end-load_start).count() << " ms.\n";
    cout << "Copy time     : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(copy_end-copy_start).count() << " ms.\n";
    cout << "Indexing time : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(indexing_end-indexing_start).count() << " ms.\n";
    cout << "Reassign time : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(reassign_end-reassign_start).count() << " ms.\n";

    return 0;
}

void save_dict(const char *fname, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map) {
    std::ofstream out;
    out.open(fname, std::fstream::out);

    out << r_so_map.size() << "\n";
    for (auto it = r_so_map.begin(); it != r_so_map.end(); ++it)
        out << it->first << " " << it->second << "\n";
    out << r_p_map.size() << "\n";
    for (auto it = r_p_map.begin(); it != r_p_map.end(); ++it)
        out << it->first << " " << it->second << "\n";
    out << r_l_map.size() << "\n";
    for (auto it = r_l_map.begin(); it != r_l_map.end(); ++it)
        out << it->first << " " << it->second << "\n";

    out.close();
}

void reassign_dict_random(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                  DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::map<TYPEID, TYPEID> so_swap_dict, p_swap_dict;
    size_t so_data_size = so_map.size();
    size_t p_data_size = p_map.size();

    TYPEID_HOST_VEC so_ids;
    std::vector<std::string> so_data;
    for (auto &p: so_map) {
        so_data.push_back(p.first);
        so_ids.push_back(p.second);
    }
    unsigned seed1 = 9283923;
    std::shuffle(so_ids.begin(), so_ids.end(), std::default_random_engine(seed1));
    size_t i = 0;
    for (auto &p: so_map) {
        so_swap_dict[p.second] = so_ids[i]; i++;
    }
    so_map.clear(); r_so_map.clear();
    for (i = 0; i < so_data_size; ++i) {
        so_map[so_data[i]] = so_ids[i];
        r_so_map[so_ids[i]] = so_data[i];
    }

    TYPEID_HOST_VEC p_ids;
    std::vector<std::string> p_data;
    for (auto &p: p_map) {
        p_data.push_back(p.first);
        p_ids.push_back(p.second);
    }
    unsigned seed2 = 36942389;
    std::shuffle(p_ids.begin(), p_ids.end(), std::default_random_engine(seed2));
    i = 0;
    for (auto &p: p_map) {
        p_swap_dict[p.second] = p_ids[i]; i++;
    }
    p_map.clear(); r_p_map.clear();
    for (i = 0; i < p_data_size; ++i) {
        p_map[p_data[i]] = p_ids[i];
        r_p_map[p_ids[i]] = p_data[i];
    }

    for (i = 0; i < s.size(); ++i) s[i] = so_swap_dict[s[i]];
    for (i = 0; i < p.size(); ++i) p[i] = p_swap_dict[p[i]];
    for (i = 0; i < o.size(); ++i) o[i] = so_swap_dict[o[i]];
}

void reassign_dict_bfs(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                       DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
                       REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map) {
    size_t triple_size = std::max(s.size(), o.size());
    std::unordered_map<TYPEID, TYPEID> new_map;

    // Reassign id (BFS)
    LinkedData linkedData(triple_size);
    for (size_t i = 0; i < triple_size; ++i) linkedData.addLink(s[i], o[i], p[i]);
    linkedData.reassignIdByBfs(new_map);

    // Update s, o
    for (size_t i = 0; i < triple_size; ++i) {
        s[i] = new_map[s[i]];
        o[i] = new_map[o[i]];
    }

    // Update so dict
    DICTTYPE new_so_map;
    REVERSE_DICTTYPE new_r_so_map;
    for (auto &p : so_map) {
        new_so_map[p.first] = new_map[p.second];
        new_r_so_map[new_map[p.second]] = p.first;
    }
    so_map.clear();
    r_so_map.clear();
    so_map = new_so_map;
    r_so_map = new_r_so_map;
}

void reassign_dict_sort_degree(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                               DICTTYPE &so_map, REVERSE_DICTTYPE &r_so_map) {
    size_t triple_size = std::max(s.size(), o.size());
    std::unordered_map<TYPEID, TYPEID> new_map;

    LinkedData linkedData(triple_size);
    for (size_t i = 0; i < triple_size; ++i) linkedData.addLink(s[i], o[i], p[i]);
}

void reassign_dict_with_dim_reduction(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
    DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {

    std::vector<std::set<TYPEID>> iri_vec(so_map.size());
    ofstream out("./tools/tmp-so-vec.txt");
    for (size_t i = 0; i < s.size(); ++i) {
        iri_vec[ s[i] - 1 ].insert(p[i] - 1);
        if (o[i] < LITERAL_START_ID) iri_vec[ o[i] - 1 ].insert(p[i] - 1);
    }
    out << so_map.size() << " " << p_map.size() << "\n";
    for (size_t i = 0; i < iri_vec.size(); ++i) {
        for (auto e : iri_vec[i]) out << e << " ";
        out << "\n";
    }
    out.close();

    int ret = system("python3 ./tools/pca.py");
    std::cout << ret << "\n";

    size_t old_id, new_id;
    std::unordered_map<TYPEID, TYPEID> new_map;
    ifstream in("./tools/reassigned-so-id.txt");
    while (in >> old_id >> new_id) new_map[old_id] = new_id;
    in.close();

    // Update s, o
    size_t triple_size = s.size();
    for (size_t i = 0; i < triple_size; ++i) {
        s[i] = new_map[s[i]];
        o[i] = new_map[o[i]];
    }

    // Update so dict
    DICTTYPE new_so_map;
    REVERSE_DICTTYPE new_r_so_map;
    for (auto &p : so_map) {
        new_so_map[p.first] = new_map[p.second];
        new_r_so_map[new_map[p.second]] = p.first;
    }
    so_map.clear(); so_map = new_so_map;
    r_so_map.clear(); r_so_map = new_r_so_map;
}

// Generate a random integer between min_v and max_v
unsigned random(unsigned min_v, unsigned max_v) {
    return std::rand() % (max_v - min_v + 1) + min_v;
}

void reassign_spread(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o,
                     DICTTYPE &so_map, DICTTYPE &p_map,
                     REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    size_t i;
    const unsigned MIN_WIDTH = 2;
    const unsigned MAX_WIDTH = 10;
    size_t so_data_size = so_map.size();
    size_t p_data_size = p_map.size();
    std::unordered_map<TYPEID, TYPEID> so_new_assign, p_new_assign;
    
    std::srand(100);

    r_so_map.clear();
    for (auto &p: so_map) {
        unsigned width = random(MIN_WIDTH, MAX_WIDTH);
        TYPEID old_value = so_map[p.first];
        so_map[p.first] = p.second * width;
        r_so_map[p.second * width] = p.first;
        so_new_assign[old_value] = p.second * width;
    }
    
    r_p_map.clear();
    for (auto &p: p_map) {
        unsigned width = random(MIN_WIDTH, MAX_WIDTH);
        TYPEID old_value = p_map[p.first];
        p_map[p.first] = p.second * width;
        r_p_map[p.second * width] = p.first;
        p_new_assign[old_value] = p.second * width;
    }
    
    for (i = 0; i < s.size(); ++i) s[i] = so_new_assign[s[i]];
    for (i = 0; i < p.size(); ++i) p[i] = p_new_assign[p[i]];
    for (i = 0; i < o.size(); ++i) o[i] = so_new_assign[o[i]];
}

bool is_one_vector(uint8_t term_vec[], size_t vectorSize) {
    return std::all_of(term_vec, term_vec + vectorSize, [](int e){ return e == 1; });
}

unsigned int vector_sum(uint8_t term_vec[], size_t vectorSize) {
    return std::accumulate(term_vec, term_vec + vectorSize, 0);
}

void sort_triple_by_pso(TYPEID_HOST_VEC &s, TYPEID_HOST_VEC &p, TYPEID_HOST_VEC &o) {
    TYPEID_DEVICE_VEC d_s = s, d_p = p, d_o = o;

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_p.begin(), d_s.begin(), d_o.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_p.end(), d_s.end(), d_o.end()));
    thrust::sort(zip_begin, zip_end);

    s = d_s, p = d_p, o = d_o;
}
