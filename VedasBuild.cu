#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <unordered_map>
#include "vedas.h"
#include "LinkedData.h"
#include "RdfData.h"
#include "VedasStorage.h"

using namespace std;

void save_dict(const char *fname, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);
void shuffle_dict(vector<TYPEID> &s, vector<TYPEID> &p, vector<TYPEID> &o,
                  DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);
void reassign_dict(vector<TYPEID> &s, vector<TYPEID> &p, vector<TYPEID> &o,
                   DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Usage : " << argv[0] << " <Database Name> <N-Triples file path>\n";
        return -1;
    }

    vector<TYPEID> subjects, predicates, objects;
    DICTTYPE so_map, p_map;
    REVERSE_DICTTYPE r_so_map, r_p_map;

    auto load_start = std::chrono::high_resolution_clock::now();
    load_rdf2(argv[2], subjects, predicates, objects, so_map, p_map, r_so_map, r_p_map);
    auto load_end = std::chrono::high_resolution_clock::now();

    /*auto shuffle_start = std::chrono::high_resolution_clock::now();
    shuffle_dict(subjects, predicates, objects, so_map, p_map, r_so_map, r_p_map);
    auto end_start = std::chrono::high_resolution_clock::now();*/

    reassign_dict(subjects, predicates, objects, so_map, p_map, r_so_map, r_p_map);

    std::string dict_path = argv[1]; dict_path += ".vdd";
    std::string data_path = argv[1]; data_path += ".vds";

    save_dict(dict_path.c_str(), r_so_map, r_p_map);

    auto copy_start = std::chrono::high_resolution_clock::now();
    RdfData rdfData(subjects, predicates, objects);
    auto copy_end = std::chrono::high_resolution_clock::now();

    auto indexing_start = std::chrono::high_resolution_clock::now();
    VedasStorage *vedasStorage = new VedasStorage(rdfData, false);
    vedasStorage->write(data_path.c_str());
    auto indexing_end = std::chrono::high_resolution_clock::now();

    /* Save graph file for visualize */
    ofstream node_out("./tools/nodes.txt");
    for (TYPEID node = 1; node <= subjects.size(); ++node) node_out << node << "\n";
    node_out.close();
    ofstream edge_out("./tools/edges.txt");
    for (TYPEID i = 1; i <= subjects.size(); ++i) edge_out << subjects[i] << " " << objects[i] << "\n";
    edge_out.close();

    std::cout << "Build Finished!\n";
    cout << "Loading time   : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(load_end-load_start).count() << " ms.\n";
    cout << "Copy time      : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(copy_end-copy_start).count() << " ms.\n";
    cout << "Indexing time  : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(indexing_end-indexing_start).count() << " ms.\n";
    //cout << "Shuffling time : " << std::setw(8) << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(end_start-shuffle_start).count() << " ms.\n";

    return 0;
}

void save_dict(const char *fname, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    std::ofstream out;
    out.open(fname, std::fstream::out);

    out << r_so_map.size() << "\n";
    for (auto it = r_so_map.begin(); it != r_so_map.end(); ++it)
        out << it->first << " " << it->second << "\n";
    out << r_p_map.size() << "\n";
    for (auto it = r_p_map.begin(); it != r_p_map.end(); ++it)
        out << it->first << " " << it->second << "\n";

    out.close();
}

void shuffle_dict(vector<TYPEID> &s, vector<TYPEID> &p, vector<TYPEID> &o,
                  DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::map<TYPEID, TYPEID> so_swap_dict, p_swap_dict;
    size_t so_data_size = so_map.size();
    size_t p_data_size = p_map.size();

    std::vector<TYPEID> so_ids;
    std::vector<std::string> so_data;
    for (auto &p: so_map) {
        so_data.push_back(p.first);
        so_ids.push_back(p.second);
    }
    std::shuffle(so_ids.begin(), so_ids.end(), std::default_random_engine(seed));
    size_t i = 0;
    for (auto &p: so_map) {
        so_swap_dict[p.second] = so_ids[i]; i++;
    }
    so_map.clear(); r_so_map.clear();
    for (i = 0; i < so_data_size; ++i) {
        so_map[so_data[i]] = so_ids[i];
        r_so_map[so_ids[i]] = so_data[i];
    }

    std::vector<TYPEID> p_ids;
    std::vector<std::string> p_data;
    for (auto &p: p_map) {
        p_data.push_back(p.first);
        p_ids.push_back(p.second);
    }
    std::shuffle(p_ids.begin(), p_ids.end(), std::default_random_engine(seed));
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

void reassign_dict(vector<TYPEID> &s, vector<TYPEID> &p, vector<TYPEID> &o,
                   DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    size_t triple_size = s.size();
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
