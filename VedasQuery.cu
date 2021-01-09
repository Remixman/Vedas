#include <iostream>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iomanip>
#include <vector>
#include <string>
#include <ctime>
#include <cassert>
#include <chrono>
#include <thrust/copy.h>
#include <moderngpu/context.hxx>
#include <raptor2/raptor2.h>
#include <rasqal/rasqal.h>
#include "ctpl_stl.h"
#include "vedas.h"
#include "InputParser.h"
#include "TriplePattern.h"
#include "SparqlQuery.h"
#include "RdfData.h"
#include "VedasStorage.h"
#include "QueryExecutor.h"

using namespace mgpu;
using namespace std;

void load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <Database Name>\n";
        return -1;
    }
    
    InputParser input_parser(argc, argv);
    bool preload = input_parser.cmdOptionExists("-preload");
    bool parallel_sche = input_parser.cmdOptionExists("-psche");

    bool device1 = input_parser.cmdOptionExists("-d1");
    if (device1) cudaSetDevice(1);

    int device;
    cudaGetDevice(&device);
    std::cout << "Device ID : " << device << "\n";
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    std::cout << "Clock rate : " << deviceProp.clockRate << " , Memory clock rate : "
              << deviceProp.memoryClockRate << "\n";

    
    
    int poolSize = 2; // TODO: set thread pool size fron argument
    ctpl::thread_pool threadPool(poolSize);
    
#if 0
    if (input.cmdOptionExists("-h")) { }
    std::string &storage_type = input.getCmdOption("-storage");
    if (storage_type.empty()) {
        storage_type = "cpu";
    }
#endif
    
    standard_context_t context;
    vector<TYPEID> subjects, predicates, objects;
    DICTTYPE so_map, p_map;
    REVERSE_DICTTYPE r_so_map, r_p_map;

    std::string dict_path = argv[1]; dict_path += ".vdd";
    std::string data_path = argv[1]; data_path += ".vds";
    // load_dummy_rdf(subjects, predicates, objects, so_map, p_map, r_so_map, r_p_map);

    auto load_start = std::chrono::high_resolution_clock::now();
    load_dict(dict_path.c_str(), so_map, p_map, r_so_map, r_p_map);
    QueryExecutor::r_p_map = &r_p_map;
    QueryExecutor::r_so_map = &r_so_map;
    auto load_end = std::chrono::high_resolution_clock::now();

    auto indexing_start = std::chrono::high_resolution_clock::now();
    VedasStorage *vedasStorage = new VedasStorage(data_path.c_str(), preload);
    auto indexing_end = std::chrono::high_resolution_clock::now();

    std::cout << "Load Database [" << argv[1] << "]\n";
    cout << "Triple Size          : " << vedasStorage->getTripleSize() << "\n";
    cout << "Subject Index Size   : " << vedasStorage->getSubjectIndexSize() << "\n";
    cout << "Predicate Index Size : " << vedasStorage->getPredicateIndexSize() << "\n";
    cout << "Object Index Size    : " << vedasStorage->getObjectIndexSize() << "\n";
    cout << "Loading time   : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(load_end-load_start).count() << " ms.\n";
    cout << "Indexing time  : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(indexing_end-indexing_start).count() << " ms.\n";
    cout << "\n\n";

    std::string cmd_line, op, arg, sparql_query;
    while (true) {
        cout << "vedas> "; getline(cin, cmd_line);
        // cmd_line = "sparql /work/rm/query/S1.txt";
        stringstream cmd_ss(cmd_line);
        cmd_ss >> op;
        arg = (cmd_line.size() > op.size())? cmd_line.substr(op.size() + 1) : "";

        int plan_id = 0;
        if (op == "sparql") {
            // Load files
            fstream sin(arg.c_str(), std::fstream::in);
            std::string query_str((std::istreambuf_iterator<char>(sin)), std::istreambuf_iterator<char>());
            sparql_query = query_str;
            if (arg.find("C1.txt") != std::string::npos) { plan_id = 13; }
            else if (arg.find("C2.txt") != std::string::npos) { plan_id = 14; }
            else if (arg.find("C3.txt") != std::string::npos) { plan_id = 15; }
            else if (arg.find("F1.txt") != std::string::npos) { plan_id = 16; }
            else if (arg.find("F2.txt") != std::string::npos) { plan_id = 17; }
            else if (arg.find("F3.txt") != std::string::npos) { plan_id = 18; }
            else if (arg.find("F4.txt") != std::string::npos) { plan_id = 19; }
            else if (arg.find("F5.txt") != std::string::npos) { plan_id = 20; }
            else if (arg.find("S1.txt") != std::string::npos) { plan_id = 1; }
            else if (arg.find("S2.txt") != std::string::npos) { plan_id = 2; }
            else if (arg.find("S3.txt") != std::string::npos) { plan_id = 3; }
            else if (arg.find("S4.txt") != std::string::npos) { plan_id = 4; }
            else if (arg.find("S5.txt") != std::string::npos) { plan_id = 5; }
            else if (arg.find("S6.txt") != std::string::npos) { plan_id = 6; }
            else if (arg.find("S7.txt") != std::string::npos) { plan_id = 7; }
            else if (arg.find("L1.txt") != std::string::npos) { plan_id = 8; }
            else if (arg.find("L2.txt") != std::string::npos) { plan_id = 9; }
            else if (arg.find("L3.txt") != std::string::npos) { plan_id = 10; }
            else if (arg.find("L4.txt") != std::string::npos) { plan_id = 11; }
            else if (arg.find("L5.txt") != std::string::npos) { plan_id = 12; }
            // DBpedia
            else if (arg.find("q1.txt") != std::string::npos) { plan_id = 21; }
            else if (arg.find("q2.txt") != std::string::npos) { plan_id = 22; }
            else if (arg.find("q3.txt") != std::string::npos) { plan_id = 23; }
            else if (arg.find("q4.txt") != std::string::npos) { plan_id = 24; }
            else if (arg.find("q5.txt") != std::string::npos) { plan_id = 25; }
            else if (arg.find("q6.txt") != std::string::npos) { plan_id = 26; }
            else if (arg.find("q7.txt") != std::string::npos) { plan_id = 27; }
            else if (arg.find("q8.txt") != std::string::npos) { plan_id = 28; }
            else if (arg.find("q9.txt") != std::string::npos) { plan_id = 29; }
            else {
                std::cout << "Cannot found pre-defined query plan\n";
                // assert(false);
            }
            sin.close();
        } else if (op == "exit") {
            break;
        } else {
            // SPARQL Query
        }

        cout << sparql_query << "\n";
        SparqlQuery query(sparql_query.c_str(), so_map, p_map);
        query.print();

        auto query_start = std::chrono::high_resolution_clock::now();
        SparqlResult result;
        QueryExecutor::upload_ms = 0.0;
        QueryExecutor::join_ns = 0.0;
        QueryExecutor::alloc_copy_ns = 0.0;
        QueryExecutor::swap_index_ns = 0.0;
        QueryExecutor::download_ns = 0.0;
        QueryExecutor::eliminate_duplicate_ns = 0.0;
        QueryExecutor executor(vedasStorage, &threadPool, parallel_sche, &context, plan_id);
        executor.query(query, result);
        auto query_end = std::chrono::high_resolution_clock::now();
        result.printResult(r_so_map, r_p_map);

        cout << "Query time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(query_end-query_start).count() << " ms. "
             << "(" << std::setprecision(9) << std::chrono::duration_cast<std::chrono::nanoseconds>(query_end-query_start).count() << " ns.)\n";
        cout << "Total upload time     : " << std::setprecision(3) << QueryExecutor::upload_ms << " ms.\n";
        cout << "Total join time       : " << std::setprecision(3) << QueryExecutor::join_ns / 1e6 << " ms. ("
             << std::setprecision(9) << QueryExecutor::join_ns << " ns.)\n";
        cout << "Total index swap time : " << std::setprecision(3) << QueryExecutor::swap_index_ns / 1e6 << " ms. ("
             << std::setprecision(9) << QueryExecutor::swap_index_ns << " ns.)\n";
        cout << "Total download time   : " << std::setprecision(3) << QueryExecutor::download_ns / 1e6 << " ms. ("
             << std::setprecision(9) << QueryExecutor::download_ns << " ns.)\n";
        // Device allocation and copy time
        cout << "Total alloc/copy time : " << std::setprecision(3) << QueryExecutor::alloc_copy_ns / 1e6 << " ms. ("
             << std::setprecision(9) << QueryExecutor::alloc_copy_ns << " ns.)\n";
        cout << "Total eliminate duplicate time : " << std::setprecision(3) << QueryExecutor::eliminate_duplicate_ns / 1e6 << " ms. ("
             << std::setprecision(9) << QueryExecutor::eliminate_duplicate_ns << " ns.)\n";
        // break;
    }

    return 0;
}

void load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map) {
    std::ifstream in;
    in.open(fname, std::fstream::in);

    size_t s; string tmp;

    std::getline(in, tmp);
    s = std::stoul(tmp);
    so_map.reserve(s);
    r_so_map.reserve(s);
    for (size_t i = 0; i < s; ++i) {
        std::getline(in, tmp);
        size_t first_space = tmp.find(" ");
        TYPEID id = static_cast<TYPEID>(std::stoul(tmp.substr(0, first_space)));
        string resource = tmp.substr(first_space + 1);
        so_map[resource] = id;
        r_so_map[id] = resource;
    }

    std::getline(in, tmp);
    s = std::stoul(tmp);
    p_map.reserve(s);
    r_p_map.reserve(s);
    for (size_t i = 0; i < s; ++i) {
        std::getline(in, tmp);
        size_t first_space = tmp.find(" ");
        TYPEID id = static_cast<TYPEID>(std::stoul(tmp.substr(0, first_space)));
        string resource = tmp.substr(first_space + 1);
        p_map[resource] = id;
        r_p_map[id] = resource;
    }

    in.close();
}
