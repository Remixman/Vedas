#include <iostream>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iomanip>
#include <future>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <cassert>
#include <chrono>
#include <vector>
#include <thrust/copy.h>
#include <moderngpu/context.hxx>
#include <raptor2/raptor2.h>
#include <rasqal/rasqal.h>
// #include <cuda_profiler_api.h>
#include "vedas.h"
#include "ExecutionWorker.h"
#include "InputParser.h"
#include "TriplePattern.h"
#include "SparqlQuery.h"
#include "RdfData.h"
#include "VedasStorage.h"
#include "QueryExecutor.h"

using namespace mgpu;
using namespace std;

int load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
               REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map);
void print_ms_ns_time(const std::string& label, double ns_time);
void print_exec_log(std::vector<ExecuteLogRecord> &records);

int main(int argc, char **argv) {
    QueryExecutor::ENABLE_UPDATE_BOUND_AFTER_JOIN = true;

    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <Database Name>\n";
        return -1;
    }

    std::vector<int> gpu_ids;
    InputParser input_parser(argc, argv);
    bool preload = input_parser.cmdOptionExists("-preload");
    std::string sparql_path = input_parser.getCmdOption("-sparql-path");
    bool fullIndex = input_parser.cmdOptionExists("-full-index");
    bool literalDict = input_parser.cmdOptionExists("-ldict");
    if (sparql_path != "") {
        std::cout << "SPARQL PATH : " << sparql_path << "\n";
    }
    bool disableBoundAfterJoin = input_parser.cmdOptionExists("-dis-bound-after-join");
    if (disableBoundAfterJoin) QueryExecutor::ENABLE_UPDATE_BOUND_AFTER_JOIN = false;

    int max_device;
    cudaGetDeviceCount(&max_device);

    std::string device_num = input_parser.getCmdOption("-d"); // Example -d 0,1,2,3 or -d 2
    if (device_num != "") {
        string idStr;
        istringstream idstream(device_num);
        while (getline(idstream, idStr, ',')) {
            int device_id = std::stoi(idStr);
            if (device_id >= max_device) {
                std::cerr << "Invalid gpu id " << device_id << " of " << max_device << "\n";
            }
            gpu_ids.push_back(device_id);
        }
        if (gpu_ids.size() > 0 && gpu_ids[0] < max_device) {
            cudaSetDevice(gpu_ids[0]);
        }
    }
    if (gpu_ids.size() == 0) gpu_ids.push_back(0);

    int device;
    cudaGetDevice(&device);
    std::cout << "Device ID : " << device << "\n";
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    std::cout << "Clock rate : " << deviceProp.clockRate << " , Memory clock rate : "
              << deviceProp.memoryClockRate << "\n";

    ExecutionWorker worker(gpu_ids);

    standard_context_t context;
    vector<TYPEID> subjects, predicates, objects;
    DICTTYPE so_map, p_map, l_map;
    REVERSE_DICTTYPE r_so_map, r_p_map, r_l_map;

    std::string dict_path = argv[1]; dict_path += ".vdd";
    std::string data_path = argv[1]; data_path += ".vds";
    // load_test_rdf(subjects, predicates, objects, so_map, p_map, r_so_map, r_p_map);

    auto load_start = std::chrono::high_resolution_clock::now();
    auto load_dict_future = std::async(std::launch::async, load_dict, dict_path.c_str(),
                                       std::ref(so_map), std::ref(p_map), std::ref(l_map),
                                       std::ref(r_so_map), std::ref(r_p_map), std::ref(r_l_map));

    auto load_data_start = std::chrono::high_resolution_clock::now();
    VedasStorage *vedasStorage = new VedasStorage(data_path.c_str(), preload, fullIndex);
    auto load_data_end = std::chrono::high_resolution_clock::now();

    load_dict_future.get();
    QueryExecutor::r_p_map = &r_p_map;
    QueryExecutor::r_so_map = &r_so_map;
    QueryExecutor::r_l_map = &r_l_map;
    auto load_end = std::chrono::high_resolution_clock::now();

    QueryExecutor::ENABLE_FULL_INDEX = fullIndex;
    QueryExecutor::ENABLE_LITERAL_DICT = literalDict;
    // QueryExecutor::ENABLE_PREUPLOAD_BOUND_DICT = false;
    QueryExecutor::ENABLE_BOUND_DICT_AFTER_JOIN = false;

    // QueryExecutor::objectHistogram = new Histogram(
    //             "hist-object-equal-width.txt", "hist-object-equal-depth.txt");
    // QueryExecutor::subjectHistogram = new Histogram(
    //             "hist-subject-equal-width.txt", "hist-subject-equal-depth.txt");
    
    std::cout << "Load Database [" << argv[1] << "]\n";
    std::cout << "Triple Size          : " << vedasStorage->getTripleSize() << "\n";
    std::cout << "Subject Index Size   : " << vedasStorage->getSubjectIndexSize() << "\n";
    std::cout << "Predicate Index Size : " << vedasStorage->getPredicateIndexSize() << "\n";
    std::cout << "Object Index Size    : " << vedasStorage->getObjectIndexSize() << "\n";
    std::cout << "Dict Loading time    : " << std::setprecision(3) << QueryExecutor::load_dict_ms << " ms.\n";
    std::cout << "Data Loading time    : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(load_data_end-load_data_start).count() << " ms.\n";
    std::cout << "Dict/Data Loading time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(load_end-load_start).count() << " ms.\n";
    std::cout << "\n\n";

    double psBc = vedasStorage->psBoundaryCompactness();
    double poBc = vedasStorage->poBoundaryCompactness();
    std::cout << "PSO Boundary Compactness : " << std::setprecision(3) << log(psBc) << " (" << psBc << ")\n";
    std::cout << "POS Boundary Compactness : " << std::setprecision(3) << log(poBc) << " (" << poBc << ")\n";

    std::string cmd_line, op, arg, sparql_query;
    while (true) {
        
        try {
            if (sparql_path != "") {
                op = "sparql";
                arg = sparql_path;
            } else {
                std::cout << "vedas> "; getline(cin, cmd_line);
                stringstream cmd_ss(cmd_line);
                cmd_ss >> op;
                arg = (cmd_line.size() > op.size())? cmd_line.substr(op.size() + 1) : "";
            }

            if (op == "sparql") {
                // Load files
                fstream sin(arg.c_str(), std::fstream::in);
                if (!sin.good()) continue; // if file not exists
                
                std::string query_str((std::istreambuf_iterator<char>(sin)), std::istreambuf_iterator<char>());
                sparql_query = query_str;
                sin.close();
            } else if (op == "exit") {
                break;
            } else {
                std::cerr << "Not supported command\n";
                // SPARQL Query
                continue;
            }

            std::cout << sparql_query << "\n";
            SparqlQuery query(sparql_query.c_str(), so_map, p_map, l_map);
            query.print();

            auto query_start = std::chrono::high_resolution_clock::now();
            SparqlResult result;
            QueryExecutor::initTime();
            QueryExecutor executor(vedasStorage, &worker, &context);
            executor.setGpuIds(gpu_ids);
            executor.query(query, result);
            auto query_end = std::chrono::high_resolution_clock::now();
            result.printResult(r_so_map, r_p_map, r_l_map);

            cout << "Query time : " << std::setprecision(3) << std::chrono::duration_cast<std::chrono::milliseconds>(query_end-query_start).count() << " ms. "
                << "(" << std::setprecision(9) << std::chrono::duration_cast<std::chrono::nanoseconds>(query_end-query_start).count() << " ns.)\n";

            print_ms_ns_time("Total upload time     ", QueryExecutor::upload_ns);
            print_ms_ns_time("Total pre-scan time   ", QueryExecutor::prescan_extra_ns);
            print_ms_ns_time("Total join time       ", QueryExecutor::join_ns);
            print_ms_ns_time("Total index swap time ", QueryExecutor::swap_index_ns);
            print_ms_ns_time("Total download time   ", QueryExecutor::download_ns);
            // Device allocation and copy time
            print_ms_ns_time("Total alloc/copy time ", QueryExecutor::alloc_copy_ns);
            print_ms_ns_time("Total eliminate duplicate time", QueryExecutor::eliminate_duplicate_ns);
            print_ms_ns_time("Total scan and split time", QueryExecutor::scan_to_split_ns);
            print_ms_ns_time("Total EIF time        ", QueryExecutor::eif_ns);
            print_ms_ns_time("Total P2P transfer time ", QueryExecutor::p2p_transfer_ns);
            std::cout << '\n';
            
            std::cout <<      "# of updating empty interval : " << QueryExecutor::eif_count << '\n';

            print_ms_ns_time("Total convert to id time ", QueryExecutor::convert_to_id_ns);
            print_ms_ns_time("Total convert to IRI time", QueryExecutor::convert_to_iri_ns);

            print_exec_log(QueryExecutor::exe_log);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n'; 
        }

        if (sparql_path != "") break;
    }

    // delete QueryExecutor::objectHistogram;
    // delete QueryExecutor::subjectHistogram;

    return 0;
}

int load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
                REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map) {
    auto load_start = std::chrono::high_resolution_clock::now();

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

    if (std::getline(in, tmp)) {
        s = std::stoul(tmp);
        l_map.reserve(s);
        r_l_map.reserve(s);
        for (size_t i = 0; i < s; ++i) {
            std::getline(in, tmp);
            size_t first_space = tmp.find(" ");
            TYPEID id = static_cast<TYPEID>(std::stoul(tmp.substr(0, first_space)));
            string resource = tmp.substr(first_space + 1);
            l_map[resource] = id;
            r_l_map[id] = resource;
        }
    }

    in.close();

    auto load_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::load_dict_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end-load_start).count();

    return 0;
}

void print_ms_ns_time(const std::string& label, double ns_time) {
    std::cout << label << " : " << std::setprecision(3) << ns_time / 1e6 << " ms. ("
        << std::setprecision(8) << ns_time << " ns.)\n";
}

void print_exec_log(std::vector<ExecuteLogRecord> &records) {
    size_t total_upload = 0, total_join = 0, total_swap = 0;
    for (auto r : records) {
        std::cout << '(' << r.deviceId << ')';
        switch (r.op) {
            case JOIN_OP:
            {
                double selectivity = r.param3 * 1.0 / std::max(r.param1, r.param2);
                std::cout << "JOIN   [" << r.param1 << " x " << r.param2 << " : " << r.param3 << "] (" << r.paramstr << ")     "
                        << "Selectivity : " << selectivity << '\n';
                total_join += r.param1 + r.param2;
                break;
            }
            case UPLOAD_OP:
                std::cout << "UPLOAD [" << r.param1 << " x " << r.param2 << "] (" << r.paramstr << ")\n";
                total_upload += r.param1;
                break;
            case SWAP_OP:
                std::cout << "SWAP   [" << r.param1 << " x " << r.param2 << "] (" << r.paramstr << ")\n";
                total_swap += r.param1;
                break;
        }
    }
    std::cout << "*******************************\n";
    std::cout << "TOTAL JOIN       : " << total_join << "\n";
    std::cout << "TOTAL UPLOAD     : " << total_upload << "\n";
    std::cout << "TOTAL INDEX SWAP : " << total_swap << "\n";
    std::cout << "*******************************\n";
}
