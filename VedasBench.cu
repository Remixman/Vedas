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

void load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map, 
    REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map);
    
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <database_name> <testsuite_name> <path_to_queries>\n";
        return -1;
    }
    
    InputParser input_parser(argc, argv);
    bool preload = input_parser.cmdOptionExists("-preload");

    std::string device_num = input_parser.getCmdOption("-d");
    if (device_num != "") {
        int max_device;
        cudaGetDeviceCount(&max_device);
        int device_id = std::stoi(device_num);
        if (device_id < max_device) cudaSetDevice(device_id);
    }

    int device;
    cudaGetDevice(&device);
    std::cout << "Device ID : " << device << "\n";
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    std::cout << "Clock rate : " << deviceProp.clockRate << " , Memory clock rate : "
              << deviceProp.memoryClockRate << "\n";

    std::map<std::string, std::vector<std::string>> testsuite_query_dict;
    testsuite_query_dict["watdiv"] = { 
      "C1.txt", "C2.txt", "C3.txt",
      "F1.txt", "F2.txt", "F3.txt", "F4.txt", "F5.txt",
      "S1.txt", "S2.txt", "S3.txt", "S4.txt", "S5.txt", "S6.txt", "S7.txt",
      "L1.txt", "L2.txt", "L3.txt", "L4.txt", "L5.txt" };
    int test_time = 20, select_time = 5;
    
    int poolSize = 2; // TODO: set thread pool size fron argument
    ctpl::thread_pool threadPool(poolSize);
    
    standard_context_t context;
    vector<TYPEID> subjects, predicates, objects;
    DICTTYPE so_map, p_map, l_map;
    REVERSE_DICTTYPE r_so_map, r_p_map, r_l_map;

    std::string dict_path = argv[1]; dict_path += ".vdd";
    std::string data_path = argv[1]; data_path += ".vds";
    std::string test_suite = argv[2];
    std::string query_path = argv[3];

    std::ofstream elapse_time_file;
    std::time_t unixtime = std::time(nullptr);
    std::string log_file_name = "elapse_time_" + test_suite + std::to_string(unixtime) + ".log";
    elapse_time_file.open(log_file_name.c_str(), std::ofstream::out);

    auto load_start = std::chrono::high_resolution_clock::now();
    load_dict(dict_path.c_str(), so_map, p_map, l_map, r_so_map, r_p_map, r_l_map);
    QueryExecutor::r_p_map = &r_p_map;
    QueryExecutor::r_so_map = &r_so_map;
    QueryExecutor::r_l_map = &r_l_map;
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

    if (test_suite == "watdiv") {
        for (auto queryFileName : testsuite_query_dict[test_suite]) {
            elapse_time_file << queryFileName << " : ";
            // std::cout << "========================================\n";
            // std::cout << "           " << queryFileName << "\n";
            // std::cout << "========================================\n";
            auto queryFullPath = query_path + "/" + queryFileName;
            fstream sin(queryFullPath.c_str(), std::fstream::in);
            string sparql_query((std::istreambuf_iterator<char>(sin)), std::istreambuf_iterator<char>());

            int plan_id = 0;
            if (queryFileName == "C1.txt") { plan_id = 13; }
            else if (queryFileName == "C2.txt") { plan_id = 14; }
            else if (queryFileName == "C3.txt") { plan_id = 15; }
            else if (queryFileName == "F1.txt") { plan_id = 16; }
            else if (queryFileName == "F2.txt") { plan_id = 17; }
            else if (queryFileName == "F3.txt") { plan_id = 18; }
            else if (queryFileName == "F4.txt") { plan_id = 19; }
            else if (queryFileName == "F5.txt") { plan_id = 20; }
            else if (queryFileName == "S1.txt") { plan_id = 1; }
            else if (queryFileName == "S2.txt") { plan_id = 2; }
            else if (queryFileName == "S3.txt") { plan_id = 3; }
            else if (queryFileName == "S4.txt") { plan_id = 4; }
            else if (queryFileName == "S5.txt") { plan_id = 5; }
            else if (queryFileName == "S6.txt") { plan_id = 6; }
            else if (queryFileName == "S7.txt") { plan_id = 7; }
            else if (queryFileName == "L1.txt") { plan_id = 8; }
            else if (queryFileName == "L2.txt") { plan_id = 9; }
            else if (queryFileName == "L3.txt") { plan_id = 10; }
            else if (queryFileName == "L4.txt") { plan_id = 11; }
            else if (queryFileName == "L5.txt") { plan_id = 12; }
            else if (queryFileName == "LINEAR1.txt") { plan_id = 99; }

            vector<double> timeNanos(test_time);
            int resultSize = -1;
            for (int t = 0; t < test_time; ++t) {
                SparqlQuery query(sparql_query.c_str(), so_map, p_map);

                auto query_start = std::chrono::high_resolution_clock::now();
                SparqlResult result;
                QueryExecutor::initTime();
                QueryExecutor executor(vedasStorage, &threadPool, false, &context, plan_id);
                executor.query(query, result);
                auto query_end = std::chrono::high_resolution_clock::now();
                timeNanos[t] = std::chrono::duration_cast<std::chrono::nanoseconds>(query_end-query_start).count();
                resultSize = result.getResultIR()->size();
            }

            std::sort(timeNanos.begin(), timeNanos.end());
            double avgTime = std::accumulate(timeNanos.begin(), timeNanos.begin() + select_time, 0.0) / (1.0 * select_time);
            std::cout << queryFileName << " : " << std::setprecision(9) << avgTime/1e6
                      << " ms. (" << std::setprecision(9) << avgTime << " ns.)\n";

            elapse_time_file << " " << avgTime/1e6 << " ms. | " << resultSize << " rows \n";
        }
    } else {
        cout << "Unknown testsuite\n";
    }

    elapse_time_file.close();

    return 0;
}

void load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
                REVERSE_DICTTYPE &r_so_map, REVERSE_DICTTYPE &r_p_map, REVERSE_DICTTYPE &r_l_map) {
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
}
