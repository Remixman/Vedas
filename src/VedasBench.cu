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
#include <thread>
#include <thrust/copy.h>
#include <moderngpu/context.hxx>
#include <raptor2/raptor2.h>
#include <rasqal/rasqal.h>
// #include "ctpl_stl.h"
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

int main(int argc, char **argv) {
    QueryExecutor::ENABLE_UPDATE_BOUND_AFTER_JOIN = true;

    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <database_name> <testsuite_name> <path_to_queries>\n";
        return -1;
    }

    std::vector<int> gpu_ids;
    InputParser input_parser(argc, argv);
    bool preload = input_parser.cmdOptionExists("-preload");
    bool showstat = input_parser.cmdOptionExists("-show-stat");
    bool fullIndex = input_parser.cmdOptionExists("-full-index");
    bool literalDict = input_parser.cmdOptionExists("-ldict");
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
        if (gpu_ids.size() > 0 && gpu_ids[0] < max_device) cudaSetDevice(gpu_ids[0]);
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
    testsuite_query_dict["lubm"] = {
      "lq1.txt", "lq2.txt", "lq4.txt", "lq7.txt" };
    int test_time = 20, warmup_time = 5, select_time = 5;
    if (showstat) {
        test_time = 1, warmup_time = 0, select_time = 1;
    }

    // ctpl::thread_pool threadPool(poolSize);
    ExecutionWorker worker(gpu_ids);

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

    auto load_dict_future = std::async(std::launch::async, load_dict, dict_path.c_str(),
                                       std::ref(so_map), std::ref(p_map), std::ref(l_map),
                                       std::ref(r_so_map), std::ref(r_p_map), std::ref(r_l_map));

    VedasStorage *vedasStorage = new VedasStorage(data_path.c_str(), preload, fullIndex);

    load_dict_future.get();
    QueryExecutor::r_p_map = &r_p_map;
    QueryExecutor::r_so_map = &r_so_map;
    QueryExecutor::r_l_map = &r_l_map;

    QueryExecutor::ENABLE_FULL_INDEX = fullIndex;
    QueryExecutor::ENABLE_LITERAL_DICT = literalDict;
    // QueryExecutor::ENABLE_PREUPLOAD_BOUND_DICT = false;
    QueryExecutor::ENABLE_BOUND_DICT_AFTER_JOIN = false;
    
    

    // QueryExecutor::objectHistogram = new Histogram(
    //             "hist-object-equal-width.txt", "hist-object-equal-depth.txt");
    // QueryExecutor::subjectHistogram = new Histogram(
    //             "hist-subject-equal-width.txt", "hist-subject-equal-depth.txt");

    std::cout << "Load Database [" << argv[1] << "]\n";
    std::cout << "Triple Size          : " << vedasStorage->getTripleSize() << '\n';
    std::cout << "Subject Index Size   : " << vedasStorage->getSubjectIndexSize() << '\n';
    std::cout << "Predicate Index Size : " << vedasStorage->getPredicateIndexSize() << '\n';
    std::cout << "Object Index Size    : " << vedasStorage->getObjectIndexSize() << '\n';
    std::cout << "\n\n";

    elapse_time_file << "DATA : " << argv[1] << '\n';
    elapse_time_file << "FULL INDEX : " << QueryExecutor::ENABLE_FULL_INDEX << '\n';

    if (test_suite == "watdiv" || test_suite == "lubm") {

        int testNum = testsuite_query_dict[test_suite].size();
        vector<vector<double>> queryTimesNano(testNum);
        vector<int> queryResultSizes(testNum);
        vector<size_t> queryUploadNum(testNum, 0), queryJoinNum(testNum, 0), querySwapNum(testNum, 0);
        for (int t = 0; t < warmup_time + test_time; ++t) {

            for (int q = 0; q < testsuite_query_dict[test_suite].size(); q++) {
                auto queryFileName = testsuite_query_dict[test_suite][q];
                auto queryFullPath = query_path + "/" + queryFileName;
                fstream sin(queryFullPath.c_str(), std::fstream::in);
                string sparql_query((std::istreambuf_iterator<char>(sin)), std::istreambuf_iterator<char>());

                SparqlQuery query(sparql_query.c_str(), so_map, p_map, l_map);
                auto query_start = std::chrono::high_resolution_clock::now();
                SparqlResult result;
                QueryExecutor::initTime();
                QueryExecutor executor(vedasStorage, &worker, &context);
                executor.query(query, result);
                auto query_end = std::chrono::high_resolution_clock::now();

                if (t >= warmup_time) {
                    queryTimesNano[q].push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(query_end-query_start).count());
                }
                if (t == 0) {
                    queryResultSizes[q] = result.getResultIR()->size();
                    for (auto r : QueryExecutor::exe_log) {
                        switch (r.op) {
                            case JOIN_OP:   queryJoinNum[q] += (r.param1 + r.param2); break;
                            case UPLOAD_OP: queryUploadNum[q] += r.param1; break;
                            case SWAP_OP:   querySwapNum[q] += r.param1; break;
                        }
                    }
                }
            }

            this_thread::sleep_for(chrono::milliseconds(2000));
        }

        for (int q = 0; q < testsuite_query_dict[test_suite].size(); q++) {
            auto queryFileName = testsuite_query_dict[test_suite][q];
            elapse_time_file << queryFileName << " : ";

            std::sort(queryTimesNano[q].begin(), queryTimesNano[q].end());
            // double avgTime = std::accumulate(queryTimesNano[q].begin(), queryTimesNano[q].begin() + select_time, 0.0) / (1.0 * select_time);
            double avgTime = queryTimesNano[q][0];
            std::cout << queryFileName << " : " << std::setprecision(9) << avgTime/1e6
                      << " ms. (" << std::setprecision(9) << avgTime << " ns.)\n";
            std::cout << "\tMIN : " << std::setprecision(2) << avgTime/1e6 << '\n';

            elapse_time_file << ' ' << avgTime/1e6 << " ms. | " << queryResultSizes[q] << " rows ";
            if (showstat) {
                elapse_time_file << "| U: " << queryUploadNum[q] << ' ';
                elapse_time_file << "| J: " << queryJoinNum[q] << ' ';
                elapse_time_file << "| S: " << querySwapNum[q];
            }
            elapse_time_file << "\n";
        }

        elapse_time_file << "Upload : " << std::accumulate(queryUploadNum.begin(), queryUploadNum.end(), 0) << "\n";
        elapse_time_file << "Join   : " << std::accumulate(queryJoinNum.begin(), queryJoinNum.end(), 0) << "\n";
        elapse_time_file << "Swap   : " << std::accumulate(querySwapNum.begin(), querySwapNum.end(), 0) << "\n";

    } else {
        cout << "Unknown testsuite\n";
    }

    elapse_time_file.close();

    return 0;
}

int load_dict(const char *fname, DICTTYPE &so_map, DICTTYPE &p_map, DICTTYPE &l_map,
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
    return 0;
}
