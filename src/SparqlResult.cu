#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include "vedas.h"
#include "FullRelationIR.h"
#include "SparqlResult.h"
#include "QueryExecutor.h"

using namespace std;

SparqlResult::SparqlResult()
{
    resultIR = nullptr;
}

void SparqlResult::setResult(IR *ir) {
    resultIR = dynamic_cast<FullRelationIR*>(ir);
    assert(resultIR != nullptr);
}

FullRelationIR *SparqlResult::getResultIR() {
    return resultIR;
}

vector<string> SparqlResult::getHeaderVariables()
{
    return this->header;
}

vector<vector<TYPEID>> SparqlResult::get()
{
    return this->results;
}

vector<TYPEID> SparqlResult::get(size_t i) {
    return this->results[i];
}

void SparqlResult::printResult(REVERSE_DICTTYPE &so_dict, REVERSE_DICTTYPE &p_dict, REVERSE_DICTTYPE &l_dict) const {
#ifdef DEBUG
    std::cout << "Result : \n";
    this->resultIR->print();
#endif

    bool compact_print = true;

    // TODO: calculate column width
    size_t column_num = this->resultIR->getColumnNum();
    size_t max_widths[column_num];

    size_t screen_size = 80;
    if (compact_print) {
        size_t width = (screen_size - (3 * column_num)) / column_num;
        fill(max_widths, max_widths + column_num, width);
    } else {
        fill(max_widths, max_widths + column_num, 0);
        for (size_t r = 0; r < this->results.size(); r++) {
            for (size_t i = 0; i < this->resultIR->getColumnNum(); ++i) {
                if (this->results[r][i] > max_widths[i])
                    max_widths[i] = this->results[r][i];
            }
        }
    }

    int col_width = 6;

    // Print header
    size_t sum_width = accumulate(max_widths, max_widths + column_num, 0);
    size_t table_width = sum_width + (column_num * 2) + 2;
    cout << string(table_width, '=') << "\n";
    for (size_t i = 0; i < this->resultIR->getColumnNum(); ++i) {
        cout << "| " << std::setw(col_width) << this->resultIR->getHeader(i) << " ";
    }
    cout << "|\n";
    for (size_t i = 0; i < this->resultIR->getColumnNum(); ++i) {
        cout << "| " << std::setw(col_width) << (this->resultIR->getIsPredicate(i)? "pred" : "-") << " ";
    }
    cout << "|\n";

    auto download_start = std::chrono::high_resolution_clock::now();
    std::vector<TYPEID_HOST_VEC> relations(resultIR->getColumnNum());
    for (size_t i = 0; i < resultIR->getColumnNum(); ++i) {
        relations[i].resize(resultIR->getRelationSize(i));
        thrust::copy(resultIR->getRelation(i)->begin(), resultIR->getRelation(i)->end(), relations[i].begin());
    }
    auto download_end = std::chrono::high_resolution_clock::now();
    QueryExecutor::download_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(download_end-download_start).count();

    int RESULT_LIMIT = 5; // 15
    cout << string(table_width, '=') << "\n";
    for (size_t r = 0; r < resultIR->getRelationSize(0); r++) {
        for (size_t i = 0; i < this->resultIR->getColumnNum(); ++i) {
            TYPEID data = relations[i][r];
            auto convert_start = std::chrono::high_resolution_clock::now();

            string dataStr = "";
            if (QueryExecutor::ENABLE_LITERAL_DICT) {
                if (resultIR->getIsPredicate(i)) dataStr = p_dict[data];
                else dataStr = (data >= LITERAL_START_ID)? l_dict[data] : so_dict[data];
            } else {
                dataStr = resultIR->getIsPredicate(i)? p_dict[data] : so_dict[data];
            }
            auto convert_end = std::chrono::high_resolution_clock::now();
            QueryExecutor::convert_to_iri_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(convert_end-convert_start).count();

            //if (data.size() > max_widths[i]) data = data
            cout << "| " << std::setw(col_width) << dataStr << " ";
        }
        cout << "|\n";

        if (r > RESULT_LIMIT) {
            if (resultIR->getRelationSize(0) > RESULT_LIMIT) {
                std::cout << "....\n";
            }
            break;
        }
    }
    cout << string(table_width, '=') << "\n";
    std::cout << "Total : " << resultIR->getRelationSize(0) << " rows\n";
}
