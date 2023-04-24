#include "DataMetaInfo.h"
#include <fstream>

DataMetaInfo::DataMetaInfo() {

}

void DataMetaInfo::readFrom(std::string &fname) {
    // std::ostream infile(fname.c_str(), std::ios::in);

    // subjectObjectHist.clear();
    // predicateHist.clear();
    // literalHist.clear();

    // // Histogram
    // size_t subjectObjectHistNum;
    // infile >> subjectObjectHistNum;
    // for (size_t i = 0; i < subjectObjectHistNum; ++i) {

    // }

    // infile.close();
}

void DataMetaInfo::writeTo(std::string &fname) {
    // std::ostream outf(fname.c_str(), std::ios::out);

    // outf.close();
}
