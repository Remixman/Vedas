#ifndef DATAMETAINFO_H
#define DATAMETAINFO_H

#include <string>
#include <vector>

class DataMetaInfo
{
public:
    DataMetaInfo();
    void readFrom(std::string &fname);
    void writeTo(std::string &fname);

    std::vector<size_t> subjectObjectHist;
    std::vector<size_t> predicateHist;
    std::vector<size_t> literalHist;
};

#endif // DATAMETAINFO_H
