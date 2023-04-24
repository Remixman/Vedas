#ifndef INPUTPARSER_H
#define INPUTPARSER_H

#include <string>
#include <vector>

/// @author iain from https://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c
class InputParser{
public:
    InputParser (int &argc, char **argv);
    const std::string& getCmdOption(const std::string &option) const;
    bool cmdOptionExists(const std::string &option) const;
private:
    std::vector <std::string> tokens;
};

#endif