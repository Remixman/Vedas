#include "LinkedData.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <utility>

LinkedData::LinkedData(size_t n)
{
    this->n = n;
    links.resize(n + 1);
}

void LinkedData::addLink(TYPEID i, TYPEID j, TYPEID pred) {
    // XXX: we dont use index 0 (id is from 1 to n)
    links[i].push_back(std::make_pair(j, pred));
    links[j].push_back(std::make_pair(i, pred));
}

TYPEID current_id;
std::vector<bool> visited;
void recursiveAssign(TYPEID source, std::unordered_map<TYPEID, TYPEID> &reassign_map,
                     std::vector<std::vector<std::pair<TYPEID, TYPEID>>> &links) {

    std::queue<TYPEID> q;

    q.push(source); visited[source] = true; reassign_map[source] = current_id++;
    while (!q.empty()) {
        TYPEID node = q.front(); q.pop();
        auto &link_vec = links[node];
        for (size_t i = 0; i < link_vec.size(); ++i) {
            TYPEID p = link_vec[i].first;
            if (!visited[p]) {
                q.push(p); visited[p] = true; reassign_map[p] = current_id++;
            }
        }
    }
}

void LinkedData::reassignIdByBfs(std::unordered_map<TYPEID, TYPEID> &reassign_map) {
    // Search for most less degree node
    /*TYPEID source = 1; size_t min_degree = n + 1;
    for (size_t i = 1; i < links.size(); i++) {
        if (links[i].size() < min_degree) {
            min_degree = links[i].size();
            source = i;
        }
    }*/

    current_id = 1;
    visited.resize(n + 1, false);
    // std::cout << "Source is " << source << "\n";
    for (TYPEID node = 1; node < n + 1; ++node) {
        if (!visited[node]) recursiveAssign(node, reassign_map, links);
    }

    std::cout << "N is " << n << "\n";
    std::cout << "Last id is " << current_id << "\n";
}
