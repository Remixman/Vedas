#ifndef SEGMENT_TREE_H
#define SEGMENT_TREE_H

#include <vector>

class SegmentTree {
public:
    SegmentTree(const std::vector<int> &_A);
    int rmq(int i, int j);
    int rmqValue(int i, int j);
    int value(int p);

private:
    int n;
    std::vector<int> A, st;

    int left(int p);
    int right(int p);
    void build(int p,int l,int r);
    int rmq(int p, int l, int r, int i, int j);
};

#endif
