#include "SegmentTree.h"

SegmentTree::SegmentTree(const std::vector<int> &_A) {
    A=_A;
    n = (int)(A.size());
    st.assign(4*n,0);
    build(1,0,n-1);
}

int SegmentTree::rmq(int i, int j) {
    return this->rmq(1, 0, n-1, i, j);
}

int SegmentTree::rmqValue(int i, int j) {
    return A[this->rmq(i, j)];
}

int SegmentTree::value(int p) {
    return A[p];
}

int SegmentTree::left(int p) { return (p<<1); }

int SegmentTree::right(int p) { return (p<<1)+1; }

void SegmentTree::build(int p,int l,int r) {
    if (l==r) st[p]=l;
    else {
        build(left(p),l,(l+r)/2);
        build(right(p),((l+r)/2)+1,r);
        int li = st[left(p)], ri = st[right(p)];
        st[p] = (A[li]<A[ri]) ? li : ri;
    }
}

int SegmentTree::rmq(int p, int l, int r, int i, int j) {
    if (i>r || j<l) return -1;
    else if(l>=i && r<=j) return st[p];

    int li = this->rmq(left(p),l,(l+r)/2,i,j);
    int ri = this->rmq(right(p),((l+r)/2)+1,r,i,j);

    if (li==-1) return ri;
    else if(ri==-1) return li;
    return (A[li]<A[ri]) ? li : ri;
}
