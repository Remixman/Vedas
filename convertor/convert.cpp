#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cstdlib>
#include <cstdio>

using namespace std;

typedef unsigned TYPEID;

int initSPO(TYPEID **subjects, TYPEID **predicates, TYPEID **objects);

int main(int argc, char **argv) {
  map<TYPEID, TYPEID> soMap;

  if (argc <= 1) {
    cerr << "Usage : convert .tid path\n";
  }

  ofstream newOut;
  newOut.open("out.index-tid", ios::out);
  FILE * triple_f = fopen(argv[1], "rb");
  if (!triple_f) {
    newOut.close();
    cerr << "Cannot open TripleID file\n";
    return 0;
  }

  size_t chunk_size = 4096, count;
  TYPEID *data_b = (TYPEID *) malloc(sizeof(TYPEID ) * chunk_size * 3);
  while (!feof(triple_f)) {
    count = fread(data_b, sizeof(TYPEID), chunk_size * 3, triple_f);
    for (int k = 0; k < count; k += 3) {
      TYPEID subject = data_b[k + 0];
      TYPEID predicate = data_b[k + 1];
      TYPEID object = data_b[k + 2];

      /*if (soMap.count(subject) == 0) soMap[subject] = soMap.size();
      if (soMap.count(object) == 0) soMap[object] = soMap.size();

      subject = soMap[subject];
      object = soMap[object];*/

      newOut << subject << " " << predicate << " " << object << "\n";
    }
  }

  cout << "Subject and Object unique number is : " << soMap.size() << "\n";

  newOut.close();
  free(data_b);
  fclose(triple_f);

  return 0;
}