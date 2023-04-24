#include <metis.h>
// https://gist.github.com/erikzenker/c4dc42c8d5a8c1cd3e5a
#include <iostream>

int main() {
  
  idx_t nVertices = 6;
  idx_t nEdges    = 7;
  idx_t nCon      = 1;
  idx_t nParts    = 4;
  
  idx_t objval;
  idx_t part[nVertices];
  
  // Indexes of starting points in adjacent array
  idx_t xadj[nVertices+1] = {0,2,5,7,9,12,14};

  // Adjacent vertices in consecutive index order
  idx_t adjncy[2 * nEdges] = {1,3,0,4,2,1,5,0,4,3,1,5,4,2};
  
  // Weights of vertices
  // if all weights are equal then can be set to NULL
  idx_t vwgt[nVertices * nCon];
  
  int ret = METIS_PartGraphKway(
    &nVertices,       // The number of vertices
    &nCon,            // The number of balancing constraints
    xadj,
    adjncy,
    NULL, NULL, NULL, &nParts, NULL,
    NULL, NULL, &objval, part
  );
  
  switch (ret) {
    case METIS_OK:
      std::cout << "OK\n";
      break;
    case METIS_ERROR_INPUT:
    case METIS_ERROR_MEMORY:
    case METIS_ERROR:
      break;
  }
  
  for(unsigned part_i = 0; part_i < nVertices; part_i++){
	  std::cout << part_i << " " << part[part_i] << '\n';
   }
  
  return 0;
}