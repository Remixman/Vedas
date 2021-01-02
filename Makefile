GREEN='\033[0;32m'
NC='\033[0m' # No Color

MGPU_FLAGS = -g -DVERBOSE_DEBUG -DTIME_DEBUG -gencode arch=compute_30,code=compute_30 -I ../moderngpu/src --expt-extended-lambda -Wno-deprecated-declarations -lraptor2 -lrasqal
CUDPP_FLAGS = -L /home/remixman/cudpp/build/lib -lcudpp -lcudpp_hash
# OMP_ENABLE_FLAG = -Xcompiler " -fopenmp"

all: vdQuery vdBuild vdClean
	echo ${GREEN}===================== SUCCESS =====================${NC}

testPartition: TestPartition.cpp
	g++ -std=c++11 TestPartition.cpp -o testPartition -lmetis

vdBuild: FullRelationIR.o RdfData.o InputParser.o VedasStorage.o VedasBuild.cu LinkedData.o
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) *.o loader.cu -o vdBuild VedasBuild.cu 

vdQuery: TriplePattern.o QueryExecutor.o VedasStorage.o QueryIndex.o QueryPlan.o RdfData.o IndexIR.o FullRelationIR.o SelectQueryJob.o JoinQueryJob.o PlaningGraph.o InputParser.o IR.o VedasQuery.cu loader.cu
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) *.o loader.cu VedasQuery.cu -o vdQuery

vdClean: VedasClean.cu
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) VedasClean.cu -o vdClean

entail: join.cu
	nvcc -std=c++11 $(MGPU_FLAGS) $(CUDPP_FLAGS) -o join join.cu 

exp: exp.cu
	nvcc -std=c++11 $(MGPU_FLAGS) $(CUDPP_FLAGS) -o exp exp.cu 

SparqlQuery.o: SparqlQuery.h SparqlQuery.cu vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o SparqlQuery.o SparqlQuery.cu

SparqlResult.o: FullRelationIR.o SparqlResult.h SparqlResult.cu vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o SparqlResult.o SparqlResult.cu

TriplePattern.o: TriplePattern.h TriplePattern.cu vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o TriplePattern.o TriplePattern.cu

QueryExecutor.o: SparqlResult.o SparqlQuery.o QueryExecutor.h QueryExecutor.cu vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o QueryExecutor.o QueryExecutor.cu

QueryIndex.o: QueryIndex.h QueryIndex.cu vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o QueryIndex.o QueryIndex.cu

VedasStorage.o: VedasStorage.cu VedasStorage.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o VedasStorage.o VedasStorage.cu

RdfData.o: RdfData.cu RdfData.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o RdfData.o RdfData.cu

QueryPlan.o: SparqlResult.o JoinQueryJob.o QueryPlan.cu QueryPlan.h vedas.h 
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o QueryPlan.o QueryPlan.cu

IR.o: IR.cu IR.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o IR.o IR.cu

IndexIR.o: IR.o IndexIR.cu IndexIR.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o IndexIR.o IndexIR.cu

FullRelationIR.o: IR.o FullRelationIR.cu FullRelationIR.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o FullRelationIR.o FullRelationIR.cu

QueryJob.o: QueryJob.cu QueryJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o QueryJob.o QueryJob.cu

SelectQueryJob.o: QueryJob.o QueryExecutor.o SelectQueryJob.cu SelectQueryJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o SelectQueryJob.o SelectQueryJob.cu

JoinQueryJob.o: QueryJob.o JoinQueryJob.cu JoinQueryJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o JoinQueryJob.o JoinQueryJob.cu

PlaningGraph.o: PlaningGraph.cu PlaningGraph.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o PlaningGraph.o PlaningGraph.cu
	
InputParser.o: InputParser.cu InputParser.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o InputParser.o InputParser.cu
	
LinkedData.o: LinkedData.cu LinkedData.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o LinkedData.o LinkedData.cu

clean:
	rm -rf join vdBuild vdQuery vdClean *.o
