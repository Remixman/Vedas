GREEN='\033[0;32m'
NC='\033[0m' # No Color
SMS ?= 30
# SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))

MGPU_FLAGS = -g -DVERBOSE_DEBUG -DTIME_DEBUG -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM) -I ../moderngpu/src --expt-extended-lambda -Wno-deprecated-declarations -lraptor2 -lrasqal
CUDPP_FLAGS = -L /home/remixman/cudpp/build/lib -lcudpp -lcudpp_hash
# OMP_ENABLE_FLAG = -Xcompiler " -fopenmp"

all: vdQuery vdBuild vdClean
	echo ${GREEN}===================== SUCCESS =====================${NC}

testPartition: TestPartition.cpp
	g++ -std=c++11 TestPartition.cpp -o testPartition -lmetis

vdBuild: FullRelationIR.o RdfData.o InputParser.o VedasStorage.o ExecutionWorker.o LinkedData.o VedasBuild.cu
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) *.o loader.cu -o vdBuild VedasBuild.cu 

vdQuery: TriplePattern.o QueryExecutor.o VedasStorage.o QueryIndex.o QueryPlan.o RdfData.o IndexIR.o FullRelationIR.o SwapIndexJob.o SelectQueryJob.o JoinQueryJob.o InputParser.o IR.o ExecutionPlanTree.o ExecutionWorker.o VariableNode.o IrNode.o TriplePatternNode.o VedasQuery.cu loader.cu
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

QueryExecutor.o: SparqlResult.o SparqlQuery.o JoinGraph.o QueryExecutor.h QueryExecutor.cu vedas.h
	nvcc -std=c++11 -I ./ $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o QueryExecutor.o QueryExecutor.cu

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

ExecutionPlanTree.o: ExecutionPlanTree.cu ExecutionPlanTree.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o ExecutionPlanTree.o ExecutionPlanTree.cu

QueryJob.o: QueryJob.cu QueryJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o QueryJob.o QueryJob.cu

SelectQueryJob.o: QueryJob.o QueryExecutor.o SelectQueryJob.cu SelectQueryJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o SelectQueryJob.o SelectQueryJob.cu

JoinQueryJob.o: QueryJob.o JoinQueryJob.cu JoinQueryJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o JoinQueryJob.o JoinQueryJob.cu

SwapIndexJob.o: QueryJob.o SwapIndexJob.cu SwapIndexJob.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o SwapIndexJob.o SwapIndexJob.cu
	
InputParser.o: InputParser.cu InputParser.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o InputParser.o InputParser.cu
	
LinkedData.o: LinkedData.cu LinkedData.h vedas.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o LinkedData.o LinkedData.cu

ExecutionWorker.o: ExecutionWorker.cu ExecutionWorker.h vedas.h
	nvcc -std=c++11 -Xcompiler="-pthread" $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o ExecutionWorker.o ExecutionWorker.cu

JoinGraph.o: JoinGraph/JoinGraph.cu JoinGraph/JoinGraph.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o JoinGraph.o JoinGraph/JoinGraph.cu

VariableNode.o: JoinGraph/VariableNode.cu JoinGraph/VariableNode.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o VariableNode.o JoinGraph/VariableNode.cu

IrNode.o: JoinGraph/IrNode.cu JoinGraph/IrNode.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o IrNode.o JoinGraph/IrNode.cu

TriplePatternNode.o: JoinGraph/TriplePatternNode.cu JoinGraph/TriplePatternNode.h
	nvcc -std=c++11 $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) -c -o TriplePatternNode.o JoinGraph/TriplePatternNode.cu

clean:
	rm -rf join vdBuild vdQuery vdClean *.o
