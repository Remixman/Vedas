GREEN='\033[0;32m'
NC='\033[0m' # No Color
SMS ?= 60
# SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
NT := 128
VT := 11
SORTEDSEARCH_VT := 15

DEBUG_LEVEL = #-g -G
EXTRA_CONFIGS = -DUPDATE_BOUND_AFTER_JOIN #-DLITERAL_DICT #-DAUTO_PLANNER -DVERBOSE_DEBUG -DTIME_DEBUG 
MGPU_FLAGS = ${DEBUG_LEVEL} $(EXTRA_CONFIGS) \
						-DNT=$(NT) -DVT=$(VT) -DSORTEDSEARCH_VT=$(SORTEDSEARCH_VT) \
						-std=c++11 \
						-gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM) \
						-I ./ --expt-extended-lambda -Wno-deprecated-declarations \
						-lraptor2 -lrasqal
CUDPP_FLAGS = -L /home/remixman/cudpp/build/lib -lcudpp -lcudpp_hash
# OMP_ENABLE_FLAG = -Xcompiler " -fopenmp"

all: vdQuery vdBuild vdBench vdClean
	echo ${GREEN}===================== SUCCESS =====================${NC}

modernGpuConfig:
	cp moderngpu-edits/kernel_load_balance.hxx moderngpu/kernel_load_balance.hxx
	cp moderngpu-edits/kernel_scan.hxx moderngpu/kernel_scan.hxx
	cp moderngpu-edits/kernel_sortedsearch.hxx moderngpu/kernel_sortedsearch.hxx

testPartition: TestPartition.cpp
	g++ -std=c++11 TestPartition.cpp -o testPartition -lmetis

vdBuild: FullRelationIR.o RdfData.o InputParser.o VedasStorage.o ExecutionWorker.o LinkedData.o VedasBuild.cu
	nvcc $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) *.o loader.cu -o vdBuild VedasBuild.cu

vdBench: TriplePattern.o QueryExecutor.o VedasStorage.o QueryIndex.o QueryPlan.o RdfData.o IndexIR.o FullRelationIR.o SwapIndexJob.o SelectQueryJob.o JoinQueryJob.o InputParser.o IR.o ExecutionPlanTree.o ExecutionWorker.o VariableNode.o IrNode.o TriplePatternNode.o VedasBench.cu loader.cu
	nvcc $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) *.o loader.cu -o vdBench VedasBench.cu

vdQuery: TriplePattern.o QueryExecutor.o VedasStorage.o QueryIndex.o QueryPlan.o RdfData.o IndexIR.o FullRelationIR.o SwapIndexJob.o SelectQueryJob.o JoinQueryJob.o InputParser.o IR.o ExecutionPlanTree.o ExecutionWorker.o VariableNode.o IrNode.o TriplePatternNode.o VedasQuery.cu loader.cu
	nvcc $(MGPU_FLAGS) $(OMP_ENABLE_FLAG) *.o loader.cu VedasQuery.cu -o vdQuery

vdClean: VedasClean.cu
	nvcc $(MGPU_FLAGS) VedasClean.cu -o vdClean

entail: join.cu
	nvcc $(MGPU_FLAGS) $(CUDPP_FLAGS) -o join join.cu 

exp: exp.cu
	nvcc $(MGPU_FLAGS) $(CUDPP_FLAGS) -o exp exp.cu 

SparqlQuery.o: SparqlQuery.h SparqlQuery.cu vedas.h
	nvcc $(MGPU_FLAGS) -c -o SparqlQuery.o SparqlQuery.cu

SparqlResult.o: FullRelationIR.o SparqlResult.h SparqlResult.cu vedas.h
	nvcc $(MGPU_FLAGS) -c -o SparqlResult.o SparqlResult.cu

TriplePattern.o: TriplePattern.h TriplePattern.cu vedas.h
	nvcc $(MGPU_FLAGS) -c -o TriplePattern.o TriplePattern.cu

QueryExecutor.o: SparqlResult.o SparqlQuery.o JoinGraph.o QueryExecutor.h QueryExecutor.cu vedas.h
	nvcc -I ./ $(MGPU_FLAGS) -c -o QueryExecutor.o QueryExecutor.cu

QueryIndex.o: QueryIndex.h QueryIndex.cu vedas.h
	nvcc $(MGPU_FLAGS) -c -o QueryIndex.o QueryIndex.cu

VedasStorage.o: VedasStorage.cu VedasStorage.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o VedasStorage.o VedasStorage.cu

RdfData.o: RdfData.cu RdfData.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o RdfData.o RdfData.cu

QueryPlan.o: SparqlResult.o JoinQueryJob.o QueryPlan.cu QueryPlan.h vedas.h 
	nvcc $(MGPU_FLAGS) -c -o QueryPlan.o QueryPlan.cu

IR.o: IR.cu IR.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o IR.o IR.cu

IndexIR.o: IR.o IndexIR.cu IndexIR.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o IndexIR.o IndexIR.cu

FullRelationIR.o: IR.o FullRelationIR.cu FullRelationIR.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o FullRelationIR.o FullRelationIR.cu

ExecutionPlanTree.o: ExecutionPlanTree.cu ExecutionPlanTree.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o ExecutionPlanTree.o ExecutionPlanTree.cu

QueryJob.o: QueryJob.cu QueryJob.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o QueryJob.o QueryJob.cu

SelectQueryJob.o: QueryJob.o QueryExecutor.o SelectQueryJob.cu SelectQueryJob.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o SelectQueryJob.o SelectQueryJob.cu

JoinQueryJob.o: QueryJob.o JoinQueryJob.cu JoinQueryJob.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o JoinQueryJob.o JoinQueryJob.cu

SwapIndexJob.o: QueryJob.o SwapIndexJob.cu SwapIndexJob.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o SwapIndexJob.o SwapIndexJob.cu
	
InputParser.o: InputParser.cu InputParser.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o InputParser.o InputParser.cu
	
LinkedData.o: LinkedData.cu LinkedData.h vedas.h
	nvcc $(MGPU_FLAGS) -c -o LinkedData.o LinkedData.cu

ExecutionWorker.o: ExecutionWorker.cu ExecutionWorker.h vedas.h
	nvcc -Xcompiler="-pthread" $(MGPU_FLAGS) -c -o ExecutionWorker.o ExecutionWorker.cu

JoinGraph.o: JoinGraph/JoinGraph.cu JoinGraph/JoinGraph.h
	nvcc $(MGPU_FLAGS) -c -o JoinGraph.o JoinGraph/JoinGraph.cu

VariableNode.o: JoinGraph/VariableNode.cu JoinGraph/VariableNode.h
	nvcc $(MGPU_FLAGS) -c -o VariableNode.o JoinGraph/VariableNode.cu

IrNode.o: JoinGraph/IrNode.cu JoinGraph/IrNode.h
	nvcc $(MGPU_FLAGS) -c -o IrNode.o JoinGraph/IrNode.cu

TriplePatternNode.o: JoinGraph/TriplePatternNode.cu JoinGraph/TriplePatternNode.h
	nvcc $(MGPU_FLAGS) -c -o TriplePatternNode.o JoinGraph/TriplePatternNode.cu

clean:
	rm -rf join vdBuild vdQuery vdClean vdBench *.o
