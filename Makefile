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

DEBUG_LEVEL = #-g -G

EXTRA_CONFIGS := -std=c++17 -O3 -Xptxas -O3
# EXTRA_CONFIGS := $(EXTRA_CONFIGS) -DVERBOSE_DEBUG
# EXTRA_CONFIGS := $(EXTRA_CONFIGS) -DTIME_DEBUG


MGPU_FLAGS = ${DEBUG_LEVEL} $(EXTRA_CONFIGS) \
						-gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM) \
						-I ./ --expt-extended-lambda -Wno-deprecated-declarations \
						-lraptor2 -lrasqal

VEDAS_QUERY_DEPS =  IR.o FullRelationIR.o DataMetaInfo.o RdfData.o TriplePattern.o \
                    SparqlResult.o QueryExecutor.o VedasStorage.o QueryPlan.o QueryGraph.o JoinGraph.o \
                    QueryJob.o JoinQueryJob.o SelectQueryJob.o IndexSwapJob.o TransferJob.o \
                    InputParser.o ExecutionPlanTree.o ExecutionWorker.o EmptyIntervalDict.o \
                    SparqlQuery.o DataMetaInfo.o SegmentTree.o Histogram.o

VEDAS_BUILD_DEPS = 	SegmentTree.o DataMetaInfo.o Histogram.o \
										InputParser.o VedasStorage.o RdfData.o \
										src/LinkedData.cu src/LinkedData.h

all: vdQuery vdBuild vdBench vdClean
	echo ${GREEN}===================== SUCCESS =====================${NC}

vdBuild: $(VEDAS_BUILD_DEPS) src/VedasBuild.cu src/loader.cu
	nvcc $(MGPU_FLAGS) *.o src/loader.cu -o vdBuild src/VedasBuild.cu src/LinkedData.cu

vdBench: $(VEDAS_QUERY_DEPS) src/VedasBench.cu src/loader.cu
	nvcc $(MGPU_FLAGS) *.o src/loader.cu -o vdBench src/VedasBench.cu

vdQuery: $(VEDAS_QUERY_DEPS) src/VedasQuery.cu src/loader.cu
	nvcc $(MGPU_FLAGS) *.o src/loader.cu src/VedasQuery.cu -o vdQuery

vdClean: src/VedasClean.cu
	nvcc $(MGPU_FLAGS) src/VedasClean.cu -o vdClean

SparqlResult.o: src/SparqlResult.h src/SparqlResult.cu src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o SparqlResult.o src/SparqlResult.cu

SparqlQuery.o: src/SparqlQuery.h src/SparqlQuery.cu src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o SparqlQuery.o src/SparqlQuery.cu

TriplePattern.o: src/TriplePattern.h src/TriplePattern.cu src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o TriplePattern.o src/TriplePattern.cu

QueryExecutor.o: src/QueryExecutor.h src/QueryExecutor.cu src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o QueryExecutor.o src/QueryExecutor.cu

VedasStorage.o: src/VedasStorage.cu src/VedasStorage.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o VedasStorage.o src/VedasStorage.cu

RdfData.o: src/RdfData.h src/RdfData.cu src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o RdfData.o src/RdfData.cu

DataMetaInfo.o: src/DataMetaInfo.h src/DataMetaInfo.cu src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o DataMetaInfo.o src/DataMetaInfo.cu

QueryPlan.o: src/QueryPlan.cu src/QueryPlan.h src/vedas.h 
	nvcc $(MGPU_FLAGS) -c -o QueryPlan.o src/QueryPlan.cu

IR.o: src/IR.cu src/IR.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o IR.o src/IR.cu

FullRelationIR.o: src/IR.h src/FullRelationIR.cu src/FullRelationIR.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o FullRelationIR.o src/FullRelationIR.cu

ExecutionPlanTree.o: TriplePattern.o src/ExecutionPlanTree.cu src/ExecutionPlanTree.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o ExecutionPlanTree.o src/ExecutionPlanTree.cu

QueryGraph.o: src/QueryGraph.cu src/QueryGraph.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o QueryGraph.o src/QueryGraph.cu

JoinGraph.o: src/JoinGraph.cu src/JoinGraph.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o JoinGraph.o src/JoinGraph.cu

QueryJob.o: src/QueryJob.cu src/QueryJob.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o QueryJob.o src/QueryJob.cu

JoinQueryJob.o: src/JoinQueryJob.cu src/JoinQueryJob.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o JoinQueryJob.o src/JoinQueryJob.cu

SelectQueryJob.o: src/SelectQueryJob.cu src/SelectQueryJob.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o SelectQueryJob.o src/SelectQueryJob.cu

IndexSwapJob.o: src/IndexSwapJob.cu src/IndexSwapJob.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o IndexSwapJob.o src/IndexSwapJob.cu

TransferJob.o: src/TransferJob.cu src/TransferJob.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o TransferJob.o src/TransferJob.cu

InputParser.o: src/InputParser.cu src/InputParser.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o InputParser.o src/InputParser.cu

ExecutionWorker.o: src/ExecutionWorker.cu src/ExecutionWorker.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o ExecutionWorker.o src/ExecutionWorker.cu

EmptyIntervalDict.o: src/EmptyIntervalDict.cu src/EmptyIntervalDict.h src/vedas.h
	nvcc $(MGPU_FLAGS) -c -o EmptyIntervalDict.o src/EmptyIntervalDict.cu

SegmentTree.o: src/util/SegmentTree.cu src/util/SegmentTree.h
	nvcc $(MGPU_FLAGS) -c -o SegmentTree.o src/util/SegmentTree.cu 

Histogram.o: src/Histogram.cu src/Histogram.h
	nvcc $(MGPU_FLAGS) -c -o Histogram.o src/Histogram.cu



test: FORCE
	nvcc $(MGPU_FLAGS) -o testrunner test/*.cu $(TESTSOURCES) -lgtest
	./testrunner

clean:
	rm -rf join vdBuild vdQuery vdClean vdBench *.o

testclean:
	rm testrunner

FORCE: ;

testPartition: tools/TestPartition.cpp
	g++ -std=c++11 tools/TestPartition.cpp -o testPartition -lmetis
