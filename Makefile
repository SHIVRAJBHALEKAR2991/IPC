##############################################################################
# Makefile — N-Body Gravitational Simulation
# Builds: serial, openmp, cuda, mpi  →  bin/
##############################################################################

# ── Directories ────────────────────────────────────────────────────────────
SRC     := src
INC     := include
BIN     := bin
SCRIPTS := scripts

# ── Compiler toolchain ─────────────────────────────────────────────────────
CXX      := g++
MPICXX   := mpicxx
NVCC     := nvcc

# ── Common flags ───────────────────────────────────────────────────────────
# -O3           : maximum scalar optimisation
# -march=native : tune for host CPU (auto-vectorisation, AVX etc.)
# -std=c++17    : modern C++ for structured bindings / if-constexpr
CXXFLAGS  := -O3 -march=native -std=c++17 -I$(INC) -Wall -Wextra
OMPFLAGS  := -fopenmp
MPIFLAGS  :=                        # extra flags if needed
# -arch=sm_80   : Ampere (A100, RTX 30xx).  Change to sm_75 for Turing,
#                 sm_86 for GA102 (RTX 3090), sm_90 for Hopper (H100).
# -Xcompiler -O3: pass O3 through to the host-side C++ compiler inside nvcc
NVCCFLAGS := -O3 -arch=sm_80 -std=c++17 -I$(INC) \
             -Xcompiler -O3 \
             --use_fast_math           \
             -lineinfo                 \
             --generate-line-info

# ── Simulation parameters (override on command line) ───────────────────────
N       ?= 16384
STEPS   ?= 10

# ── Default target ─────────────────────────────────────────────────────────
.PHONY: all serial openmp cuda mpi clean bench help

all: serial openmp cuda mpi

# ── Create bin directory ───────────────────────────────────────────────────
$(BIN):
	mkdir -p $(BIN)

# ── Serial ─────────────────────────────────────────────────────────────────
serial: $(BIN)/serial

$(BIN)/serial: $(SRC)/serial.cpp $(INC)/common.h | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  [OK]  Built $@"

# ── OpenMP ─────────────────────────────────────────────────────────────────
openmp: $(BIN)/openmp

$(BIN)/openmp: $(SRC)/openmp.cpp $(INC)/common.h | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "  [OK]  Built $@"

# ── CUDA ───────────────────────────────────────────────────────────────────
cuda: $(BIN)/cuda

$(BIN)/cuda: $(SRC)/cuda.cu $(INC)/common.h | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "  [OK]  Built $@"

# ── MPI ────────────────────────────────────────────────────────────────────
mpi: $(BIN)/mpi

$(BIN)/mpi: $(SRC)/mpi.cpp $(INC)/common.h | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) -o $@ $<
	@echo "  [OK]  Built $@"

# ── Quick runs (for manual testing, not timed) ─────────────────────────────
run-serial: $(BIN)/serial
	./$(BIN)/serial $(N) $(STEPS)

run-openmp: $(BIN)/openmp
	OMP_NUM_THREADS=$$(nproc) ./$(BIN)/openmp $(N) $(STEPS)

run-cuda: $(BIN)/cuda
	./$(BIN)/cuda $(N) $(STEPS)

run-mpi: $(BIN)/mpi
	mpirun --oversubscribe -np $$(( $$(nproc) < 8 ? $$(nproc) : 8 )) ./$(BIN)/mpi $(N) $(STEPS)

# ── Full benchmark (calls benchmark.py) ────────────────────────────────────
bench: all
	python3 $(SCRIPTS)/benchmark.py --n $(N) --steps $(STEPS)

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BIN) timing_results.csv timing_results.png
	@echo "  [OK]  Cleaned"

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  N-Body Gravitational Simulation — Build Targets"
	@echo "  ================================================"
	@echo "  make all          Build serial, openmp, cuda, mpi"
	@echo "  make serial       Build serial baseline"
	@echo "  make openmp       Build OpenMP version"
	@echo "  make cuda         Build CUDA version"
	@echo "  make mpi          Build MPI version"
	@echo "  make run-serial   Quick serial run"
	@echo "  make run-openmp   Quick OpenMP run"
	@echo "  make run-cuda     Quick CUDA run"
	@echo "  make run-mpi      Quick MPI run"
	@echo "  make bench        Run full benchmark + generate CSV + plot"
	@echo "  make clean        Remove build artefacts"
	@echo ""
	@echo "  Override parameters:"
	@echo "    make bench N=8192 STEPS=20"
	@echo ""
	@echo "  Change CUDA arch:"
	@echo "    make cuda NVCCFLAGS='-O3 -arch=sm_75 -std=c++17 -I$(INC)'"
	@echo ""
