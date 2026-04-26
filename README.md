# N-Body Gravitational Simulation — HPC Benchmark Suite

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-sm__80-green)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-5.0-orange)](https://www.openmp.org/)
[![MPI](https://img.shields.io/badge/MPI-3.1-red)](https://www.mpi-forum.org/)

---

## Problem Statement

### What problem are we solving?

Imagine you have thousands of stars or planets floating in space. Every single object pulls every other object towards itself using gravity. To simulate how they all move over time, we need to calculate the gravitational force between **every possible pair** of objects at every time step.

For example, if you have **N = 16,384 particles**, that means roughly **268 million force calculations per step**. If you run 10 simulation steps, that's **2.68 billion calculations** in total. Doing this on a single CPU core takes several seconds per step — too slow for any real-world use.

### Why is this hard?

The core challenge is the **O(N²) complexity** — the number of calculations grows as the square of the number of particles:

| Particles (N) | Force pairs per step | Work grows by |
|---------------|----------------------|---------------|
| 1,000 | ~1 million | — |
| 10,000 | ~100 million | 100× |
| 100,000 | ~10 billion | 10,000× |

Doubling the number of particles makes the problem **4× harder**. This means a naive single-threaded program simply cannot keep up as the simulation grows.

### How do we solve it?

This project implements the same simulation in **four different ways**, each using a different technique to speed it up by splitting the work across multiple processors:

| Implementation | Strategy | Typical Speedup |
|----------------|----------|-----------------|
| **Serial** | Baseline — one CPU core, no parallelism | 1× (reference) |
| **OpenMP** | Splits work across all CPU cores on one machine | ~5–15× |
| **CUDA** | Offloads work to thousands of GPU cores | ~200–800× |
| **MPI** | Distributes work across multiple processes (or machines) | ~4–8× |

The goal is to show how the same mathematical problem can be accelerated dramatically using parallel programming — and to measure exactly how much each approach helps.

---

A professional-grade **O(N²) brute-force N-Body gravitational simulation** implemented in four parallel programming paradigms — Serial, OpenMP, CUDA (shared-memory tiling), and MPI — designed to demonstrate and benchmark HPC acceleration strategies.

---

## Repository Structure

```
IPC/
├── include/
│   └── common.h          # Shared Particle struct, constants, timing utils
├── src/
│   ├── serial.cpp         # Baseline single-threaded implementation
│   ├── openmp.cpp         # Multi-core via OpenMP (schedule(static))
│   ├── cuda.cu            # GPU via CUDA shared-memory tiling (sm_80)
│   └── mpi.cpp            # Multi-node via MPI_Allgather
├── scripts/
│   ├── benchmark.py       # Orchestrator → timing_results.csv + plot
│   └── setup_env.sh       # Dependency installer (Ubuntu/Debian)
├── Makefile               # Build system with -O3 / -arch=sm_80 flags
└── README.md
```

---

## Environment Setup

### Automatic (Ubuntu 20.04 / 22.04)

```bash
sudo bash scripts/setup_env.sh
```

### Manual package list

| Package            | Purpose                          |
|--------------------|----------------------------------|
| `build-essential`  | GCC, G++, Make                   |
| `cmake`            | Build support                    |
| `libomp-dev`       | OpenMP LLVM runtime headers      |
| `openmpi-bin`      | `mpirun` launcher                |
| `libopenmpi-dev`   | MPI headers + link libraries     |
| `nvidia-cuda-toolkit` | `nvcc`, `cuda_runtime.h`     |

```bash
sudo apt-get install -y build-essential cmake libomp-dev \
    openmpi-bin libopenmpi-dev nvidia-cuda-toolkit
pip install matplotlib pandas tabulate
```

---

## Build

```bash
# Build all targets
make all

# Build individual targets
make serial
make openmp
make cuda
make mpi

# Override CUDA architecture (e.g. Turing sm_75, Hopper sm_90)
make cuda NVCCFLAGS="-O3 -arch=sm_75 -std=c++17 -Iinclude"
```

---

## Run

### Individual binaries

```bash
# Serial  — N=16384  steps=10
./bin/serial 16384 10

# OpenMP  — uses all available cores
OMP_NUM_THREADS=16 ./bin/openmp 16384 10

# CUDA    — warm-up step included internally
./bin/cuda 16384 10

# MPI     — 8 ranks
mpirun -np 8 ./bin/mpi 16384 10
```

### Full benchmark

```bash
make bench                      # N=16384, steps=10  (defaults)
make bench N=8192 STEPS=20      # custom parameters
```

This runs all four implementations, writes **`timing_results.csv`**, and produces **`timing_results.png`**.

---

## Technical Design

### `common.h` — Shared Infrastructure

```c
struct Particle {
    float x, y, z;    // Position [m]
    float vx, vy, vz; // Velocity [m/s]
    float mass;        // Mass     [kg]
};
```

All implementations use the same `init_particles()` with seed `42` for reproducibility.

---

### Serial (`src/serial.cpp`)

Straightforward double-loop with a softened gravitational kernel:

```
F_ij = G * m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
```

Followed by an Euler velocity-position integration step.

---

### OpenMP (`src/openmp.cpp`)

```cpp
#pragma omp parallel for schedule(static) default(none) shared(p, ax, ay, az, n)
for (int i = 0; i < n; ++i) { ... }
```

- **`schedule(static)`**: equal-size chunks per thread — optimal here because every iteration `i` has identical work (N-1 inner iterations).
- Private accumulators `axi/ayi/azi` avoid any critical/atomic operations in the inner loop.
- A second parallel-for integrates velocities and positions.

---

### CUDA (`src/cuda.cu`) — Shared Memory Tiling

The key optimisation is the **tiled force kernel**:

```
Grid:  ceil(N / TILE_SIZE) blocks
Block: TILE_SIZE threads  (TILE_SIZE = 256)
```

Each block cooperatively loads `TILE_SIZE` source particles into `__shared__` memory:

```
Total DRAM reads = N × (N / TILE_SIZE)   [tiled]
                 ≪ N × N                  [naive]
```

Benefits:
- `rsqrtf()` — single-precision GPU fast reciprocal sqrt
- `#pragma unroll 8` — compiler-directed inner-loop unrolling
- `--use_fast_math` — enables FTZ, MAD fusion

---

### MPI (`src/mpi.cpp`) — Allgather / Replicated-Data Pattern

```
Each rank r owns particles [displs[r], displs[r] + counts[r])
```

**Each step:**
1. Each rank computes forces only for its own slice using the *full* particle array.
2. `MPI_Allgatherv` distributes all updated local slices → every rank holds the complete new state.

```
Communication cost / step = O(N × P)   (P = number of ranks)
Compute cost / rank       = O(N² / P)
```

A registered `MPI_PARTICLE` contiguous type (7 floats) avoids manual serialisation.

---

## Benchmark Output

```
============================================================
  N-Body HPC Benchmark Suite
  N=16384  steps=10
============================================================

Serial       | N= 16384 | steps= 10 | total=XXXXX ms | per-step=XXXX ms | ~X.XX GFLOP/s
OpenMP       | N= 16384 | steps= 10 | total=XXXXX ms | per-step=XXXX ms | ~X.XX GFLOP/s
CUDA         | N= 16384 | steps= 10 | total=XXXXX ms | per-step=XXXX ms | ~X.XX GFLOP/s
MPI          | N= 16384 | steps= 10 | total=XXXXX ms | per-step=XXXX ms | ~X.XX GFLOP/s
```

CSV columns: `Implementation, N, Steps, Total_ms, PerStep_ms, Speedup_vs_Serial, GFLOP_s`

---

## Expected Speedups (indicative — hardware dependent)

| Implementation | Typical Speedup (vs Serial) |
|----------------|-----------------------------|
| OpenMP (16 cores) | ~12–15×                  |
| CUDA (A100)       | ~200–800×                |
| MPI (8 ranks)     | ~6–8× (single node)      |

---

## Tuning Notes

| Parameter | File | Default | Notes |
|-----------|------|---------|-------|
| `TILE_SIZE` | `cuda.cu` | 256 | Must divide evenly; try 128/512 |
| `-arch=sm_XX` | `Makefile` | sm_80 | sm_75=Turing, sm_86=GA102, sm_90=Hopper |
| `OMP_NUM_THREADS` | env var | `nproc` | Set before launching `./bin/openmp` |
| `mpirun -np P` | command | CPU count | Try P = 1,2,4,8 for scaling study |
| `N_PARTICLES` | `common.h` | 16384 | Override: `-DN_PARTICLES=32768` |
| `SOFTENING` | `common.h` | 1e-9 | Prevents singularity at r≈0 |

---

## License

MIT — free to use, modify, and redistribute.
