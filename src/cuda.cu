/**
 * cuda.cu
 * -------
 * CUDA-accelerated O(N²) N-Body gravitational simulation using
 * Shared Memory Tiling for improved memory throughput.
 *
 * Algorithm — Shared Memory Tile Pattern:
 * ────────────────────────────────────────
 *   The N×N interaction matrix is divided into (N/TILE) × (N/TILE) blocks.
 *   Each CUDA thread block of TILE threads cooperatively loads a tile of
 *   TILE source particles into __shared__ memory, then all threads compute
 *   forces from those TILE particles before loading the next tile.
 *
 *   This converts N²/TILE global-memory reads of 28-byte Particle structs
 *   into N²/TILE shared-memory accesses, greatly reducing DRAM bandwidth
 *   pressure (each particle position is read from DRAM exactly N/TILE
 *   times instead of N times).
 *
 * Thread / block mapping:
 *   - One thread per particle (target particle i = blockIdx.x*blockDim.x + threadIdx.x)
 *   - blockDim.x = TILE_SIZE (must be a compile-time constant for __shared__)
 *   - gridDim.x  = ceil(N / TILE_SIZE)
 *
 * Build (via Makefile):
 *   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude -o bin/cuda src/cuda.cu
 *
 * Run:
 *   ./bin/cuda [N] [steps]
 */

#include "common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
// Tile size — must be a compile-time constant for __shared__ arrays.
// 256 threads/block is generally optimal for Ampere; tune per GPU.
// ─────────────────────────────────────────────────────────────────────────────
#define TILE_SIZE 256

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error checking macro
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Device-side position/mass tuple stored in shared memory
// ─────────────────────────────────────────────────────────────────────────────

struct float4_pm {
    float x, y, z, mass;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tiled N-Body force kernel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * nbody_force_tiled_kernel
 *
 * Each thread is responsible for particle i = global thread index.
 * The kernel iterates over tiles of the particle array, loading each tile
 * cooperatively into shared memory before accumulating forces.
 *
 * @param pos_mass  float4 array [x, y, z, mass] for all N particles (read)
 * @param vel       velocity array [vx, vy, vz]  for all N particles (read/write)
 * @param n         total number of particles
 * @param dt        time-step
 */
__global__ void nbody_force_tiled_kernel(
    const float4* __restrict__ pos_mass,   // [x,y,z,mass] packed
    float4*       __restrict__ vel,         // [vx,vy,vz,_]
    const int n,
    const float dt)
{
    // Shared memory tile for source particles (position + mass)
    __shared__ float4 tile[TILE_SIZE];

    // Global thread index = target particle index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load target particle (may be out-of-bounds — handle below)
    float xi = 0.f, yi = 0.f, zi = 0.f;
    float vxi = 0.f, vyi = 0.f, vzi = 0.f;
    if (i < n) {
        float4 pi = pos_mass[i];
        xi = pi.x; yi = pi.y; zi = pi.z;
        float4 vi = vel[i];
        vxi = vi.x; vyi = vi.y; vzi = vi.z;
    }

    float axi = 0.f, ayi = 0.f, azi = 0.f;

    // ── Tile loop ─────────────────────────────────────────────────────────
    int n_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < n_tiles; ++t) {
        // Collaborative load: thread threadIdx.x loads source particle
        // (t * TILE_SIZE + threadIdx.x) into shared memory
        int j_src = t * TILE_SIZE + threadIdx.x;
        if (j_src < n) {
            tile[threadIdx.x] = pos_mass[j_src];
        } else {
            // Padding: zero mass → no contribution
            tile[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();  // ensure tile is fully populated before use

        // ── Inner loop over tile ─────────────────────────────────────────
        if (i < n) {
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                int j_global = t * TILE_SIZE + k;

                float dx = tile[k].x - xi;
                float dy = tile[k].y - yi;
                float dz = tile[k].z - zi;
                float mj = tile[k].w;

                float dist2     = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                float inv_dist  = rsqrtf(dist2);           // fast GPU rsqrt
                float inv_dist3 = inv_dist * inv_dist * inv_dist;

                // Mask self-interaction
                float factor = (j_global != i && j_global < n)
                               ? G * mj * inv_dist3 : 0.f;

                axi += factor * dx;
                ayi += factor * dy;
                azi += factor * dz;
            }
        }
        __syncthreads();  // prevent next tile load while inner loop runs
    }

    // ── Euler integration and write-back ──────────────────────────────────
    if (i < n) {
        vxi += axi * dt;
        vyi += ayi * dt;
        vzi += azi * dt;

        // Update velocity
        vel[i] = make_float4(vxi, vyi, vzi, 0.f);

        // Update position (inline — avoid extra global read)
        float4 pi = pos_mass[i];
        pi.x += vxi * dt;
        pi.y += vyi * dt;
        pi.z += vzi * dt;

        // Cast away const via raw pointer — pos_mass is logically in/out
        ((float4*)pos_mass)[i] = pi;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host utility: pack Particle[] → float4 arrays
// ─────────────────────────────────────────────────────────────────────────────

static void pack_particles(const Particle* p, int n,
                           float4* pos_mass, float4* vel) {
    for (int i = 0; i < n; ++i) {
        pos_mass[i] = make_float4(p[i].x, p[i].y, p[i].z, p[i].mass);
        vel[i]      = make_float4(p[i].vx, p[i].vy, p[i].vz, 0.f);
    }
}

static void unpack_particles(Particle* p, int n,
                             const float4* pos_mass, const float4* vel) {
    for (int i = 0; i < n; ++i) {
        p[i].x    = pos_mass[i].x;
        p[i].y    = pos_mass[i].y;
        p[i].z    = pos_mass[i].z;
        p[i].mass = pos_mass[i].w;
        p[i].vx   = vel[i].x;
        p[i].vy   = vel[i].y;
        p[i].vz   = vel[i].z;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int n     = (argc > 1) ? atoi(argv[1]) : N_PARTICLES;
    int steps = (argc > 2) ? atoi(argv[2]) : N_STEPS;

    // Query GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("=== CUDA N-Body Simulation ===\n");
    printf("GPU: %s  |  SM: %d.%d  |  SMs: %d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("N=%d  steps=%d  tile=%d\n\n", n, steps, TILE_SIZE);

    // ── Host buffers ──────────────────────────────────────────────────────
    Particle* h_particles = new Particle[n];
    init_particles(h_particles, n);

    float4* h_pos_mass = new float4[n];
    float4* h_vel      = new float4[n];
    pack_particles(h_particles, n, h_pos_mass, h_vel);

    // ── Device buffers ────────────────────────────────────────────────────
    float4 *d_pos_mass, *d_vel;
    CUDA_CHECK(cudaMalloc(&d_pos_mass, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel,      n * sizeof(float4)));

    CUDA_CHECK(cudaMemcpy(d_pos_mass, h_pos_mass, n * sizeof(float4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel, n * sizeof(float4),
                          cudaMemcpyHostToDevice));

    // ── CUDA events for accurate GPU timing ──────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    dim3 block(TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up pass (not timed) — amortises JIT compilation cost
    nbody_force_tiled_kernel<<<grid, block>>>(d_pos_mass, d_vel, n, DT);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore initial state for fair timing
    CUDA_CHECK(cudaMemcpy(d_pos_mass, h_pos_mass, n * sizeof(float4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel, n * sizeof(float4),
                          cudaMemcpyHostToDevice));

    // ── Timed simulation loop ─────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int s = 0; s < steps; ++s) {
        nbody_force_tiled_kernel<<<grid, block>>>(d_pos_mass, d_vel, n, DT);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    // ── Copy result back ──────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_pos_mass, d_pos_mass, n * sizeof(float4),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel, d_vel, n * sizeof(float4),
                          cudaMemcpyDeviceToHost));

    // ── Report ─────────────────────────────────────────────────────────────
    print_result("CUDA", n, steps, (double)gpu_ms);
    printf("CSV,CUDA,%d,%d,%.6f,%s\n", n, steps, gpu_ms, prop.name);

    // ── Cleanup ───────────────────────────────────────────────────────────
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_pos_mass);
    cudaFree(d_vel);
    delete[] h_pos_mass;
    delete[] h_vel;
    delete[] h_particles;
    return 0;
}
