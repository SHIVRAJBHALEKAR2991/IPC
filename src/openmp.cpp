/**
 * openmp.cpp
 * ----------
 * OpenMP-parallelised O(N²) N-Body gravitational simulation.
 *
 * Parallelisation strategy:
 *   - The outer force loop (over particle i) is distributed with
 *     `#pragma omp parallel for schedule(static)`.
 *   - Each thread owns a private accumulator (axi/ayi/azi) so no
 *     atomic or critical sections are needed in the hot path.
 *   - The integration step is also parallelised with a separate
 *     parallel-for, sharing the same thread pool.
 *
 * Build (via Makefile):
 *   g++ -O3 -std=c++17 -fopenmp -Iinclude -o bin/openmp src/openmp.cpp
 *
 * Run:
 *   OMP_NUM_THREADS=<T>  ./bin/openmp [N] [steps]
 */

#include "common.h"
#include <omp.h>
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
// Force + integration kernel (single step)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * compute_forces_omp - parallelised pairwise gravitational force calculation
 * using OpenMP with static scheduling.
 *
 * @param p   particle array (in/out)
 * @param n   number of particles
 * @param dt  time-step
 */
static void compute_forces_omp(Particle* __restrict__ p, int n, float dt) {

    // Temporary acceleration buffers (shared, filled by parallel loop)
    float* ax = new float[n]();
    float* ay = new float[n]();
    float* az = new float[n]();

    // ── O(N²) pairwise force — parallelised over i ───────────────────────
    //
    // schedule(static) gives equal-size chunks to each thread.
    // This is optimal when all iterations cost the same, which holds for
    // the full N-body force loop (every i sees N-1 neighbours).
    #pragma omp parallel for schedule(static) default(none) \
        shared(p, ax, ay, az, n)
    for (int i = 0; i < n; ++i) {
        float xi = p[i].x, yi = p[i].y, zi = p[i].z;
        float axi = 0.f, ayi = 0.f, azi = 0.f;

        // Inner loop: accumulate forces from all other particles
        for (int j = 0; j < n; ++j) {
            float dx = p[j].x - xi;
            float dy = p[j].y - yi;
            float dz = p[j].z - zi;

            float dist2     = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            float inv_dist  = 1.0f / sqrtf(dist2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            // Mask self-interaction (cheaper than a branch in the inner loop)
            float factor    = (i != j) ? G * p[j].mass * inv_dist3 : 0.f;

            axi += factor * dx;
            ayi += factor * dy;
            azi += factor * dz;
        }

        // Write result — each thread writes to its own i slot → no race
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }

    // ── Euler integration — also parallelised ────────────────────────────
    #pragma omp parallel for schedule(static) default(none) \
        shared(p, ax, ay, az, n)
    for (int i = 0; i < n; ++i) {
        p[i].vx += ax[i] * dt;
        p[i].vy += ay[i] * dt;
        p[i].vz += az[i] * dt;

        p[i].x  += p[i].vx * dt;
        p[i].y  += p[i].vy * dt;
        p[i].z  += p[i].vz * dt;
    }

    delete[] ax;
    delete[] ay;
    delete[] az;
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int n     = (argc > 1) ? atoi(argv[1]) : N_PARTICLES;
    int steps = (argc > 2) ? atoi(argv[2]) : N_STEPS;
    int nthreads = omp_get_max_threads();

    printf("=== OpenMP N-Body Simulation ===\n");
    printf("N=%d  steps=%d  threads=%d\n\n", n, steps, nthreads);

    Particle* particles = new Particle[n];
    init_particles(particles, n);

    // ── Timed simulation loop ─────────────────────────────────────────────
    auto t_start = now();

    for (int s = 0; s < steps; ++s) {
        compute_forces_omp(particles, n, DT);
    }

    auto t_end = now();
    double total_ms = elapsed_ms(t_start, t_end);

    // ── Report ─────────────────────────────────────────────────────────────
    print_result("OpenMP", n, steps, total_ms);
    printf("CSV,OpenMP,%d,%d,%.6f,%d\n", n, steps, total_ms, nthreads);

    delete[] particles;
    return 0;
}
