/**
 * serial.cpp
 * ----------
 * Baseline single-threaded O(N²) N-Body gravitational simulation.
 *
 * Serves as the reference implementation against which OpenMP, CUDA, and MPI
 * speedups are measured.  Every physical step is identical to the parallel
 * versions so that results can be cross-validated.
 *
 * Build (via Makefile):
 *   g++ -O3 -std=c++17 -Iinclude -o bin/serial src/serial.cpp
 *
 * Run:
 *   ./bin/serial [N] [steps]
 */

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Force + integration kernel (single step)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * compute_forces_serial - calculates pairwise gravitational acceleration
 * for every particle and integrates velocities + positions (Euler step).
 *
 * @param p     particle array (read + write)
 * @param n     number of particles
 * @param dt    time-step
 */
static void compute_forces_serial(Particle* __restrict__ p, int n, float dt) {

    // Accumulate per-particle acceleration in a temporary buffer so that
    // we can do a clean Euler integration at the end of the step.
    float* ax = new float[n]();
    float* ay = new float[n]();
    float* az = new float[n]();

    // ── O(N²) pairwise force loop ─────────────────────────────────────────
    for (int i = 0; i < n; ++i) {
        float xi = p[i].x, yi = p[i].y, zi = p[i].z;
        float axi = 0.f, ayi = 0.f, azi = 0.f;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            float dx = p[j].x - xi;
            float dy = p[j].y - yi;
            float dz = p[j].z - zi;

            float dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            float inv_dist  = 1.0f / sqrtf(dist2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float force     = G * p[j].mass * inv_dist3;

            axi += force * dx;
            ayi += force * dy;
            azi += force * dz;
        }
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }

    // ── Euler integration ─────────────────────────────────────────────────
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

    printf("=== Serial N-Body Simulation ===\n");
    printf("N=%d  steps=%d\n\n", n, steps);

    Particle* particles = new Particle[n];
    init_particles(particles, n);

    // ── Warm up cache (not timed) ─────────────────────────────────────────
    // (intentionally skipped for serial — no GPU/NUMA warm-up needed)

    // ── Timed simulation loop ─────────────────────────────────────────────
    auto t_start = now();

    for (int s = 0; s < steps; ++s) {
        compute_forces_serial(particles, n, DT);
    }

    auto t_end = now();
    double total_ms = elapsed_ms(t_start, t_end);

    // ── Report ─────────────────────────────────────────────────────────────
    print_result("Serial", n, steps, total_ms);

    // Dump CSV line for benchmark.py to collect
    printf("CSV,Serial,%d,%d,%.6f\n", n, steps, total_ms);

    delete[] particles;
    return 0;
}
