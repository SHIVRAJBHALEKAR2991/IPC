/**
 * common.h
 * --------
 * Shared data structures, constants, and utility functions for the
 * N-Body Gravitational Simulation (OpenMP / CUDA / MPI comparison).
 *
 * All implementations #include this header so that the Particle layout
 * and physical constants are identical across every backend.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// Simulation constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of particles
#ifndef N_PARTICLES
#define N_PARTICLES 16384
#endif

/// Gravitational constant (dimensionless for benchmark purposes)
constexpr float G = 6.674e-11f;

/// Softening length to avoid singularities
constexpr float SOFTENING = 1e-9f;

/// Integration time-step
constexpr float DT = 0.01f;

/// Total number of simulation steps to run
#ifndef N_STEPS
#define N_STEPS 10
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Particle structure
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Particle - stores the full kinematic and physical state of a body.
 *
 * Layout is kept as a Structure-of-Arrays-friendly flat struct so that
 * it can be used directly with CUDA device memory and MPI derived types.
 */
struct Particle {
    float x,  y,  z;   ///< Position  [m]
    float vx, vy, vz;  ///< Velocity  [m/s]
    float mass;         ///< Mass      [kg]
};

// ─────────────────────────────────────────────────────────────────────────────
// Initialization helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * init_particles - fills an array of N Particle objects with deterministic
 * pseudo-random state so every backend starts from the identical initial
 * conditions, making timing comparisons fair.
 *
 * @param p   pointer to particle array of length n
 * @param n   number of particles
 * @param seed  random seed (default 42)
 */
inline void init_particles(Particle* p, int n, unsigned int seed = 42) {
    srand(seed);
    const float pos_scale  = 1.0e11f;  // ~1 AU spread
    const float vel_scale  = 1.0e3f;   // ~1 km/s
    const float mass_scale = 1.0e24f;  // ~1 Earth mass

    for (int i = 0; i < n; ++i) {
        p[i].x    = ((float)rand() / RAND_MAX - 0.5f) * pos_scale;
        p[i].y    = ((float)rand() / RAND_MAX - 0.5f) * pos_scale;
        p[i].z    = ((float)rand() / RAND_MAX - 0.5f) * pos_scale;
        p[i].vx   = ((float)rand() / RAND_MAX - 0.5f) * vel_scale;
        p[i].vy   = ((float)rand() / RAND_MAX - 0.5f) * vel_scale;
        p[i].vz   = ((float)rand() / RAND_MAX - 0.5f) * vel_scale;
        p[i].mass = ((float)rand() / RAND_MAX + 0.1f) * mass_scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing helper (wall-clock, portable)
// ─────────────────────────────────────────────────────────────────────────────

using Clock    = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint now() { return Clock::now(); }

inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Result reporting
// ─────────────────────────────────────────────────────────────────────────────

inline void print_result(const char* impl, int n, int steps, double total_ms) {
    double per_step_ms = total_ms / steps;
    double gflops      = (double)n * n * 20.0 / per_step_ms / 1e6; // ~20 FLOPs per pair
    printf("%-12s | N=%6d | steps=%3d | total=%.3f ms | per-step=%.3f ms | ~%.2f GFLOP/s\n",
           impl, n, steps, total_ms, per_step_ms, gflops);
}
