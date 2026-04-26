/**
 * mpi.cpp
 * -------
 * MPI-distributed O(N²) N-Body gravitational simulation.
 *
 * Communication strategy:
 *   - All N particles are partitioned equally across P ranks (rows of the
 *     interaction matrix).
 *   - Before each force-calculation step every rank broadcasts its local
 *     sub-array via MPI_Allgather so that every rank holds the full particle
 *     state and can compute forces for its slice independently.
 *   - After integration, a second MPI_Allgather synchronises the updated
 *     positions / velocities for the next step.
 *
 * This is the replicated-data / Allgather pattern — simple and scalable
 * as long as N fits in each node's memory.
 *
 * Build (via Makefile):
 *   mpicxx -O3 -std=c++17 -Iinclude -o bin/mpi src/mpi.cpp
 *
 * Run:
 *   mpirun -np <P> ./bin/mpi [N] [steps]
 */

#include "common.h"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// MPI Particle type registration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * create_particle_type - registers a contiguous MPI derived type for
 * 'struct Particle' so MPI can send/receive whole particles without manual
 * serialisation.
 */
static MPI_Datatype create_particle_type() {
    MPI_Datatype mpi_particle;
    // Particle = 7 floats (x,y,z,vx,vy,vz,mass) laid out contiguously
    MPI_Type_contiguous(7, MPI_FLOAT, &mpi_particle);
    MPI_Type_commit(&mpi_particle);
    return mpi_particle;
}

// ─────────────────────────────────────────────────────────────────────────────
// Force + integration for a sub-range [i_start, i_end)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * compute_forces_range - each rank computes forces only for its own slice
 * [i_start, i_end) but uses the full particle array 'all' for the j-loop.
 *
 * @param all      full particle array (N particles, read-only for j-loop)
 * @param local    local slice (i_start..i_end-1) to integrate (in/out)
 * @param i_start  first global index owned by this rank
 * @param i_end    one past the last global index owned by this rank
 * @param n_total  total particle count
 * @param dt       time-step
 */
static void compute_forces_range(
        const Particle* __restrict__ all,
        Particle*       __restrict__ local,
        int i_start, int i_end,
        int n_total, float dt)
{
    int local_n = i_end - i_start;

    std::vector<float> ax(local_n, 0.f);
    std::vector<float> ay(local_n, 0.f);
    std::vector<float> az(local_n, 0.f);

    // ── Force accumulation ────────────────────────────────────────────────
    for (int li = 0; li < local_n; ++li) {
        int gi  = i_start + li;   // global index
        float xi = all[gi].x, yi = all[gi].y, zi = all[gi].z;
        float axi = 0.f, ayi = 0.f, azi = 0.f;

        for (int j = 0; j < n_total; ++j) {
            float dx = all[j].x - xi;
            float dy = all[j].y - yi;
            float dz = all[j].z - zi;

            float dist2     = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            float inv_dist  = 1.0f / sqrtf(dist2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float factor    = (gi != j) ? G * all[j].mass * inv_dist3 : 0.f;

            axi += factor * dx;
            ayi += factor * dy;
            azi += factor * dz;
        }
        ax[li] = axi;
        ay[li] = ayi;
        az[li] = azi;
    }

    // ── Euler integration ─────────────────────────────────────────────────
    for (int li = 0; li < local_n; ++li) {
        local[li].vx += ax[li] * dt;
        local[li].vy += ay[li] * dt;
        local[li].vz += az[li] * dt;

        local[li].x  += local[li].vx * dt;
        local[li].y  += local[li].vy * dt;
        local[li].z  += local[li].vz * dt;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n     = (argc > 1) ? atoi(argv[1]) : N_PARTICLES;
    int steps = (argc > 2) ? atoi(argv[2]) : N_STEPS;

    if (rank == 0) {
        printf("=== MPI N-Body Simulation ===\n");
        printf("N=%d  steps=%d  ranks=%d\n\n", n, steps, size);
    }

    // ── Register MPI particle type ────────────────────────────────────────
    MPI_Datatype MPI_PARTICLE = create_particle_type();

    // ── Partition particles among ranks ──────────────────────────────────
    // Use a simple block decomposition; handle remainder on the last rank.
    int base_chunk = n / size;
    int remainder  = n % size;

    // sendcounts and displacements for MPI_Allgatherv
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = base_chunk + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
    }

    int i_start = displs[rank];
    int i_end   = i_start + counts[rank];
    int local_n = counts[rank];

    // ── Allocate buffers ──────────────────────────────────────────────────
    Particle* all_particles   = new Particle[n];        // full state (all ranks)
    Particle* local_particles = new Particle[local_n];  // this rank's slice

    // Rank 0 initialises and broadcasts the full initial state
    if (rank == 0) init_particles(all_particles, n);
    MPI_Bcast(all_particles, n, MPI_PARTICLE, 0, MPI_COMM_WORLD);

    // Copy this rank's slice into local buffer
    memcpy(local_particles, all_particles + i_start,
           local_n * sizeof(Particle));

    // ── Timed simulation loop ─────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int s = 0; s < steps; ++s) {
        // 1. Every rank computes forces for its own slice using full state
        compute_forces_range(all_particles, local_particles,
                             i_start, i_end, n, DT);

        // 2. Gather updated slices → every rank has the full new state
        //    MPI_Allgather: each rank contributes counts[rank] particles,
        //    resulting in the complete N-particle array on all ranks.
        MPI_Allgatherv(
            local_particles, local_n, MPI_PARTICLE,    // send
            all_particles, counts.data(),               // recv (all)
            displs.data(), MPI_PARTICLE,
            MPI_COMM_WORLD
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end   = MPI_Wtime();
    double total_ms = (t_end - t_start) * 1000.0;

    // ── Report (rank 0 only) ──────────────────────────────────────────────
    if (rank == 0) {
        print_result("MPI", n, steps, total_ms);
        printf("CSV,MPI,%d,%d,%.6f,%d\n", n, steps, total_ms, size);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    MPI_Type_free(&MPI_PARTICLE);
    delete[] all_particles;
    delete[] local_particles;

    MPI_Finalize();
    return 0;
}
