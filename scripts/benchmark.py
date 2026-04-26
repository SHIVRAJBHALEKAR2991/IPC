#!/usr/bin/env python3
"""
benchmark.py
============
Orchestrates the N-Body simulation benchmarks for all four implementations
(Serial, OpenMP, CUDA, MPI), collects timing results, writes a
``timing_results.csv``, and generates a publication-quality comparison chart.

Usage
-----
    python3 scripts/benchmark.py [--n N] [--steps S] [--mpi-ranks R]
                                 [--skip-cuda] [--skip-mpi]
                                 [--out-csv PATH] [--out-png PATH]

The script reads CSV lines printed by each binary in the format:
    CSV,<impl>,<N>,<steps>,<total_ms>[,<extra>]

Dependencies
------------
    pip install matplotlib pandas tabulate

Author: HPC Benchmark Suite
"""

import argparse
import subprocess
import sys
import os
import csv
import re
import time
from pathlib import Path
from typing import Optional

# ── Optional rich output ──────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe on HPC nodes
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR   = REPO_ROOT / "bin"

IMPL_COLORS = {
    "Serial": "#ef4444",   # red
    "OpenMP": "#3b82f6",   # blue
    "CUDA":   "#22c55e",   # green
    "MPI":    "#f59e0b",   # amber
}

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="N-Body HPC Benchmark Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n",         type=int,  default=16384, help="Number of particles")
    p.add_argument("--steps",     type=int,  default=10,    help="Simulation steps")
    p.add_argument("--mpi-ranks", type=int,  default=None,
                   help="MPI ranks (defaults to nproc)")
    p.add_argument("--skip-cuda", action="store_true", help="Skip CUDA benchmark")
    p.add_argument("--skip-mpi",  action="store_true", help="Skip MPI benchmark")
    p.add_argument("--out-csv",   type=str,  default="timing_results.csv")
    p.add_argument("--out-png",   type=str,  default="timing_results.png")
    p.add_argument("--warmup",    action="store_true",
                   help="Run each binary once before timing (already done inside CUDA binary)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Runner helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> Optional[str]:
    """Run a subprocess and return its stdout, or None on failure."""
    print(f"\n  [RUN]  Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,   # 1-hour hard limit
        )
    except FileNotFoundError:
        print(f"  [FAIL]  Binary not found for {label}. "
              f"Did you run 'make {label.lower()}'?")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [FAIL]  {label} timed out.")
        return None

    wall = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [FAIL]  {label} exited with code {result.returncode}")
        print("       stderr:", result.stderr[:500])
        return None

    print(f"  [OK]  {label} finished in {wall:.2f}s (wall clock)")
    print(result.stdout, end="")
    return result.stdout


def _parse_csv_line(stdout: str) -> Optional[dict]:
    """
    Extract the CSV-tagged result line from a binary's stdout.
    Format: CSV,<impl>,<N>,<steps>,<total_ms>[,<extra>]
    """
    for line in stdout.splitlines():
        if line.startswith("CSV,"):
            parts = line.split(",")
            if len(parts) < 5:
                continue
            return {
                "Implementation": parts[1],
                "N":              int(parts[2]),
                "Steps":          int(parts[3]),
                "Total_ms":       float(parts[4]),
                "Extra":          parts[5] if len(parts) > 5 else "",
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-implementation launchers
# ─────────────────────────────────────────────────────────────────────────────

def run_serial(n: int, steps: int) -> Optional[dict]:
    exe = BIN_DIR / "serial"
    out = _run([str(exe), str(n), str(steps)], "Serial")
    return _parse_csv_line(out) if out else None


def run_openmp(n: int, steps: int) -> Optional[dict]:
    import multiprocessing
    nthreads = str(multiprocessing.cpu_count())
    env = {**os.environ, "OMP_NUM_THREADS": nthreads}
    exe = BIN_DIR / "openmp"
    cmd = [str(exe), str(n), str(steps)]
    print(f"\n  [RUN]  Running: OMP_NUM_THREADS={nthreads} {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=3600)
    except FileNotFoundError:
        print("  [FAIL]  bin/openmp not found. Run 'make openmp'.")
        return None
    print(result.stdout, end="")
    return _parse_csv_line(result.stdout) if result.returncode == 0 else None


def run_cuda(n: int, steps: int) -> Optional[dict]:
    exe = BIN_DIR / "cuda"
    out = _run([str(exe), str(n), str(steps)], "CUDA")
    return _parse_csv_line(out) if out else None


def run_mpi(n: int, steps: int, ranks: Optional[int]) -> Optional[dict]:
    import multiprocessing
    if ranks is None:
        # Cap at 8 by default; --oversubscribe lets OpenMPI exceed slot limits
        ranks = min(multiprocessing.cpu_count(), 8)
    exe = BIN_DIR / "mpi"
    out = _run(["mpirun", "--oversubscribe", "-np", str(ranks), str(exe), str(n), str(steps)], "MPI")
    return _parse_csv_line(out) if out else None


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(records: list[dict], path: str) -> None:
    fieldnames = ["Implementation", "N", "Steps", "Total_ms",
                  "PerStep_ms", "Speedup_vs_Serial", "GFLOP_s", "Extra"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  [CSV]  CSV written -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: derive speedup, GFLOP/s
# ─────────────────────────────────────────────────────────────────────────────

def enrich_records(records: list[dict]) -> list[dict]:
    serial_ms = next(
        (r["Total_ms"] for r in records if r["Implementation"] == "Serial"),
        None,
    )
    for r in records:
        n     = r["N"]
        steps = r["Steps"]
        ms    = r["Total_ms"]
        per   = ms / steps
        # ~20 FLOPs per ordered pair (3 sub, 3 mul for dist², 1 sqrt, 3 mul, 3 add = conservative)
        gflops = n * n * 20.0 / per / 1e6

        r["PerStep_ms"]   = round(per, 4)
        r["GFLOP_s"]      = round(gflops, 3)
        r["Speedup_vs_Serial"] = (
            round(serial_ms / ms, 2) if serial_ms else "N/A"
        )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(records: list[dict], out_png: str) -> None:
    if not HAS_MATPLOTLIB:
        print("  [WARN]  matplotlib not installed -- skipping plot.")
        return

    impls  = [r["Implementation"] for r in records]
    ms     = [r["PerStep_ms"]     for r in records]
    gflops = [r["GFLOP_s"]        for r in records]
    colors = [IMPL_COLORS.get(i, "#6b7280") for i in impls]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")

    # ── Common axis style ─────────────────────────────────────────────────
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # ── Panel 1: Per-step time (ms) — lower is better ────────────────────
    bars1 = axes[0].bar(impls, ms, color=colors, linewidth=0,
                        zorder=3, alpha=0.9)
    axes[0].set_title("Time per Step (ms)  — lower is better",
                      fontsize=13, pad=12)
    axes[0].set_ylabel("ms / step", fontsize=11)
    axes[0].grid(axis="y", color="#334155", linewidth=0.7, zorder=0)
    for bar, val in zip(bars1, ms):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(ms) * 0.01,
                     f"{val:.2f}", ha="center", va="bottom",
                     color="white", fontsize=10, fontweight="bold")

    # ── Panel 2: GFLOP/s — higher is better ──────────────────────────────
    bars2 = axes[1].bar(impls, gflops, color=colors, linewidth=0,
                        zorder=3, alpha=0.9)
    axes[1].set_title("Effective Throughput (GFLOP/s)  — higher is better",
                      fontsize=13, pad=12)
    axes[1].set_ylabel("GFLOP/s", fontsize=11)
    axes[1].grid(axis="y", color="#334155", linewidth=0.7, zorder=0)
    for bar, val in zip(bars2, gflops):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(gflops) * 0.01,
                     f"{val:.1f}", ha="center", va="bottom",
                     color="white", fontsize=10, fontweight="bold")

    # ── Speedup annotation ────────────────────────────────────────────────
    speedup_text = "  |  ".join(
        f"{r['Implementation']}: {r['Speedup_vs_Serial']}×"
        for r in records if r["Implementation"] != "Serial"
    )
    fig.text(0.5, 0.01, f"Speedup vs Serial  →  {speedup_text}",
             ha="center", color="#94a3b8", fontsize=10)

    n_val = records[0]["N"] if records else "?"
    s_val = records[0]["Steps"] if records else "?"
    fig.suptitle(
        f"N-Body Gravitational Simulation  |  N={n_val:,}  |  O(N²) Brute Force",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [PLOT]  Plot saved -> {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  N-Body HPC Benchmark Suite")
    print(f"  N={args.n}  steps={args.steps}")
    print("=" * 60)

    results = []

    # ── Serial ────────────────────────────────────────────────────────────
    r = run_serial(args.n, args.steps)
    if r: results.append(r)

    # ── OpenMP ────────────────────────────────────────────────────────────
    r = run_openmp(args.n, args.steps)
    if r: results.append(r)

    # ── CUDA ──────────────────────────────────────────────────────────────
    if not args.skip_cuda:
        r = run_cuda(args.n, args.steps)
        if r: results.append(r)

    # ── MPI ───────────────────────────────────────────────────────────────
    if not args.skip_mpi:
        r = run_mpi(args.n, args.steps, args.mpi_ranks)
        if r: results.append(r)

    if not results:
        print("\n  [FAIL]  No results collected. Check that binaries are built.")
        sys.exit(1)

    # ── Enrich + write ────────────────────────────────────────────────────
    results = enrich_records(results)
    write_csv(results, args.out_csv)

    # ── Console table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    headers = ["Implementation", "N", "Steps",
               "Total_ms", "PerStep_ms", "Speedup_vs_Serial", "GFLOP_s"]
    rows = [[r[h] for h in headers] for r in results]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline",
                       floatfmt=".3f"))
    else:
        for row in rows:
            print("  ", "  ".join(str(v) for v in row))

    # ── Plot ──────────────────────────────────────────────────────────────
    make_plot(results, args.out_png)

    print("\n  [DONE]  Benchmark complete.\n")


if __name__ == "__main__":
    main()
