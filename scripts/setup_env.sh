#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Install all HPC dependencies for the N-Body benchmark suite
# Target: Ubuntu 20.04 / 22.04 / Debian Bookworm
# Run as root or with sudo:  sudo bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

# ── Ensure running as root ────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[ERROR]${NC} Please run as root: sudo bash $0"
    exit 1
fi

section "1 / 5  System update"
apt-get update -qq
apt-get upgrade -y -qq

section "2 / 5  Core build tools"
apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    pkg-config \
    numactl \
    linux-tools-common \
    linux-tools-generic
info "Core tools installed."

section "3 / 5  OpenMP"
# libomp-dev provides the LLVM OpenMP runtime headers; libgomp is bundled with GCC.
apt-get install -y -qq libomp-dev
info "OpenMP installed  (libomp-dev + libgomp via GCC)."

section "4 / 5  Open MPI"
apt-get install -y -qq \
    openmpi-bin \
    libopenmpi-dev \
    openmpi-common \
    libpmi2-0-dev
info "OpenMPI installed."
mpirun --version | head -1

section "5 / 5  NVIDIA CUDA Toolkit"
# Check whether a GPU is present
if command -v nvidia-smi &>/dev/null; then
    info "nvidia-smi found — proceeding with CUDA install."
else
    warn "nvidia-smi not detected. Installing CUDA toolkit anyway (nvcc only)."
fi

# Add NVIDIA CUDA repository keyring
wget -q "$CUDA_KEYRING_URL" -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
apt-get update -qq

# Install the toolkit (nvcc, libraries, headers) without driver
apt-get install -y -qq nvidia-cuda-toolkit

info "CUDA toolkit installed."
nvcc --version | head -4 || warn "nvcc not in PATH — add /usr/local/cuda/bin to PATH."

# ── Python benchmark dependencies ─────────────────────────────────────────────
section "Python dependencies"
if command -v python3 &>/dev/null; then
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet matplotlib pandas tabulate
    info "Python packages installed (matplotlib, pandas, tabulate)."
else
    warn "python3 not found — skipping Python deps."
fi

# ── PATH hints ────────────────────────────────────────────────────────────────
echo ""
echo "  Add these lines to your ~/.bashrc if not already present:"
echo '  export PATH=/usr/local/cuda/bin:$PATH'
echo '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
echo ""

info "All dependencies installed successfully."
echo ""
echo "  Next steps:"
echo "    cd $(dirname "$(realpath "$0")")/.."
echo "    make all"
echo "    make bench"
