#!/bin/bash
#SBATCH --job-name=adc-worker
#SBATCH --account=def-YOUR_ACCOUNT    # <-- Replace with your DRAC allocation
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # Request 1 GPU (see GPU options below)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Example SLURM job script for running the ADC Antenna worker on DRAC Alliance
# HPC clusters (Fir, Rorqual, Narval, etc.).
#
# GPU options (adjust --gres):
#   Fir/Rorqual: gpu:h100:1
#   Narval:      gpu:a100:1
#   Cedar:       gpu:v100l:1
#
# ──────────────────────────────────────────────────────────────────────
# One-time setup (run interactively, NOT in this script):
#
#   module load python/3.12 cuda/12.6
#   uv venv ~/venvs/adc --python 3.12
#   source ~/venvs/adc/bin/activate
#   cd ~/projects/ami-data-companion
#   # --no-deps: let the lockfile control versions; avoids conflicts with
#   #            system packages on DRAC nodes.
#   uv pip install --no-deps -r <(uv export --no-hashes --frozen)
#   uv pip install --no-deps .
#
#   # Create .env with your Antenna credentials:
#   cat > .env <<EOF
#   AMI_ANTENNA_API_BASE_URL=https://antenna.insectai.org/api/v2
#   AMI_ANTENNA_API_AUTH_TOKEN=your_token_here
#   AMI_ANTENNA_API_BATCH_SIZE=4
#   AMI_ANTENNA_SERVICE_NAME="DRAC Worker"
#   EOF
#
#   # Set service name and register worker with Antenna (once):
#   export AMI_ANTENNA_SERVICE_NAME="DRAC Worker"
#   ami worker register
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Catch unedited placeholder in #SBATCH --account
if [[ "${SLURM_JOB_ACCOUNT:-}" == *YOUR_ACCOUNT* ]]; then
    echo "ERROR: Replace --account in this script with your DRAC allocation." >&2
    exit 1
fi

module load python/3.12 cuda/12.6

source ~/venvs/adc/bin/activate
cd ~/projects/ami-data-companion
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in $(pwd). See one-time setup instructions in this script." >&2
    exit 1
fi
set -a; source .env; set +a

# Register pipelines on each run (idempotent).
# Service name is read from AMI_ANTENNA_SERVICE_NAME in .env (hostname appended automatically).
ami worker register

# Run with timeout slightly shorter than SLURM --time to allow clean shutdown.
# The worker handles SIGTERM gracefully and finishes the current batch.
# Exit code 124 from timeout is expected (time limit reached), not an error.
timeout --signal=SIGTERM 11h55m ami worker \
    || { rc=$?; [[ $rc -eq 124 ]] && echo "Worker reached time limit — exiting cleanly." && exit 0; exit $rc; }
