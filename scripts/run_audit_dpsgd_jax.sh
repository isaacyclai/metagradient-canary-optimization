#!/bin/bash
#SBATCH --job-name=dpsgd_jax
#SBATCH --output=logs/dpsgd_jax_%j.out
#SBATCH --error=logs/dpsgd_jax_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1 -C cuda80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# ==============================================================================
# SLURM Job Script: DP-SGD Audit with JAX
# ==============================================================================
# This script uses JAX-based DP-SGD which matches the original paper's
# jax-privacy implementation and handles larger batch sizes more efficiently.
#
# Usage:
#   sbatch scripts/run_audit_dpsgd_jax.sh
#   EPSILON=4.0 sbatch scripts/run_audit_dpsgd_jax.sh
#
# ==============================================================================

set -e

# Create directories
mkdir -p logs results

# Activate virtual environment
source .venv/bin/activate

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Default parameters (can be overridden by environment variables)
EPSILON="${EPSILON:-8.0}"
DELTA="${DELTA:-1e-5}"
EPOCHS="${EPOCHS:-100}"
NUM_CANARIES="${NUM_CANARIES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MODEL="${MODEL:-wrn16_4}"
NOISE_MULTIPLIER="${NOISE_MULTIPLIER:-1.75}"
SEEDS="${SEEDS:-5}"
DATA_DIR="${DATA_DIR:-./data}"
CANARY_PATH="${CANARY_PATH:-}"
OUTPUT="${OUTPUT:-results/dpsgd_jax_eps${EPSILON}_${SLURM_JOB_ID}.json}"

echo "=============================================="
echo "DP-SGD Audit (JAX Implementation)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=============================================="
echo "Parameters:"
echo "  Epsilon: $EPSILON"
echo "  Delta: $DELTA"
echo "  Model: $MODEL"
echo "  Noise multiplier: $NOISE_MULTIPLIER"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Num canaries: $NUM_CANARIES"
echo "  Seeds: $SEEDS"
echo "Output: $OUTPUT"
echo "=============================================="

# Check JAX installation
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Build command
CMD="uv run experiments/run_audit_dpsgd_jax.py \
    --epsilon $EPSILON \
    --delta $DELTA \
    --epochs $EPOCHS \
    --num-canaries $NUM_CANARIES \
    --batch-size $BATCH_SIZE \
    --model $MODEL \
    --seeds $SEEDS \
    --data-dir $DATA_DIR \
    --output $OUTPUT"

# Add noise multiplier if specified
if [ -n "$NOISE_MULTIPLIER" ]; then
    CMD="$CMD --noise-multiplier $NOISE_MULTIPLIER"
fi

# Add canary path if specified
if [ -n "$CANARY_PATH" ]; then
    CMD="$CMD --canary-path $CANARY_PATH"
fi

echo "Command: $CMD"
echo "=============================================="

# Run the experiment
$CMD

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
