#!/bin/bash
#SBATCH --job-name=canary_opt
#SBATCH --output=logs/canary_opt_%j.out
#SBATCH --error=logs/canary_opt_%j.err
#SBATCH --time=10
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# change time to 24:00:00 later

# ==============================================================================
# SLURM Job Script: Canary Optimization (Algorithm 5)
# ==============================================================================
# Usage:
#   sbatch scripts/run_canary_opt.sh
#   META_STEPS=100 sbatch scripts/run_canary_opt.sh
# ==============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment (adjust path as needed)
source .venv/bin/activate

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Default parameters (can be overridden by environment variables)
NUM_CANARIES="${NUM_CANARIES:-1000}"
META_STEPS="${META_STEPS:-50}"
EPOCHS_PER_STEP="${EPOCHS_PER_STEP:-12}"
CANARY_LR="${CANARY_LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT="${OUTPUT:-results/optimized_canaries_${SLURM_JOB_ID}.pt}"
LOG_DIR="${LOG_DIR:-logs}"

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Meta steps: $META_STEPS"
echo "Output: $OUTPUT"
echo "=============================================="

# Build command
CMD="uv run experiments/run_canary_opt.py \
    --num-canaries $NUM_CANARIES \
    --meta-steps $META_STEPS \
    --epochs-per-step $EPOCHS_PER_STEP \
    --canary-lr $CANARY_LR \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --data-dir $DATA_DIR \
    --output $OUTPUT \
    --log-dir $LOG_DIR"

echo "Command: $CMD"
echo "=============================================="

# Run the experiment
$CMD

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
