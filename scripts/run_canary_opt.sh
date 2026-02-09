#!/bin/bash
#SBATCH --job-name=canary_opt
#SBATCH --output=logs/canary_opt_%j.out
#SBATCH --error=logs/canary_opt_%j.err
#SBATCH --time=24:00:00
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

# Create directories
mkdir -p logs results checkpoints

# Activate virtual environment (adjust path as needed)
source .venv/bin/activate

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Default parameters (can be overridden by environment variables)
NUM_CANARIES="${NUM_CANARIES:-1000}"
META_STEPS="${META_STEPS:-50}"
EPOCHS_PER_STEP="${EPOCHS_PER_STEP:-12}"
CANARY_LR="${CANARY_LR:-0.01}"
MODEL_LR="${MODEL_LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT="${OUTPUT:-results/optimized_canaries_${SLURM_JOB_ID}.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/job_${SLURM_JOB_ID}}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
LOG_FILE="${LOG_FILE:-logs/canary_opt_${SLURM_JOB_ID}.json}"

# Create output directories
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$CHECKPOINT_DIR"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=============================================="
echo "Parameters:"
echo "  Meta steps: $META_STEPS"
echo "  Epochs per step: $EPOCHS_PER_STEP"
echo "  Num canaries: $NUM_CANARIES"
echo "  Canary LR: $CANARY_LR"
echo "  Model LR: $MODEL_LR"
echo "Output:"
echo "  Canaries: $OUTPUT"
echo "  Checkpoints: $CHECKPOINT_DIR (every $CHECKPOINT_INTERVAL steps)"
echo "  Log file: $LOG_FILE"
echo "=============================================="

# Build command
CMD="uv run experiments/run_canary_opt.py \
    --num-canaries $NUM_CANARIES \
    --meta-steps $META_STEPS \
    --epochs-per-step $EPOCHS_PER_STEP \
    --canary-lr $CANARY_LR \
    --model-lr $MODEL_LR \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --data-dir $DATA_DIR \
    --output $OUTPUT \
    --checkpoint-dir $CHECKPOINT_DIR \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --log-file $LOG_FILE"

echo "Command: $CMD"
echo "=============================================="

# Run the experiment
$CMD

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
