#!/bin/bash
#SBATCH --job-name=audit_sgd
#SBATCH --output=logs/audit_sgd_%j.out
#SBATCH --error=logs/audit_sgd_%j.err
#SBATCH --time=10
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# change time to 12:00:00 later

# ==============================================================================
# SLURM Job Script: Audit non-DP SGD (Figure 2)
# ==============================================================================
# Usage:
#   sbatch scripts/run_audit_sgd.sh
#   sbatch scripts/run_audit_sgd.sh --model resnet9
#
# Configurable via environment variables (optional):
#   MODEL=resnet9 sbatch scripts/run_audit_sgd.sh
# ==============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust for your cluster)
module load python/3.12
module load cuda/12.9

# Activate virtual environment (adjust path as needed)
source .venv/bin/activate

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Default parameters (can be overridden by environment variables)
MODEL="${MODEL:-wrn16_4}"
STEPS="${STEPS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_CANARIES="${NUM_CANARIES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEEDS="${SEEDS:-5}"
DATA_DIR="${DATA_DIR:-./data}"
CANARY_PATH="${CANARY_PATH:-}"
OUTPUT="${OUTPUT:-results/figure2_${MODEL}_${SLURM_JOB_ID}.json}"

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "=============================================="

# Build command
CMD="uv run experiments/run_audit_sgd.py \
    --model $MODEL \
    --steps $STEPS \
    --eval-interval $EVAL_INTERVAL \
    --num-canaries $NUM_CANARIES \
    --batch-size $BATCH_SIZE \
    --seeds $SEEDS \
    --data-dir $DATA_DIR \
    --output $OUTPUT"

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
