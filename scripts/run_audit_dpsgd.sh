#!/bin/bash
#SBATCH --job-name=audit_dpsgd
#SBATCH --output=logs/audit_dpsgd_%j.out
#SBATCH --error=logs/audit_dpsgd_%j.err
#SBATCH --time=10
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# change time to 24:00:00 later

# ==============================================================================
# SLURM Job Script: Audit DP-SGD (Table 2)
# ==============================================================================
# Usage:
#   sbatch scripts/run_audit_dpsgd.sh
#   MODEL=resnet9 EPSILON=4.0 sbatch scripts/run_audit_dpsgd.sh
# ==============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment (adjust path as needed)
source .venv/bin/activate

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Default parameters (can be overridden by environment variables)
MODEL="${MODEL:-wrn16_4}"
EPSILON="${EPSILON:-8.0}"
DELTA="${DELTA:-1e-5}"
EPOCHS="${EPOCHS:-100}"
NUM_CANARIES="${NUM_CANARIES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SEEDS="${SEEDS:-5}"
DATA_DIR="${DATA_DIR:-./data}"
CANARY_PATH="${CANARY_PATH:-}"
OUTPUT="${OUTPUT:-results/table2_${MODEL}_eps${EPSILON}_${SLURM_JOB_ID}.json}"

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Model: $MODEL"
echo "Epsilon: $EPSILON, Delta: $DELTA"
echo "Output: $OUTPUT"
echo "=============================================="

# Build command
CMD="uv run experiments/run_audit_dpsgd.py \
    --model $MODEL \
    --epsilon $EPSILON \
    --delta $DELTA \
    --epochs $EPOCHS \
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
