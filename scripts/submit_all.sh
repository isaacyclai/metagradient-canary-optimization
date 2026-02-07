#!/bin/bash
# ==============================================================================
# Submit all experiment jobs to SLURM
# ==============================================================================
# Usage:
#   ./scripts/submit_all.sh           # Submit all experiments
#   ./scripts/submit_all.sh sgd       # Submit only SGD experiments
#   ./scripts/submit_all.sh dpsgd     # Submit only DP-SGD experiments
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Create necessary directories
mkdir -p logs results

MODE="${1:-all}"

submit_sgd_jobs() {
    echo "Submitting SGD audit jobs..."
    
    # WRN 16-4
    echo "  - WRN 16-4"
    sbatch scripts/run_audit_sgd.sh
    
    # ResNet-9
    echo "  - ResNet-9"
    MODEL=resnet9 sbatch scripts/run_audit_sgd.sh
}

submit_dpsgd_jobs() {
    echo "Submitting DP-SGD audit jobs..."
    
    # Different epsilon values with WRN 16-4
    for eps in 1.0 2.0 4.0 8.0; do
        echo "  - WRN 16-4, epsilon=$eps"
        MODEL=wrn16_4 EPSILON=$eps sbatch scripts/run_audit_dpsgd.sh
    done
    
    # Different epsilon values with ResNet-9
    for eps in 1.0 2.0 4.0 8.0; do
        echo "  - ResNet-9, epsilon=$eps"
        MODEL=resnet9 EPSILON=$eps sbatch scripts/run_audit_dpsgd.sh
    done
}

case "$MODE" in
    sgd)
        submit_sgd_jobs
        ;;
    dpsgd)
        submit_dpsgd_jobs
        ;;
    all)
        submit_sgd_jobs
        echo ""
        submit_dpsgd_jobs
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [sgd|dpsgd|all]"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted! Check status with: squeue -u \$USER"
