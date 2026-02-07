# Metagradient Canary Optimization

Attempted replication of [Optimizing Canaries for Privacy Auditing with Metagradient Descent](https://arxiv.org/pdf/2507.15836) (Boglioni et al., 2025).

## Setup

```bash
uv sync
```

## Usage

### 1. Optimize Canaries (Algorithm 5)

```bash
uv run python experiments/run_canary_opt.py --num-canaries 1000 --meta-steps 50
```

### 2. Audit Non-DP SGD (Figure 2)

```bash
uv run python experiments/run_audit_sgd.py --steps 10000 --eval-interval 100
```

### 3. Audit DP-SGD (Table 2)

```bash
uv run python experiments/run_audit_dpsgd.py --epsilon 8.0 --delta 1e-5 --seeds 5
```

## SLURM Cluster Usage

Submit jobs to a SLURM cluster using the provided scripts:

```bash
# SGD audit
sbatch scripts/run_audit_sgd.sh

# DP-SGD audit
sbatch scripts/run_audit_dpsgd.sh

# Canary optimization
sbatch scripts/run_canary_opt.sh
```

Configure experiments via environment variables:

```bash
MODEL=resnet9 sbatch scripts/run_audit_sgd.sh
EPSILON=4.0 MODEL=resnet9 sbatch scripts/run_audit_dpsgd.sh
```

### Quick Test Commands

Use smaller values to verify scripts work before full runs:

```bash
# Test SGD audit (~1-2 min)
STEPS=100 EVAL_INTERVAL=50 NUM_CANARIES=50 SEEDS=1 MODEL=resnet9 sbatch scripts/run_audit_sgd.sh

# Test DP-SGD audit (~2-5 min)
EPOCHS=2 NUM_CANARIES=50 SEEDS=1 BATCH_SIZE=512 MODEL=resnet9 sbatch scripts/run_audit_dpsgd.sh

# Test canary optimization (~3-5 min)
META_STEPS=2 EPOCHS_PER_STEP=2 NUM_CANARIES=50 sbatch scripts/run_canary_opt.sh
```

### Interactive Testing

```bash
# Request an interactive GPU session
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

# Run experiments directly
uv run experiments/run_audit_sgd.py --steps 100 --eval-interval 50 --num-canaries 50 --seeds 1 --model resnet9
```

