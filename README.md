# Metagradient Canary Optimization

Attempted replication of [Optimizing Canaries for Privacy Auditing with Metagradient Descent](https://arxiv.org/pdf/2507.15836) (Boglioni et al., 2025).

## Setup

```bash
uv sync
```

## Workflow

The typical workflow to replicate the paper results:

1. **Optimize canaries** using metagradient descent (Algorithm 5)
2. **Audit models** with optimized canaries to measure privacy leakage

## Quick Test Commands (Verify Setup)

Use these commands to verify everything works before running full experiments:

```bash
# Test canary optimization (~3-5 min)
uv run python experiments/run_canary_opt.py \
    --meta-steps 2 --epochs-per-step 2 --num-canaries 50 \
    --output test_canaries.pt

# Test SGD audit (~1-2 min)
uv run python experiments/run_audit_sgd.py \
    --steps 100 --eval-interval 50 --num-canaries 50 --seeds 1 --model resnet9

# Test DP-SGD audit with Opacus (~2-5 min)
uv run python experiments/run_audit_dpsgd.py \
    --epochs 2 --epsilon 8.0 --num-canaries 50 --seeds 1 --model resnet9

# Test DP-SGD audit with JAX (~2-5 min)
uv run python experiments/run_audit_dpsgd_jax.py \
    --epochs 2 --epsilon 8.0 --num-canaries 50 --batch-size 512 --seeds 1
```

## Full Experiments

### 1. Optimize Canaries (Algorithm 5)

Optimizes canary samples to maximize the loss gap between C_IN and C_OUT.

```bash
uv run python experiments/run_canary_opt.py \
    --num-canaries 1000 \
    --meta-steps 50 \
    --epochs-per-step 12 \
    --canary-lr 0.01 \
    --model-lr 0.1 \
    --output optimized_canaries.pt
```

**Key parameters:**
- `--meta-steps`: Number of optimization iterations (paper uses 50)
- `--epochs-per-step`: Full training epochs per iteration (paper uses 12)
- `--num-canaries`: Number of canary samples (paper uses 1000)

### 2. Audit Non-DP SGD (Figure 2)

Compare canary types (random, mislabeled, metagradient) for auditing standard SGD.

```bash
uv run python experiments/run_audit_sgd.py \
    --steps 10000 \
    --eval-interval 100 \
    --num-canaries 1000 \
    --seeds 5 \
    --model wrn16_4 \
    --canary-path optimized_canaries.pt
```

### 3. Audit DP-SGD (Table 2)

Audit differentially private models at various epsilon values.

```bash
uv run python experiments/run_audit_dpsgd.py \
    --epsilon 8.0 \
    --delta 1e-5 \
    --epochs 40 \
    --num-canaries 1000 \
    --seeds 5 \
    --model wrn16_4 \
    --canary-path optimized_canaries.pt
```

**Epsilon sweep:**
```bash
for eps in 1.0 2.0 4.0 8.0; do
    uv run python experiments/run_audit_dpsgd.py --epsilon $eps --delta 1e-5 --seeds 5
done
```

### 4. Audit DP-SGD with JAX (Recommended)

The original paper uses jax-privacy. This version is more memory-efficient and matches the paper.
Includes both **Steinke et al. (2023)** and **Mahloujifar et al. (2024)** auditing methods.

**First-time setup (on cluster):**
```bash
uv sync --extra jax
```

**Run audit:**
```bash
uv run python experiments/run_audit_dpsgd_jax.py \
    --epsilon 8.0 \
    --delta 1e-5 \
    --epochs 100 \
    --batch-size 4096 \
    --model wrn16_4 \
    --num-canaries 1000 \
    --seeds 5 \
    --canary-path optimized_canaries.pt
```

**Models:**
- `wrn16_4` (default): Wide ResNet 16-4 per De et al. (2022) - matches paper
- `resnet9`: Faster but different from paper

**Auditing methods:**
- **Steinke (Algorithm 2):** Unpaired membership inference with binomial statistics
- **Mahloujifar (Algorithm 3+4):** Paired IN/OUT comparison with f-DP bounds

**Output format (per seed):**
```
Achieved epsilon: 7.97          # Theoretical guarantee
Steinke epsilon: 2.15          # Empirical lower bound (Steinke)
Mahloujifar epsilon: 1.89      # Empirical lower bound (Mahloujifar)
Loss gap (IN-OUT): -0.0606
```

**SLURM:**
```bash
# Single epsilon value
sbatch scripts/run_audit_dpsgd_jax.sh

# With optimized canaries
CANARY_PATH=optimized_canaries.pt sbatch scripts/run_audit_dpsgd_jax.sh
```

**Replicate Table 2 (full epsilon sweep):**
```bash
# Step 1: Generate optimized canaries first
sbatch scripts/run_canary_opt.sh

# Step 2: Run DP-SGD audit at ε=8.0 for each canary type
CANARY_PATH=optimized_canaries.pt sbatch scripts/run_audit_dpsgd_jax.sh
```

## SLURM Cluster Usage

### Submit Jobs

```bash
# Canary optimization (run first)
sbatch scripts/run_canary_opt.sh

# SGD audit
sbatch scripts/run_audit_sgd.sh

# DP-SGD audit
sbatch scripts/run_audit_dpsgd.sh
```

### Configure via Environment Variables

```bash
# Use ResNet-9 instead of WRN-16-4
MODEL=resnet9 sbatch scripts/run_audit_sgd.sh

# DP-SGD with different epsilon
EPSILON=4.0 MODEL=resnet9 sbatch scripts/run_audit_dpsgd.sh

# Custom canary optimization
META_STEPS=100 EPOCHS_PER_STEP=12 NUM_CANARIES=1000 sbatch scripts/run_canary_opt.sh
```

### Quick Tests on Cluster

```bash
# Test canary optimization (~3-5 min)
META_STEPS=2 EPOCHS_PER_STEP=2 NUM_CANARIES=50 sbatch scripts/run_canary_opt.sh

# Test SGD audit (~1-2 min)
STEPS=100 EVAL_INTERVAL=50 NUM_CANARIES=50 SEEDS=1 MODEL=resnet9 sbatch scripts/run_audit_sgd.sh

# Test DP-SGD audit with Opacus (~2-5 min)  
EPOCHS=2 NUM_CANARIES=50 SEEDS=1 BATCH_SIZE=512 MODEL=resnet9 sbatch scripts/run_audit_dpsgd.sh

# Test DP-SGD audit with JAX (~2-5 min)
EPOCHS=2 NUM_CANARIES=50 SEEDS=1 BATCH_SIZE=512 sbatch scripts/run_audit_dpsgd_jax.sh
```

### Interactive GPU Session

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
source .venv/bin/activate
uv run python experiments/run_canary_opt.py --meta-steps 2 --epochs-per-step 2 --num-canaries 50
```

## Models

- `wrn16_4`: Wide ResNet 16-4 (default, used in paper)
- `resnet9`: ResNet-9 (faster, good for testing)

## Output Files

- `optimized_canaries.pt`: Optimized canary images and labels
- `results/`: Experiment results (JSON format)
- `logs/`: Training logs
