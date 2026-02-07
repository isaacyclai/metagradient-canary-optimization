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

