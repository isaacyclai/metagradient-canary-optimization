"""
Run DP-SGD privacy audit using JAX/
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')

import numpy as np
from scipy import special
from scipy.optimize import brentq

import jax
import jax.numpy as jnp
import optax

from src.models.resnet9_jax import create_resnet9
from src.models.wrn_jax import create_wrn16_4
from src.algorithms.dp_sgd_jax import train_dpsgd_jax, evaluate_canary_losses


def get_model(model_name: str, num_classes: int = 10):
    """Get JAX model by name."""
    if model_name == 'wrn16_4':
        return create_wrn16_4(num_classes)
    elif model_name == 'resnet9':
        return create_resnet9(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_cifar10_numpy(data_dir: str = "./data"):
    """Load CIFAR-10 as numpy arrays (NHWC format for JAX)."""
    import torchvision
    import torchvision.transforms as transforms
    
    # Load without transforms
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True
    )
    
    # Convert to numpy, normalize to [0, 1]
    train_images = train_dataset.data.astype(np.float32) / 255.0
    train_labels = np.array(train_dataset.targets)
    
    test_images = test_dataset.data.astype(np.float32) / 255.0
    test_labels = np.array(test_dataset.targets)
    
    return (train_images, train_labels), (test_images, test_labels)


# ============================================================================
# Steinke Auditing
# ============================================================================

def theorem1_bound(r: int, v: int, epsilon: float, delta: float, m: int) -> float:
    """Compute probability bound from Steinke et al. Theorem 1."""
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    
    # Use survival function of binomial
    if v > 0 and v <= r:
        f_v = 1 - special.betainc(v, r - v + 1, p)
    else:
        f_v = 1.0 if v <= 0 else 0.0
    
    # Compute correction term for delta
    correction = 0
    if delta > 0 and m > 0:
        max_term = 0
        for i in range(1, min(m + 1, v + 1)):
            if v - i > 0:
                f_vi = 1 - special.betainc(v - i, r - v + i + 1, p)
            else:
                f_vi = 1.0
            f_v_curr = 1 - special.betainc(v, r - v + 1, p) if v > 0 and v <= r else 0.0
            term = (f_vi - f_v_curr) / i if i > 0 else 0
            max_term = max(max_term, term)
        correction = 2 * m * delta * max_term
    
    return f_v + correction


def find_empirical_epsilon(
    num_correct: int,
    num_guesses: int,
    num_canaries: int,
    tau: float = 0.05,
    delta: float = 1e-5,
    epsilon_max: float = 20.0
) -> float:
    """Binary search for the largest ε satisfying Theorem 1."""
    if num_correct <= num_guesses // 2:
        return 0.0
    
    def objective(eps):
        bound = theorem1_bound(num_guesses, num_correct, eps, delta, num_canaries)
        return bound - tau
    
    try:
        eps_low, eps_high = 0.0, epsilon_max
        
        if objective(eps_low) < 0:
            return 0.0
        if objective(eps_high) > 0:
            return epsilon_max
        
        epsilon = brentq(objective, eps_low, eps_high)
        return epsilon
    except ValueError:
        return 0.0


def audit_from_losses(
    losses: np.ndarray,
    in_mask: np.ndarray,
    k_plus: int = None,
    k_minus: int = None,
    tau: float = 0.05,
    delta: float = 1e-5
) -> dict:
    """Steinke auditing using precomputed losses."""
    m = len(losses)
    scores = -losses  # Higher score = lower loss = more memorized
    
    if k_plus is None:
        k_plus = m // 4
    if k_minus is None:
        k_minus = m // 4
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    # Make guesses
    guesses = np.zeros(m, dtype=int)
    guesses[sorted_indices[:k_plus]] = 1      # Top scores -> guess IN
    guesses[sorted_indices[-k_minus:]] = -1   # Bottom scores -> guess OUT
    
    # Count correct
    true_labels = in_mask.astype(int) * 2 - 1  # True->1, False->-1
    correct = ((guesses == 1) & (true_labels == 1)) | ((guesses == -1) & (true_labels == -1))
    num_correct = correct.sum()
    num_guesses = k_plus + k_minus
    
    # Find empirical epsilon
    emp_epsilon = find_empirical_epsilon(num_correct, num_guesses, m, tau, delta)
    
    return {
        "steinke_epsilon": emp_epsilon,
        "steinke_correct": int(num_correct),
        "steinke_guesses": num_guesses,
        "steinke_accuracy": num_correct / num_guesses if num_guesses > 0 else 0.0
    }


# ============================================================================
# Mahloujifar Auditing (Algorithm 3+4)
# ============================================================================

def fdp_inverse(y: float, epsilon: float) -> float:
    """Inverse of f-DP trade-off function."""
    return y / np.exp(epsilon)


def algorithm4_bound(
    m: int, k: int, k_prime: int, tau: float, epsilon: float, s: int = 2
) -> bool:
    """Algorithm 4: Check if k' correct guesses is plausible under epsilon-DP."""
    if k_prime <= 0:
        return True
    
    # Initialize arrays
    h = np.zeros(k_prime + 1)
    r = np.zeros(k_prime + 1)
    
    r[k_prime] = tau * tau / m if m > 0 else tau
    h[k_prime] = 0
    
    # Backward iteration
    for i in range(k_prime - 1, -1, -1):
        h[i] = (s - 1) * fdp_inverse(r[i + 1], epsilon)
        if k > i:
            r[i] = r[i + 1] + (i / (k - i)) * (h[i] - h[i + 1]) if k != i else r[i + 1]
        else:
            r[i] = r[i + 1]
    
    return r[0] + h[0] >= k / m if m > 0 else True


def find_empirical_epsilon_mahloujifar(
    num_correct: int,
    num_guesses: int,
    num_pairs: int,
    tau: float = 0.05,
    epsilon_max: float = 20.0
) -> float:
    """Binary search for empirical epsilon using Mahloujifar Algorithm 4."""
    if num_correct <= num_guesses // 2:
        return 0.0
    
    def passes_test(eps):
        return algorithm4_bound(num_pairs, num_guesses, num_correct, tau, eps)
    
    eps_low, eps_high = 0.0, epsilon_max
    
    if passes_test(eps_high):
        return epsilon_max
    if not passes_test(eps_low):
        return 0.0
    
    while eps_high - eps_low > 0.01:
        eps_mid = (eps_low + eps_high) / 2
        if passes_test(eps_mid):
            eps_low = eps_mid
        else:
            eps_high = eps_mid
    
    return eps_low


def audit_mahloujifar_from_losses(
    losses: np.ndarray,
    in_mask: np.ndarray,
    k: int = None,
    tau: float = 0.05,
    seed: int = 42
) -> dict:
    """Mahloujifar auditing using paired canary comparison."""
    rng = np.random.RandomState(seed)
    scores = -losses  # Higher = more memorized
    
    # Create IN/OUT pairs
    in_indices = np.where(in_mask)[0]
    out_indices = np.where(~in_mask)[0]
    
    rng.shuffle(in_indices)
    rng.shuffle(out_indices)
    
    num_pairs = min(len(in_indices), len(out_indices))
    pairs = list(zip(in_indices[:num_pairs], out_indices[:num_pairs]))
    
    if k is None:
        k = max(1, num_pairs // 4)
    
    # Compute score differences for each pair
    diffs = [abs(scores[in_idx] - scores[out_idx]) for in_idx, out_idx in pairs]
    sorted_pair_indices = np.argsort(diffs)[::-1]
    
    # Make guesses on top k pairs (by score difference)
    num_correct = 0
    for i in range(min(k, len(pairs))):
        pair_idx = sorted_pair_indices[i]
        in_idx, out_idx = pairs[pair_idx]
        if scores[in_idx] > scores[out_idx]:
            num_correct += 1
    
    emp_epsilon = find_empirical_epsilon_mahloujifar(num_correct, k, num_pairs, tau)
    
    return {
        "mahloujifar_epsilon": emp_epsilon,
        "mahloujifar_correct": num_correct,
        "mahloujifar_guesses": k,
        "mahloujifar_pairs": num_pairs,
        "mahloujifar_accuracy": num_correct / k if k > 0 else 0.0
    }


def run_dp_audit_jax(
    canary_type: str,
    canary_images: np.ndarray,
    canary_labels: np.ndarray,
    train_data: tuple,
    target_epsilon: float,
    target_delta: float,
    num_epochs: int,
    batch_size: int,
    model_name: str = 'wrn16_4',
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """Run a single DP-SGD audit experiment with JAX."""
    # Random IN/OUT split
    rng = np.random.RandomState(seed)
    m = len(canary_images)
    perm = rng.permutation(m)
    in_mask = np.zeros(m, dtype=bool)
    in_mask[perm[:m//2]] = True
    
    # Create model (WRN-16-4 by default per paper)
    model = get_model(model_name, num_classes=10)
    
    # Train with DP-SGD
    params, achieved_epsilon, train_stats = train_dpsgd_jax(
        model=model,
        train_data=train_data,
        canary_images=canary_images,
        canary_labels=canary_labels,
        in_mask=in_mask,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose
    )
    
    # Evaluate canary losses
    canary_stats = evaluate_canary_losses(
        model, params,
        canary_images, canary_labels, in_mask
    )
    
    # Compute per-canary losses for auditing
    images = jnp.array(canary_images)
    labels = jnp.array(canary_labels)
    logits = model.apply({'params': params}, images, train=False)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    losses_np = np.array(losses)
    
    # Steinke auditing
    steinke_stats = audit_from_losses(losses_np, in_mask, tau=0.05, delta=target_delta)
    
    # Mahloujifar auditing
    mahloujifar_stats = audit_mahloujifar_from_losses(losses_np, in_mask, tau=0.05, seed=seed)
    
    return {
        "canary_type": canary_type,
        "achieved_epsilon": achieved_epsilon,
        "target_epsilon": target_epsilon,
        "in_mask": in_mask.tolist(),
        **canary_stats,
        **steinke_stats,
        **mahloujifar_stats,
        **train_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Audit DP-SGD with JAX (paper-faithful)")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num-canaries", type=int, default=1000, help="Number of canaries")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--model", type=str, default="wrn16_4", choices=["wrn16_4", "resnet9"],
                        help="Model architecture (default: wrn16_4 per paper)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--canary-path", type=str, default=None, help="Path to optimized canaries")
    parser.add_argument("--output", type=str, default="results/dpsgd_jax_audit.json", help="Output file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DP-SGD Audit (JAX Implementation)")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Target epsilon: {args.epsilon}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    train_data, test_data = load_cifar10_numpy(args.data_dir)
    train_images, train_labels = train_data
    
    # Setup canaries
    num_canaries = args.num_canaries
    
    # Use subset of training data as canaries
    rng = np.random.RandomState(42)
    canary_indices = rng.permutation(len(train_images))[:num_canaries]
    
    random_canary_images = train_images[canary_indices]
    random_canary_labels = train_labels[canary_indices]
    
    # Remove canaries from training set
    train_mask = np.ones(len(train_images), dtype=bool)
    train_mask[canary_indices] = False
    clean_train_data = (train_images[train_mask], train_labels[train_mask])
    
    # Prepare canary types
    canary_configs = {
        "random": (random_canary_images, random_canary_labels),
        "mislabeled": (random_canary_images, create_mislabeled_canaries_np(random_canary_labels)),
    }
    
    # Load optimized canaries if provided
    if args.canary_path:
        import torch
        optimized = torch.load(args.canary_path, weights_only=False)
        opt_images = optimized["images"].numpy()
        opt_labels = optimized["labels"].numpy()
        
        # Convert from PyTorch NCHW to JAX NHWC format
        if opt_images.shape[1] == 3:  # NCHW format
            opt_images = np.transpose(opt_images, (0, 2, 3, 1))
        
        canary_configs["metagradient"] = (opt_images, opt_labels)
        print(f"Loaded optimized canaries from {args.canary_path}")
        print(f"  Shape: {opt_images.shape} (NHWC)")
    
    # Run experiments
    all_results = []
    
    for canary_type, (canary_imgs, canary_lbls) in canary_configs.items():
        print(f"\n{'='*60}")
        print(f"Canary type: {canary_type}")
        print(f"{'='*60}")
        
        for seed in range(args.seeds):
            print(f"\nSeed {seed + 1}/{args.seeds}")
            
            result = run_dp_audit_jax(
                canary_type=canary_type,
                canary_images=canary_imgs,
                canary_labels=canary_lbls,
                train_data=clean_train_data,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                model_name=args.model,
                seed=seed,
                verbose=True
            )
            result["seed"] = seed
            all_results.append(result)
            
            print(f"  Achieved epsilon: {result['achieved_epsilon']:.2f}")
            print(f"  Steinke epsilon:    {result['steinke_epsilon']:.2f} ({result['steinke_correct']}/{result['steinke_guesses']} correct)")
            print(f"  Mahloujifar epsilon: {result['mahloujifar_epsilon']:.2f} ({result['mahloujifar_correct']}/{result['mahloujifar_guesses']} correct)")
            print(f"  Loss gap (IN-OUT): {result['loss_gap']:.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            "config": vars(args),
            "results": all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


def create_mislabeled_canaries_np(labels: np.ndarray, num_classes: int = 10, seed: int = 42) -> np.ndarray:
    """Create mislabeled canaries (numpy version)."""
    rng = np.random.RandomState(seed)
    new_labels = np.zeros_like(labels)
    
    for i, label in enumerate(labels):
        wrong_labels = [l for l in range(num_classes) if l != label]
        new_labels[i] = rng.choice(wrong_labels)
    
    return new_labels


if __name__ == "__main__":
    main()
