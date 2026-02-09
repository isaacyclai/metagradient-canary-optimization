"""Run DP-SGD privacy audit using JAX

This script uses JAX-based DP-SGD which is more memory efficient than Opacus
and matches the jax-privacy implementation used in the original paper.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, '.')

import numpy as np

import jax

from src.models.resnet9_jax import create_resnet9
from src.algorithms.dp_sgd_jax import train_dpsgd_jax, evaluate_canary_losses


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


def run_dp_audit_jax(
    canary_type: str,
    canary_images: np.ndarray,
    canary_labels: np.ndarray,
    train_data: tuple,
    target_epsilon: float,
    target_delta: float,
    num_epochs: int,
    batch_size: int,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """Run a single DP-SGD audit experiment with JAX.
    
    Args:
        canary_type: Type of canary ('random', 'mislabeled', 'metagradient')
        canary_images: Canary images [m, H, W, C]
        canary_labels: Canary labels [m]
        train_data: Tuple of (train_images, train_labels)
        target_epsilon: Target privacy budget
        target_delta: Target delta
        num_epochs: Number of training epochs
        batch_size: Batch size
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with audit results
    """
    # Random IN/OUT split
    rng = np.random.RandomState(seed)
    m = len(canary_images)
    perm = rng.permutation(m)
    in_mask = np.zeros(m, dtype=bool)
    in_mask[perm[:m//2]] = True
    
    # Create model
    model = create_resnet9(num_classes=10)
    
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
    
    return {
        "canary_type": canary_type,
        "achieved_epsilon": achieved_epsilon,
        "target_epsilon": target_epsilon,
        "in_mask": in_mask.tolist(),
        **canary_stats,
        **train_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Audit DP-SGD with JAX (paper-faithful)")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num-canaries", type=int, default=1000, help="Number of canaries")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
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
                seed=seed,
                verbose=True
            )
            result["seed"] = seed
            all_results.append(result)
            
            print(f"  Achieved epsilon: {result['achieved_epsilon']:.2f}")
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
