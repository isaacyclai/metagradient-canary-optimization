"""Audit DP-SGD models (Table 2 replication).

Trains WRN 16-4 with DP-SGD and evaluates empirical epsilon
using different canary types.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import sys
sys.path.insert(0, '.')

from src.models import wrn16_4, resnet9
from src.data import get_dataloaders, CombinedTrainDataset, create_mislabeled_canaries
from src.algorithms import train_dpsgd_with_canaries
from src.auditing import audit_steinke, audit_mahloujifar
from src.utils import set_seed, get_device, load_canaries


def run_dp_experiment(
    canary_type: str,
    canary_dataset,
    train_loader,
    target_epsilon: float,
    target_delta: float,
    num_epochs: int,
    device: str,
    model,
    optimized_canaries_path: str = None,
    seed: int = 42
):
    """Run a single DP-SGD auditing experiment.
    
    Args:
        canary_type: One of "random", "mislabeled", "metagradient"
        canary_dataset: Dataset with canary management  
        train_loader: Training data loader
        target_epsilon: Target DP epsilon
        target_delta: Target DP delta
        num_epochs: Number of training epochs
        device: Device
        model: Model factory function
        optimized_canaries_path: Path to optimized canaries
        seed: Random seed
    
    Returns:
        Dict with auditing results
    """
    set_seed(seed)
    
    # Get canaries based on type
    if canary_type == "metagradient" and optimized_canaries_path:
        canary_images, canary_labels = load_canaries(optimized_canaries_path)
    else:
        canary_images, canary_labels = canary_dataset.get_canaries()
    
    if canary_type == "mislabeled":
        canary_labels = create_mislabeled_canaries(canary_images, canary_labels)
    
    # Random split for auditing
    m = len(canary_images)
    perm = torch.randperm(m)
    in_mask = torch.zeros(m, dtype=torch.bool)
    in_mask[perm[:m//2]] = True
    
    # Initialize model
    model = model(num_classes=10)
    
    # Train with DP-SGD
    print(f"Training with DP-SGD (target ε={target_epsilon}, δ={target_delta})...")
    trained_model, achieved_epsilon = train_dpsgd_with_canaries(
        model=model,
        train_loader=train_loader,
        canary_images=canary_images,
        canary_labels=canary_labels,
        in_mask=in_mask,
        num_epochs=num_epochs,
        lr=4.0,
        max_grad_norm=1.0,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        device=device,
        verbose=True
    )
    
    print(f"Achieved ε = {achieved_epsilon:.3f}")
    
    # Audit with both procedures
    eps_steinke, correct_s, guesses_s = audit_steinke(
        trained_model, canary_images, canary_labels, in_mask,
        tau=0.05, delta=target_delta, device=device
    )
    
    eps_mahloujifar, correct_m, guesses_m = audit_mahloujifar(
        trained_model, canary_images, canary_labels, in_mask,
        tau=0.05, delta=target_delta, device=device
    )
    
    return {
        "canary_type": canary_type,
        "seed": seed,
        "achieved_epsilon": achieved_epsilon,
        "steinke": {"epsilon": eps_steinke, "correct": correct_s, "guesses": guesses_s},
        "mahloujifar": {"epsilon": eps_mahloujifar, "correct": correct_m, "guesses": guesses_m}
    }


def main():
    parser = argparse.ArgumentParser(description="Audit DP-SGD (Table 2)")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target DP epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target DP delta")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--num-canaries", type=int, default=1000, help="Number of canaries")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--canary-path", type=str, default=None, help="Path to optimized canaries")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output", type=str, default="table2_results.json", help="Output file")
    parser.add_argument("--model", type=str, default="wrn16_4", choices=["wrn16_4", "resnet9"],
                        help="Model architecture to use")
    args = parser.parse_args()
    
    # Select model factory
    model = wrn16_4 if args.model == "wrn16_4" else resnet9
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    canary_dataset, train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        num_canaries=args.num_canaries,
        batch_size=args.batch_size
    )
    
    all_results = {"random": [], "mislabeled": [], "metagradient": []}
    
    canary_types = ["random", "mislabeled"]
    if args.canary_path:
        canary_types.append("metagradient")
    
    for seed in range(args.seeds):
        print(f"\n{'='*50}")
        print(f"Seed {seed + 1}/{args.seeds}")
        print('='*50)
        
        for canary_type in canary_types:
            print(f"\nRunning with {canary_type} canaries...")
            result = run_dp_experiment(
                canary_type=canary_type,
                canary_dataset=canary_dataset,
                train_loader=train_loader,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                num_epochs=args.epochs,
                device=device,
                model=model,
                optimized_canaries_path=args.canary_path,
                seed=seed
            )
            all_results[canary_type].append(result)
            
            print(f"  Steinke ε: {result['steinke']['epsilon']:.3f}")
            print(f"  Mahloujifar ε: {result['mahloujifar']['epsilon']:.3f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.output}")
    
    # Print summary table (Table 2 format)
    print("\n" + "="*60)
    print("Table 2: Empirical Epsilon Results")
    print("="*60)
    print(f"{'Canary Type':<15} {'Audit Procedure':<15} {'Avg':<8} {'Med':<8}")
    print("-"*60)
    
    for canary_type in canary_types:
        results = all_results[canary_type]
        
        # Steinke
        eps_s = [r["steinke"]["epsilon"] for r in results]
        print(f"{canary_type:<15} {'Steinke':<15} {np.mean(eps_s):.3f}    {np.median(eps_s):.3f}")
        
        # Mahloujifar
        eps_m = [r["mahloujifar"]["epsilon"] for r in results]
        print(f"{'':<15} {'Mahloujifar':<15} {np.mean(eps_m):.3f}    {np.median(eps_m):.3f}")
        
        print()


if __name__ == "__main__":
    main()
