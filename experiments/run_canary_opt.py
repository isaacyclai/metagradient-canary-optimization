"""Run canary optimization using metagradient descent (Algorithm 5).

This script optimizes canary samples for more effective privacy auditing.
"""

import argparse
import torch
import sys
sys.path.insert(0, '.')

from src.data import get_dataloaders
from src.algorithms import optimize_canaries
from src.utils import set_seed, get_device, save_canaries, ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Optimize canaries with metagradient descent")
    parser.add_argument("--num-canaries", type=int, default=1000, help="Number of canaries")
    parser.add_argument("--meta-steps", type=int, default=50, help="Number of meta-optimization steps")
    parser.add_argument("--epochs-per-step", type=int, default=12, help="Training epochs per meta-step")
    parser.add_argument("--canary-lr", type=float, default=0.01, help="Learning rate for canary updates")
    parser.add_argument("--model-lr", type=float, default=0.1, help="Learning rate for model training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output", type=str, default="optimized_canaries.pt", help="Output file")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10...")
    canary_dataset, train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        num_canaries=args.num_canaries,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    print(f"Training samples: {len(canary_dataset.train_indices)}")
    print(f"Canary samples: {args.num_canaries}")
    
    # Setup logging
    logger = ExperimentLogger(args.log_dir, "canary_optimization")
    logger.log_config(vars(args))
    
    # Optimize canaries (Algorithm 5)
    print(f"\nOptimizing canaries with {args.meta_steps} meta-steps...")
    optimized_canaries = optimize_canaries(
        canary_dataset=canary_dataset,
        train_loader=train_loader,
        num_meta_steps=args.meta_steps,
        num_epochs_per_step=args.epochs_per_step,
        canary_lr=args.canary_lr,
        model_lr=args.model_lr,
        batch_size=args.batch_size,
        device=device,
        verbose=True
    )
    
    # Save optimized canaries
    _, canary_labels = canary_dataset.get_canaries()
    save_canaries(optimized_canaries, canary_labels, args.output)
    print(f"\nSaved optimized canaries to {args.output}")


if __name__ == "__main__":
    main()
