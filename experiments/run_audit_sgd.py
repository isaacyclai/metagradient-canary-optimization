"""Audit non-DP SGD models (Figure 2 replication).

Trains WRN 16-4 on CIFAR-10 with different canary types and
evaluates empirical epsilon at regular intervals.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import sys
sys.path.insert(0, '.')

from src.models import wrn16_4
from src.data import get_dataloaders, CombinedTrainDataset, create_mislabeled_canaries
from src.auditing import audit_steinke, audit_mahloujifar
from src.utils import set_seed, get_device, load_canaries, ExperimentLogger
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


def evaluate_at_step(
    model, 
    canary_images, 
    canary_labels, 
    in_mask, 
    device,
    tau=0.05,
    delta=1e-5
):
    """Evaluate both auditing procedures."""
    eps_steinke, correct_s, guesses_s = audit_steinke(
        model, canary_images, canary_labels, in_mask,
        tau=tau, delta=delta, device=device
    )
    
    eps_mahloujifar, correct_m, guesses_m = audit_mahloujifar(
        model, canary_images, canary_labels, in_mask,
        tau=tau, delta=delta, device=device
    )
    
    return {
        "steinke": {"epsilon": eps_steinke, "correct": correct_s, "guesses": guesses_s},
        "mahloujifar": {"epsilon": eps_mahloujifar, "correct": correct_m, "guesses": guesses_m}
    }


def run_experiment(
    canary_type: str,
    canary_dataset,
    train_loader,
    test_loader,
    total_steps: int,
    eval_interval: int,
    device: str,
    optimized_canaries_path: str = None,
    seed: int = 42
):
    """Run a single auditing experiment.
    
    Args:
        canary_type: One of "random", "mislabeled", "metagradient"
        canary_dataset: Dataset with canary management
        train_loader: Training data loader
        test_loader: Test data loader
        total_steps: Total training steps
        eval_interval: Evaluate every N steps
        device: Device
        optimized_canaries_path: Path to optimized canaries (for metagradient)
        seed: Random seed
    
    Returns:
        List of evaluation results at each checkpoint
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
    
    # Create combined dataset with C_IN
    combined_dataset = CombinedTrainDataset(
        canary_dataset.get_train_subset(),
        canary_images, canary_labels, in_mask
    )
    combined_loader = DataLoader(
        combined_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    
    # Initialize model
    model = wrn16_4(num_classes=10, use_group_norm=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    step = 0
    
    pbar = tqdm(total=total_steps, desc=f"{canary_type} canaries")
    
    while step < total_steps:
        for x, y in combined_loader:
            if step >= total_steps:
                break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            
            # Evaluate at intervals
            if step % eval_interval == 0 or step == total_steps:
                eval_result = evaluate_at_step(
                    model, canary_images, canary_labels, in_mask, device
                )
                eval_result["step"] = step
                results.append(eval_result)
                
                pbar.set_postfix({
                    "ε_S": eval_result["steinke"]["epsilon"],
                    "ε_M": eval_result["mahloujifar"]["epsilon"]
                })
            
            pbar.update(1)
    
    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Audit non-DP SGD (Figure 2)")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--num-canaries", type=int, default=1000, help="Number of canaries")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--canary-path", type=str, default=None, help="Path to optimized canaries")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output", type=str, default="figure2_results.json", help="Output file")
    args = parser.parse_args()
    
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
        print(f"\n=== Seed {seed + 1}/{args.seeds} ===")
        
        for canary_type in canary_types:
            print(f"\nRunning with {canary_type} canaries...")
            results = run_experiment(
                canary_type=canary_type,
                canary_dataset=canary_dataset,
                train_loader=train_loader,
                test_loader=test_loader,
                total_steps=args.steps,
                eval_interval=args.eval_interval,
                device=device,
                optimized_canaries_path=args.canary_path,
                seed=seed
            )
            all_results[canary_type].append(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.output}")
    
    # Print summary
    print("\n=== Summary (mean empirical epsilon at final step) ===")
    for canary_type in canary_types:
        final_epsilons_s = [r[-1]["steinke"]["epsilon"] for r in all_results[canary_type]]
        final_epsilons_m = [r[-1]["mahloujifar"]["epsilon"] for r in all_results[canary_type]]
        print(f"{canary_type}:")
        print(f"  Steinke: {np.mean(final_epsilons_s):.3f} ± {np.std(final_epsilons_s):.3f}")
        print(f"  Mahloujifar: {np.mean(final_epsilons_m):.3f} ± {np.std(final_epsilons_m):.3f}")


if __name__ == "__main__":
    main()
