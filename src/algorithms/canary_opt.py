"""Algorithm 5: Metagradient Canary Optimization.

Optimizes canary samples for privacy auditing using metagradient descent.
The surrogate objective maximizes the loss gap between IN and OUT canaries.

Reference:
    Boglioni et al. "Optimizing Canaries for Privacy Auditing with 
    Metagradient Descent" (arXiv:2507.15836)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, List
import os
import json
from datetime import datetime
from tqdm import tqdm

from ..models.resnet9 import resnet9
from ..data.datasets import CanaryDataset, CombinedTrainDataset


def train_model_with_canaries(
    model: nn.Module,
    train_dataset,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    num_epochs: int = 12,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
    device: str = "cuda",
    log_interval: int = 0
) -> Tuple[nn.Module, Dict]:
    """Train model on D ∪ C_IN.
    
    Args:
        model: Model to train
        train_dataset: Training dataset (without canaries)
        canary_images: All canary images [m, 3, 32, 32]
        canary_labels: All canary labels [m]
        in_mask: Boolean mask for C_IN
        num_epochs: Number of training epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay
        batch_size: Batch size for training
        device: Device to train on
        log_interval: Log training stats every N batches (0 = disabled)
    
    Returns:
        Tuple of (trained model, training stats dict)
    """
    model = model.to(device)
    model.train()
    
    # Ensure canary tensors are on CPU for DataLoader
    canary_images_cpu = canary_images.cpu() if canary_images.is_cuda else canary_images
    canary_labels_cpu = canary_labels.cpu() if canary_labels.is_cuda else canary_labels
    in_mask_cpu = in_mask.cpu() if in_mask.is_cuda else in_mask
    
    # Create combined dataset: D ∪ C_IN
    combined_dataset = CombinedTrainDataset(
        train_dataset,
        canary_images_cpu,
        canary_labels_cpu,
        in_mask_cpu
    )
    
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.CrossEntropyLoss()
    
    stats = {"epoch_losses": [], "epoch_accs": []}
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        scheduler.step()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        stats["epoch_losses"].append(epoch_loss)
        stats["epoch_accs"].append(epoch_acc)
    
    stats["final_loss"] = stats["epoch_losses"][-1] if stats["epoch_losses"] else 0
    stats["final_acc"] = stats["epoch_accs"][-1] if stats["epoch_accs"] else 0
    
    return model, stats


def compute_canary_stats(
    model: nn.Module,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    device: str = "cuda"
) -> Dict:
    """Compute detailed statistics on canary losses.
    
    Returns dict with IN/OUT loss means, stds, and accuracy.
    """
    model.eval()
    
    canary_images = canary_images.to(device)
    canary_labels = canary_labels.to(device)
    in_mask = in_mask.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        outputs = model(canary_images)
        losses = criterion(outputs, canary_labels)
        _, predicted = outputs.max(1)
        
        in_losses = losses[in_mask]
        out_losses = losses[~in_mask]
        
        in_correct = predicted[in_mask].eq(canary_labels[in_mask]).sum().item()
        out_correct = predicted[~in_mask].eq(canary_labels[~in_mask]).sum().item()
    
    return {
        "in_loss_mean": in_losses.mean().item(),
        "in_loss_std": in_losses.std().item() if len(in_losses) > 1 else 0,
        "out_loss_mean": out_losses.mean().item(),
        "out_loss_std": out_losses.std().item() if len(out_losses) > 1 else 0,
        "loss_gap": in_losses.mean().item() - out_losses.mean().item(),
        "in_acc": 100.0 * in_correct / len(in_losses),
        "out_acc": 100.0 * out_correct / len(out_losses),
    }


def optimize_canaries(
    canary_dataset: CanaryDataset,
    train_loader: DataLoader,
    num_meta_steps: int = 50,
    num_epochs_per_step: int = 12,
    canary_lr: float = 0.01,
    model_lr: float = 0.1,
    batch_size: int = 128,
    device: str = "cuda",
    verbose: bool = True,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 10,
    log_file: Optional[str] = None
) -> Tuple[torch.Tensor, List[Dict]]:
    """Algorithm 5: Metagradient Canary Optimization.
    
    Optimizes canary samples to maximize the loss gap between C_IN and C_OUT,
    making them more effective for privacy auditing.
    
    The algorithm:
    1. Initialize canaries C_0
    2. For t = 0 to N-1:
       a. Split C_t randomly into C_IN and C_OUT (50/50)
       b. Train model w_t on D ∪ C_IN
       c. Compute surrogate: ϕ(w_t) = L(w_t, C_IN) - L(w_t, C_OUT)
       d. Update canaries: C_{t+1} = C_t - η * sign(∇_C ϕ)
    
    Args:
        canary_dataset: Dataset with canary management
        train_loader: DataLoader for non-canary training data
        num_meta_steps: Number of meta-optimization steps (N)
        num_epochs_per_step: Training epochs per meta-step
        canary_lr: Learning rate for canary updates (η)
        model_lr: Learning rate for model training
        batch_size: Batch size for training
        device: Device to use
        verbose: Print progress
        checkpoint_dir: Directory to save checkpoints (None = no checkpoints)
        checkpoint_interval: Save checkpoint every N meta-steps
        log_file: Path to save detailed logs (None = no file logging)
    
    Returns:
        Tuple of (optimized canary images, list of per-step statistics)
    """
    # Setup logging
    history = []
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 1: Initialize canaries
    canary_images, canary_labels = canary_dataset.get_canaries()
    canary_images = canary_images.to(device).clone().requires_grad_(True)
    canary_labels = canary_labels.to(device)
    
    # Get the training dataset (without canaries)
    train_dataset = train_loader.dataset
    
    if verbose:
        print(f"Starting canary optimization:")
        print(f"  Meta-steps: {num_meta_steps}")
        print(f"  Epochs per step: {num_epochs_per_step}")
        print(f"  Num canaries: {len(canary_images)}")
        print(f"  Device: {device}")
        if checkpoint_dir:
            print(f"  Checkpoints: {checkpoint_dir} (every {checkpoint_interval} steps)")
    
    pbar = tqdm(range(num_meta_steps), disable=not verbose, desc="Meta-optimization")
    
    for t in pbar:
        step_start = datetime.now()
        
        # Step 2a: Random split into C_IN and C_OUT
        m = len(canary_images)
        perm = torch.randperm(m, device=device)
        in_mask = torch.zeros(m, dtype=torch.bool, device=device)
        in_mask[perm[:m//2]] = True
        
        # Step 2b: Train model on D ∪ C_IN
        model = resnet9(num_classes=10).to(device)
        canary_imgs_detached = canary_images.detach()
        
        model, train_stats = train_model_with_canaries(
            model=model,
            train_dataset=train_dataset,
            canary_images=canary_imgs_detached,
            canary_labels=canary_labels,
            in_mask=in_mask.cpu(),
            num_epochs=num_epochs_per_step,
            lr=model_lr,
            batch_size=batch_size,
            device=device
        )
        
        # Step 2c: Compute surrogate with gradient
        model.eval()
        
        if canary_images.grad is not None:
            canary_images.grad.zero_()
        
        outputs = model(canary_images)
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses = criterion(outputs, canary_labels)
        
        in_mask_device = in_mask.to(device)
        weights = in_mask_device.float() - (~in_mask_device).float()
        surrogate = (weights * losses).mean()
        
        # Step 2d: Compute gradient and update canaries
        surrogate.backward()
        
        grad_norm = 0.0
        with torch.no_grad():
            if canary_images.grad is not None:
                grad_norm = canary_images.grad.abs().mean().item()
                grad_sign = torch.sign(canary_images.grad)
                canary_images.sub_(canary_lr * grad_sign)
                canary_images.clamp_(0, 1)
        
        # Compute detailed stats
        canary_stats = compute_canary_stats(
            model, canary_images.detach(), canary_labels, in_mask_device, device
        )
        
        step_time = (datetime.now() - step_start).total_seconds()
        
        step_log = {
            "step": t,
            "surrogate": surrogate.item(),
            "grad_norm": grad_norm,
            "train_acc": train_stats["final_acc"],
            "train_loss": train_stats["final_loss"],
            "step_time_sec": step_time,
            **canary_stats
        }
        history.append(step_log)
        
        # Update progress bar
        if verbose:
            pbar.set_postfix({
                "surr": f"{surrogate.item():.3f}",
                "gap": f"{canary_stats['loss_gap']:.3f}",
                "acc": f"{train_stats['final_acc']:.1f}%"
            })
        
        # Save checkpoint
        if checkpoint_dir and (t + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{t+1}.pt")
            torch.save({
                "step": t + 1,
                "canary_images": canary_images.detach().cpu(),
                "canary_labels": canary_labels.cpu(),
                "history": history,
            }, checkpoint_path)
            if verbose:
                print(f"\n  Saved checkpoint: {checkpoint_path}")
    
    # Save final log
    if log_file:
        with open(log_file, 'w') as f:
            json.dump({
                "config": {
                    "num_meta_steps": num_meta_steps,
                    "num_epochs_per_step": num_epochs_per_step,
                    "canary_lr": canary_lr,
                    "model_lr": model_lr,
                    "num_canaries": m,
                },
                "history": history
            }, f, indent=2)
        if verbose:
            print(f"Saved log to: {log_file}")
    
    # Print summary
    if verbose and history:
        print(f"\nOptimization complete:")
        print(f"  Initial surrogate: {history[0]['surrogate']:.4f}")
        print(f"  Final surrogate: {history[-1]['surrogate']:.4f}")
        print(f"  Final loss gap (IN-OUT): {history[-1]['loss_gap']:.4f}")
        print(f"  Final train accuracy: {history[-1]['train_acc']:.1f}%")
    
    return canary_images.detach().cpu(), history
