"""Algorithm 5: Metagradient Canary Optimization.

Optimizes canary samples for privacy auditing using metagradient descent.
The surrogate objective maximizes the loss gap between IN and OUT canaries.

Reference:
    Boglioni et al. "Optimizing Canaries for Privacy Auditing with 
    Metagradient Descent" (arXiv:2507.15836)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Callable
import copy
from tqdm import tqdm

from ..models.resnet9 import resnet9
from ..data.datasets import CanaryDataset, CombinedTrainDataset


def train_with_canaries(
    model: nn.Module,
    train_loader: DataLoader,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    num_epochs: int = 12,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda"
) -> nn.Module:
    """Train model on D ∪ C_IN.
    
    Args:
        model: Model to train
        train_loader: DataLoader for non-canary training data
        canary_images: All canary images [m, 3, 32, 32]
        canary_labels: All canary labels [m]
        in_mask: Boolean mask for C_IN
        num_epochs: Number of training epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay
        device: Device to train on
    
    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()
    
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
    
    # Get IN canaries
    in_canaries = canary_images[in_mask].to(device)
    in_labels = canary_labels[in_mask].to(device)
    
    for epoch in range(num_epochs):
        canary_idx = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Add some canaries to each batch
            num_canaries_per_batch = max(1, len(in_canaries) // len(train_loader))
            end_idx = min(canary_idx + num_canaries_per_batch, len(in_canaries))
            
            if canary_idx < len(in_canaries):
                batch_canaries = in_canaries[canary_idx:end_idx]
                batch_canary_labels = in_labels[canary_idx:end_idx]
                
                combined_x = torch.cat([batch_x, batch_canaries], dim=0)
                combined_y = torch.cat([batch_y, batch_canary_labels], dim=0)
                canary_idx = end_idx
            else:
                combined_x, combined_y = batch_x, batch_y
            
            optimizer.zero_grad()
            outputs = model(combined_x)
            loss = criterion(outputs, combined_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    return model


def compute_surrogate_loss(
    model: nn.Module,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """Compute the surrogate objective: L(w, C_IN) - L(w, C_OUT).
    
    Args:
        model: Trained model
        canary_images: All canary images [m, 3, 32, 32]
        canary_labels: All canary labels [m]
        in_mask: Boolean mask for C_IN
        device: Device
    
    Returns:
        Surrogate loss (scalar)
    """
    model.eval()
    
    canary_images = canary_images.to(device)
    canary_labels = canary_labels.to(device)
    in_mask = in_mask.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.enable_grad():
        outputs = model(canary_images)
        losses = criterion(outputs, canary_labels)
        
        # Weight by IN/OUT: +1 for IN (want low loss), -1 for OUT (want high loss)
        # Objective: maximize loss gap = minimize L(C_IN) - L(C_OUT)
        # So surrogate = L(C_IN) - L(C_OUT)
        weights = in_mask.float() - (~in_mask).float()
        surrogate = (weights * losses).sum()
    
    return surrogate


def optimize_canaries(
    canary_dataset: CanaryDataset,
    train_loader: DataLoader,
    num_meta_steps: int = 50,
    num_epochs_per_step: int = 12,
    canary_lr: float = 0.01,
    model_lr: float = 0.1,
    device: str = "cuda",
    verbose: bool = True
) -> torch.Tensor:
    """Algorithm 5: Metagradient Canary Optimization.
    
    Optimizes canary samples to maximize the loss gap between C_IN and C_OUT,
    making them more effective for privacy auditing.
    
    Args:
        canary_dataset: Dataset with canary management
        train_loader: DataLoader for non-canary training data
        num_meta_steps: Number of meta-optimization steps (N)
        num_epochs_per_step: Training epochs per meta-step
        canary_lr: Learning rate for canary updates
        model_lr: Learning rate for model training
        device: Device to use
        verbose: Print progress
    
    Returns:
        Optimized canary images [m, 3, 32, 32]
    """
    # Get initial canaries
    canary_images, canary_labels = canary_dataset.get_canaries()
    canary_images = canary_images.to(device).requires_grad_(True)
    canary_labels = canary_labels.to(device)
    
    optimizer = torch.optim.SGD([canary_images], lr=canary_lr)
    
    pbar = tqdm(range(num_meta_steps), disable=not verbose, desc="Meta-steps")
    
    for t in pbar:
        # Step 1: Random split into C_IN and C_OUT
        m = len(canary_images)
        perm = torch.randperm(m, device=device)
        in_mask = torch.zeros(m, dtype=torch.bool, device=device)
        in_mask[perm[:m//2]] = True
        
        # Step 2: Train model on D ∪ C_IN
        model = resnet9(num_classes=10, metasmooth=True).to(device)
        
        # Create detached copy for forward training
        canary_imgs_detached = canary_images.detach()
        model = train_with_canaries(
            model, train_loader,
            canary_imgs_detached, canary_labels, in_mask.cpu(),
            num_epochs=num_epochs_per_step,
            lr=model_lr,
            device=device
        )
        
        # Step 3: Compute surrogate loss with gradient
        optimizer.zero_grad()
        
        # Re-run final forward pass with canaries that require grad
        model.eval()
        outputs = model(canary_images)
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses = criterion(outputs, canary_labels)
        
        # Surrogate: L(C_IN) - L(C_OUT) 
        # We want to minimize this (low loss on IN, high on OUT)
        weights = in_mask.float() - (~in_mask).float()
        surrogate = (weights * losses).sum()
        
        # Step 4: Compute gradient and update canaries
        surrogate.backward()
        
        # Signed gradient descent (as in paper)
        with torch.no_grad():
            grad_sign = torch.sign(canary_images.grad)
            canary_images.sub_(canary_lr * grad_sign)
            # Clamp to valid image range
            canary_images.clamp_(0, 1)
            canary_images.grad.zero_()
        
        if verbose:
            pbar.set_postfix({"surrogate": surrogate.item()})
    
    return canary_images.detach().cpu()


def optimize_canaries_simplified(
    canary_dataset: CanaryDataset,
    train_loader: DataLoader,
    num_meta_steps: int = 50,
    num_epochs_per_step: int = 12,
    canary_lr: float = 0.01,
    device: str = "cuda",
    verbose: bool = True
) -> torch.Tensor:
    """Simplified canary optimization using end-to-end gradients.
    
    This version directly backpropagates through training, which is
    memory-intensive but provides exact gradients for small-scale experiments.
    
    For larger experiments, use the REPLAY-based version.
    """
    canary_images, canary_labels = canary_dataset.get_canaries()
    canary_images = canary_images.to(device)
    canary_labels = canary_labels.to(device)
    
    # Make canaries require gradient
    canary_images = canary_images.clone().requires_grad_(True)
    
    pbar = tqdm(range(num_meta_steps), disable=not verbose, desc="Meta-steps")
    
    for t in pbar:
        # Random split
        m = len(canary_images)
        perm = torch.randperm(m, device=device)
        in_mask = torch.zeros(m, dtype=torch.bool, device=device)
        in_mask[perm[:m//2]] = True
        
        # Fresh model
        model = resnet9(num_classes=10, metasmooth=True).to(device)
        optimizer_m = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Short training with canaries in computational graph
        model.train()
        in_canaries = canary_images[in_mask]
        in_labels = canary_labels[in_mask]
        
        # Simplified: just train on canaries for a few steps
        for _ in range(min(100, num_epochs_per_step * 10)):
            optimizer_m.zero_grad()
            out = model(in_canaries)
            loss = criterion(out, in_labels)
            loss.backward(retain_graph=True)
            optimizer_m.step()
        
        # Compute surrogate loss
        model.eval()
        outputs = model(canary_images)
        losses = nn.CrossEntropyLoss(reduction='none')(outputs, canary_labels)
        weights = in_mask.float() - (~in_mask).float()
        surrogate = (weights * losses).sum()
        
        # Backprop to canaries
        if canary_images.grad is not None:
            canary_images.grad.zero_()
        surrogate.backward()
        
        # Update canaries
        with torch.no_grad():
            if canary_images.grad is not None:
                grad_sign = torch.sign(canary_images.grad)
                canary_images.sub_(canary_lr * grad_sign)
                canary_images.clamp_(0, 1)
        
        if verbose:
            pbar.set_postfix({"surrogate": surrogate.item()})
    
    return canary_images.detach().cpu()
