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
from typing import Tuple
import copy
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
    device: str = "cuda"
) -> nn.Module:
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
    
    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()
    
    # Create combined dataset: D ∪ C_IN
    combined_dataset = CombinedTrainDataset(
        train_dataset,
        canary_images,
        canary_labels,
        in_mask
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
    
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
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
    
    This is the loss gap between IN and OUT canaries.
    We want to MINIMIZE this (low loss on IN, high loss on OUT).
    
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
    
    outputs = model(canary_images)
    losses = criterion(outputs, canary_labels)
    
    # ϕ(w) = L(w, C_IN) - L(w, C_OUT)
    # Weight: +1 for IN, -1 for OUT
    weights = in_mask.float() - (~in_mask).float()
    surrogate = (weights * losses).mean()
    
    return surrogate


def optimize_canaries(
    canary_dataset: CanaryDataset,
    train_loader: DataLoader,
    num_meta_steps: int = 50,
    num_epochs_per_step: int = 12,
    canary_lr: float = 0.01,
    model_lr: float = 0.1,
    batch_size: int = 128,
    device: str = "cuda",
    verbose: bool = True
) -> torch.Tensor:
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
    
    Returns:
        Optimized canary images [m, 3, 32, 32]
    """
    # Step 1: Initialize canaries
    canary_images, canary_labels = canary_dataset.get_canaries()
    canary_images = canary_images.to(device).clone().requires_grad_(True)
    canary_labels = canary_labels.to(device)
    
    # Get the training dataset (without canaries)
    train_dataset = train_loader.dataset
    
    pbar = tqdm(range(num_meta_steps), disable=not verbose, desc="Meta-optimization")
    
    for t in pbar:
        # Step 2a: Random split into C_IN and C_OUT
        m = len(canary_images)
        perm = torch.randperm(m, device=device)
        in_mask = torch.zeros(m, dtype=torch.bool, device=device)
        in_mask[perm[:m//2]] = True
        
        # Step 2b: Train model on D ∪ C_IN
        # Use fresh model each meta-step
        model = resnet9(num_classes=10).to(device)
        
        # Detach canaries for training (we'll compute gradient separately)
        canary_imgs_detached = canary_images.detach()
        
        model = train_model_with_canaries(
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
        # Forward pass with requires_grad canaries to get gradient
        model.eval()
        
        if canary_images.grad is not None:
            canary_images.grad.zero_()
        
        outputs = model(canary_images)
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses = criterion(outputs, canary_labels)
        
        # Surrogate: L(C_IN) - L(C_OUT)
        in_mask_device = in_mask.to(device)
        weights = in_mask_device.float() - (~in_mask_device).float()
        surrogate = (weights * losses).mean()
        
        # Step 2d: Compute gradient and update canaries
        surrogate.backward()
        
        # Signed gradient descent (as specified in paper)
        with torch.no_grad():
            if canary_images.grad is not None:
                grad_sign = torch.sign(canary_images.grad)
                canary_images.sub_(canary_lr * grad_sign)
                # Clamp to valid image range [0, 1]
                canary_images.clamp_(0, 1)
        
        if verbose:
            pbar.set_postfix({"surrogate": f"{surrogate.item():.4f}"})
    
    return canary_images.detach().cpu()
