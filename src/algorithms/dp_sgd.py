"""DP-SGD training wrapper using Opacus.

Implements differentially private training for auditing experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Tuple, Optional
from tqdm import tqdm


def make_model_dp_compatible(model: nn.Module) -> nn.Module:
    """Make a model compatible with DP training.
    
    Replaces BatchNorm with GroupNorm and fixes other incompatibilities.
    
    Args:
        model: Original model
    
    Returns:
        DP-compatible model
    """
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def train_dpsgd(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 4.0,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 3.0,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[nn.Module, float]:
    """Train model with DP-SGD.
    
    Args:
        model: Model to train (will be made DP-compatible)
        train_loader: Training dataloader
        num_epochs: Number of training epochs
        lr: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier for DP
        target_epsilon: Target privacy budget
        target_delta: Privacy parameter delta
        device: Device to train on
        verbose: Print progress
    
    Returns:
        Tuple of (trained model, achieved epsilon)
    """
    # Make model DP-compatible
    model = make_model_dp_compatible(model)
    model = model.to(device)
    model.train()
    
    # Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Attach privacy engine
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=num_epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    
    # Training loop
    pbar = tqdm(range(num_epochs), disable=not verbose, desc="DP-SGD Training")
    
    for epoch in pbar:
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        epsilon = privacy_engine.get_epsilon(target_delta)
        
        if verbose:
            pbar.set_postfix({"loss": avg_loss, "ε": epsilon})
    
    final_epsilon = privacy_engine.get_epsilon(target_delta)
    
    return model, final_epsilon


def train_dpsgd_with_canaries(
    model: nn.Module,
    train_loader: DataLoader,
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    in_mask: torch.Tensor,
    num_epochs: int = 100,
    lr: float = 4.0,
    max_grad_norm: float = 1.0,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[nn.Module, float]:
    """Train model with DP-SGD including canaries.
    
    Note: For proper DP accounting, canaries should be part of the 
    DataLoader, not added separately. This is a simplified version.
    
    Args:
        model: Model to train
        train_loader: Non-canary training dataloader
        canary_images: Canary images
        canary_labels: Canary labels
        in_mask: Boolean mask for C_IN canaries
        num_epochs: Number of epochs
        lr: Learning rate
        max_grad_norm: Max gradient norm
        target_epsilon: Target epsilon
        target_delta: Delta parameter
        device: Device
        verbose: Verbose output
    
    Returns:
        Tuple of (trained model, achieved epsilon)
    """
    # Make model DP-compatible
    model = make_model_dp_compatible(model)
    model = model.to(device)
    model.train()
    
    # Get IN canaries
    in_canaries = canary_images[in_mask].to(device)
    in_labels = canary_labels[in_mask].to(device)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Attach privacy engine
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=num_epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    
    # Training loop
    pbar = tqdm(range(num_epochs), disable=not verbose, desc="DP-SGD Training")
    
    for epoch in pbar:
        total_loss = 0
        num_batches = 0
        canary_idx = 0
        
        for batch_x, batch_y in train_loader:
            print(num_batches)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Add canaries to batch (simplified - proper implementation 
            # should integrate canaries into the dataloader)
            if canary_idx < len(in_canaries):
                num_to_add = min(4, len(in_canaries) - canary_idx)
                batch_x = torch.cat([batch_x, in_canaries[canary_idx:canary_idx+num_to_add]])
                batch_y = torch.cat([batch_y, in_labels[canary_idx:canary_idx+num_to_add]])
                canary_idx += num_to_add
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        epsilon = privacy_engine.get_epsilon(target_delta)
        
        if verbose:
            pbar.set_postfix({"loss": avg_loss, "ε": epsilon})
    
    final_epsilon = privacy_engine.get_epsilon(target_delta)
    
    return model, final_epsilon
