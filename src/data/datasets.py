"""CIFAR-10 dataset utilities and canary management.

Handles dataset loading, canary set creation, and train/canary splits
as described in the paper.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List, Optional
import numpy as np
import random


def get_cifar10_transforms(train: bool = True, augment: bool = True):
    """Get transforms for CIFAR-10.
    
    Args:
        train: Whether this is for training data
        augment: Whether to apply data augmentation
    
    Returns:
        Transform composition
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def load_cifar10(
    data_dir: str = "./data",
    train: bool = True,
    augment: bool = True
) -> torchvision.datasets.CIFAR10:
    """Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load data
        train: Load training or test set
        augment: Apply data augmentation (only for training)
    
    Returns:
        CIFAR10 dataset
    """
    transform = get_cifar10_transforms(train=train, augment=augment)
    return torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )


class CanaryDataset(Dataset):
    """Dataset wrapper for canary management.
    
    Separates the training set into:
    - r = 49000 non-canary training samples
    - m = 1000 canary samples
    
    The canary samples can be optimized via metagradients.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        num_canaries: int = 1000,
        seed: int = 42
    ):
        """Initialize canary dataset.
        
        Args:
            base_dataset: Base CIFAR-10 training dataset
            num_canaries: Number of canary samples (m)
            seed: Random seed for reproducible splits
        """
        self.base_dataset = base_dataset
        self.num_canaries = num_canaries
        self.num_total = len(base_dataset)
        self.num_train = self.num_total - num_canaries
        
        # Create reproducible random split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(self.num_total)
        
        self.train_indices = indices[:self.num_train].tolist()
        self.canary_indices = indices[self.num_train:].tolist()
        
        # Initialize canary tensors (will be optimized)
        self._init_canaries()
    
    def _init_canaries(self):
        """Initialize canary images and labels from the base dataset."""
        canary_images = []
        canary_labels = []
        
        # Get raw tensors without augmentation for canaries
        no_aug_transform = get_cifar10_transforms(train=False, augment=False)
        
        for idx in self.canary_indices:
            img, label = self.base_dataset.data[idx], self.base_dataset.targets[idx]
            # Convert to tensor
            img_tensor = no_aug_transform(
                torchvision.transforms.functional.to_pil_image(img)
            )
            canary_images.append(img_tensor)
            canary_labels.append(label)
        
        self.canary_images = torch.stack(canary_images)  # [m, 3, 32, 32]
        self.canary_labels = torch.tensor(canary_labels)  # [m]
    
    def get_train_subset(self) -> Subset:
        """Get the non-canary training subset."""
        return Subset(self.base_dataset, self.train_indices)
    
    def get_canaries(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current canary images and labels.
        
        Returns:
            Tuple of (images [m, 3, 32, 32], labels [m])
        """
        return self.canary_images.clone(), self.canary_labels.clone()
    
    def set_canaries(self, canary_images: torch.Tensor):
        """Update canary images (after optimization step).
        
        Args:
            canary_images: New canary images [m, 3, 32, 32]
        """
        assert canary_images.shape == self.canary_images.shape
        self.canary_images = canary_images.detach().clone()
    
    def random_canary_split(
        self, 
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly split canaries into C_IN and C_OUT.
        
        Args:
            seed: Optional random seed
        
        Returns:
            Tuple of (images, labels, in_mask) where in_mask[i] = True if canary i is in C_IN
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        m = self.num_canaries
        perm = torch.randperm(m)
        half = m // 2
        
        in_mask = torch.zeros(m, dtype=torch.bool)
        in_mask[perm[:half]] = True
        
        return self.canary_images.clone(), self.canary_labels.clone(), in_mask


def create_mislabeled_canaries(
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    num_classes: int = 10,
    seed: int = 42
) -> torch.Tensor:
    """Create mislabeled canaries as a baseline.
    
    Each canary gets a random incorrect label.
    
    Args:
        canary_images: Canary images tensor
        canary_labels: Original correct labels
        num_classes: Number of classes
        seed: Random seed
    
    Returns:
        New labels (all incorrect)
    """
    rng = np.random.RandomState(seed)
    new_labels = []
    
    for label in canary_labels.numpy():
        # Choose a random incorrect label
        wrong_labels = [l for l in range(num_classes) if l != label]
        new_label = rng.choice(wrong_labels)
        new_labels.append(new_label)
    
    return torch.tensor(new_labels)


class CombinedTrainDataset(Dataset):
    """Combines non-canary training data with C_IN canaries."""
    
    def __init__(
        self,
        train_subset: Subset,
        canary_images: torch.Tensor,
        canary_labels: torch.Tensor,
        in_mask: torch.Tensor,
        transform=None
    ):
        """Initialize combined dataset.
        
        Args:
            train_subset: Non-canary training data
            canary_images: All canary images
            canary_labels: All canary labels
            in_mask: Boolean mask for C_IN
            transform: Optional transform for canaries
        """
        self.train_subset = train_subset
        self.in_canary_images = canary_images[in_mask]
        self.in_canary_labels = canary_labels[in_mask]
        self.transform = transform
        
        self.num_train = len(train_subset)
        self.num_in_canaries = in_mask.sum().item()
    
    def __len__(self) -> int:
        return self.num_train + self.num_in_canaries
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < self.num_train:
            return self.train_subset[idx]
        else:
            canary_idx = idx - self.num_train
            img = self.in_canary_images[canary_idx]
            label = self.in_canary_labels[canary_idx].item()
            if self.transform:
                img = self.transform(img)
            return img, label


def get_dataloaders(
    data_dir: str = "./data",
    num_canaries: int = 1000,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[CanaryDataset, DataLoader, DataLoader]:
    """Get train and test dataloaders with canary setup.
    
    Args:
        data_dir: Data directory
        num_canaries: Number of canary samples
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed
    
    Returns:
        Tuple of (canary_dataset, train_loader, test_loader)
    """
    # Load datasets
    train_dataset = load_cifar10(data_dir, train=True, augment=True)
    test_dataset = load_cifar10(data_dir, train=False, augment=False)
    
    # Create canary dataset
    canary_dataset = CanaryDataset(train_dataset, num_canaries, seed)
    
    # Create loaders
    train_loader = DataLoader(
        canary_dataset.get_train_subset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return canary_dataset, train_loader, test_loader
