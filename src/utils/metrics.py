"""Utility functions for metrics, logging, and reproducibility."""

import torch
import numpy as np
import random
import os
from typing import Dict, List, Optional
import json
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> float:
    """Compute model accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
    
    Returns:
        Accuracy as a float in [0, 1]
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return correct / total


class ExperimentLogger:
    """Simple experiment logger."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """Initialize logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment (default: timestamp)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")
        self.data: Dict[str, List] = {"metrics": [], "config": {}}
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        self.data["config"] = config
        self._save()
    
    def log_metric(self, step: int, **kwargs):
        """Log metrics at a step."""
        entry = {"step": step, **kwargs}
        self.data["metrics"].append(entry)
        self._save()
    
    def _save(self):
        """Save log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_metrics(self) -> List[Dict]:
        """Get all logged metrics."""
        return self.data["metrics"]


def save_canaries(
    canary_images: torch.Tensor,
    canary_labels: torch.Tensor,
    save_path: str
):
    """Save optimized canaries to disk.
    
    Args:
        canary_images: Canary images [m, 3, 32, 32]
        canary_labels: Canary labels [m]
        save_path: Path to save
    """
    torch.save({
        "images": canary_images,
        "labels": canary_labels
    }, save_path)


def load_canaries(load_path: str) -> tuple:
    """Load canaries from disk.
    
    Args:
        load_path: Path to load from
    
    Returns:
        Tuple of (images, labels)
    """
    data = torch.load(load_path)
    return data["images"], data["labels"]
