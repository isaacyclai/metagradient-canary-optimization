"""ResNet-9 model for efficient canary optimization.

This is a lightweight model used during metagradient canary optimization.
Despite being smaller than the auditing target (WRN 16-4), optimized canaries
transfer effectively to larger models.

Modified for metasmoothness:
- BatchNorm placed before activation (not after)
- Output scaling for stable metagradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BN before activation for metasmoothness."""
    
    def __init__(self, in_channels: int, out_channels: int, pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BN before activation for metasmoothness
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class ResBlock(nn.Module):
    """Residual block with two conv layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + residual)
        return x


class ResNet9(nn.Module):
    """ResNet-9 for CIFAR-10 with metasmoothness modifications.
    
    Architecture:
        conv(64) -> conv(128, pool) -> res(128) -> conv(256, pool) -> 
        conv(512, pool) -> res(512) -> pool -> fc(10)
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        output_scale: Scaling factor for output logits (for metasmoothness)
    """
    
    def __init__(self, num_classes: int = 10, output_scale: float = 0.1):
        super().__init__()
        self.output_scale = output_scale
        
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = ResBlock(128)
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = ResBlock(512)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x * self.output_scale
    
    def forward_unscaled(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without output scaling (for evaluation)."""
        return self.forward(x) / self.output_scale


def resnet9(num_classes: int = 10, metasmooth: bool = True) -> ResNet9:
    """Create a ResNet-9 model.
    
    Args:
        num_classes: Number of output classes
        metasmooth: If True, apply output scaling for metasmoothness
    
    Returns:
        ResNet9 model
    """
    output_scale = 0.1 if metasmooth else 1.0
    return ResNet9(num_classes=num_classes, output_scale=output_scale)
