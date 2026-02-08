"""ResNet-9 model for CIFAR-10.

Standard ResNet-9 architecture.
Architecture: conv(64) -> conv(128, pool) -> res(128) -> conv(256, pool) -> 
              conv(256, pool) -> res(256) -> pool -> fc(10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Standard ResNet-9 for CIFAR-10.
    
    Architecture (from cifar10-fast/DAWNBench):
        conv(64) -> conv(128, pool) -> res(128) -> conv(256, pool) -> 
        conv(256, pool) -> res(256) -> pool -> fc(10)
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = ResBlock(128)
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 256, pool=True)
        self.res2 = ResBlock(256)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
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
        return x


def resnet9(num_classes: int = 10) -> ResNet9:
    """Create a ResNet-9 model.
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        ResNet9 model
    """
    return ResNet9(num_classes=num_classes)
