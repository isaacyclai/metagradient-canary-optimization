"""Wide ResNet 16-4 model for DP-SGD training and auditing.

This is the target model used in the privacy auditing experiments,
following the architecture from De et al. (2022).

Reference:
    De et al. "Unlocking high-accuracy differentially private image 
    classification through scale" (arXiv:2204.13650)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WideBasicBlock(nn.Module):
    """Wide ResNet basic block with GroupNorm (DP-friendly)."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        use_group_norm: bool = True,
        num_groups: int = 32
    ):
        super().__init__()
        
        # Use GroupNorm for DP training (no batch statistics)
        if use_group_norm:
            self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
            self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                       stride=stride, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """Wide ResNet for CIFAR-10.
    
    Args:
        depth: Network depth (must satisfy (depth-4) % 6 == 0)
        widen_factor: Width multiplier for channels
        num_classes: Number of output classes
        use_group_norm: Use GroupNorm instead of BatchNorm (for DP training)
        num_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        num_classes: int = 10,
        use_group_norm: bool = True,
        num_groups: int = 32
    ):
        super().__init__()
        
        assert (depth - 4) % 6 == 0, "Depth must satisfy (depth-4) % 6 == 0"
        n = (depth - 4) // 6
        
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, 
                               padding=1, bias=False)
        
        self.layer1 = self._make_layer(channels[0], channels[1], n, stride=1,
                                        use_group_norm=use_group_norm, 
                                        num_groups=num_groups)
        self.layer2 = self._make_layer(channels[1], channels[2], n, stride=2,
                                        use_group_norm=use_group_norm,
                                        num_groups=num_groups)
        self.layer3 = self._make_layer(channels[2], channels[3], n, stride=2,
                                        use_group_norm=use_group_norm,
                                        num_groups=num_groups)
        
        if use_group_norm:
            self.norm = nn.GroupNorm(min(num_groups, channels[3]), channels[3])
        else:
            self.norm = nn.BatchNorm2d(channels[3])
        
        self.fc = nn.Linear(channels[3], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int,
        stride: int,
        use_group_norm: bool,
        num_groups: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(WideBasicBlock(in_ch, out_channels, s, 
                                         use_group_norm, num_groups))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.norm(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def wrn16_4(num_classes: int = 10, use_group_norm: bool = True) -> WideResNet:
    """Create a Wide ResNet 16-4 model.
    
    Args:
        num_classes: Number of output classes
        use_group_norm: Use GroupNorm for DP training
    
    Returns:
        WideResNet 16-4 model
    """
    return WideResNet(depth=16, widen_factor=4, num_classes=num_classes,
                      use_group_norm=use_group_norm)
