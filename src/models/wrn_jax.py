"""Wide ResNet 16-4 in Flax for CIFAR-10 with DP-SGD support.

This is the target model used in the privacy auditing experiments,
following the architecture from De et al. (2022).

Uses GroupNorm instead of BatchNorm for compatibility with per-sample
gradient computation (vmap). This matches the paper's DP-SGD setup.

Reference:
    De et al. "Unlocking high-accuracy differentially private image 
    classification through scale" (arXiv:2204.13650)
"""

import jax.numpy as jnp
from flax import linen as nn


class WideBasicBlock(nn.Module):
    """Wide ResNet basic block with GroupNorm (DP-friendly).
    
    Architecture: GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU -> Conv + shortcut
    """
    out_channels: int
    stride: int = 1
    num_groups: int = 32
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        in_channels = x.shape[-1]
        num_groups = min(self.num_groups, self.out_channels)
        in_groups = min(self.num_groups, in_channels)
        
        # Pre-activation: norm -> relu -> conv
        residual = x
        x = nn.GroupNorm(num_groups=in_groups)(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), 
                    strides=(self.stride, self.stride), padding='SAME', use_bias=False)(x)
        
        x = nn.GroupNorm(num_groups=num_groups)(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), 
                    strides=(1, 1), padding='SAME', use_bias=False)(x)
        
        # Shortcut connection
        if self.stride != 1 or in_channels != self.out_channels:
            residual = nn.Conv(self.out_channels, kernel_size=(1, 1), 
                               strides=(self.stride, self.stride), use_bias=False)(residual)
        
        return x + residual


class WideResNet(nn.Module):
    """Wide ResNet for CIFAR-10.
    
    Args:
        depth: Network depth (must satisfy (depth-4) % 6 == 0)
        widen_factor: Width multiplier for channels
        num_classes: Number of output classes
        num_groups: Number of groups for GroupNorm
    """
    depth: int = 16
    widen_factor: int = 4
    num_classes: int = 10
    num_groups: int = 32
    
    def setup(self):
        assert (self.depth - 4) % 6 == 0, "Depth must satisfy (depth-4) % 6 == 0"
        self.n_blocks = (self.depth - 4) // 6
        
        # Channel progression: [16, 16*k, 32*k, 64*k]
        self.channels = [
            16, 
            16 * self.widen_factor,   # 64 for k=4
            32 * self.widen_factor,   # 128 for k=4
            64 * self.widen_factor    # 256 for k=4
        ]
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Initial conv: 3 -> 16
        x = nn.Conv(self.channels[0], kernel_size=(3, 3), 
                    strides=(1, 1), padding='SAME', use_bias=False)(x)
        
        # Layer 1: 16 -> 64 (for k=4), stride=1
        for i in range(self.n_blocks):
            in_ch = self.channels[0] if i == 0 else self.channels[1]
            stride = 1
            x = WideBasicBlock(self.channels[1], stride=stride, 
                               num_groups=self.num_groups)(x, train)
        
        # Layer 2: 64 -> 128, stride=2 on first block
        for i in range(self.n_blocks):
            stride = 2 if i == 0 else 1
            x = WideBasicBlock(self.channels[2], stride=stride,
                               num_groups=self.num_groups)(x, train)
        
        # Layer 3: 128 -> 256, stride=2 on first block
        for i in range(self.n_blocks):
            stride = 2 if i == 0 else 1
            x = WideBasicBlock(self.channels[3], stride=stride,
                               num_groups=self.num_groups)(x, train)
        
        # Final norm and pool
        num_groups = min(self.num_groups, self.channels[3])
        x = nn.GroupNorm(num_groups=num_groups)(x)
        x = nn.relu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classifier
        x = nn.Dense(self.num_classes)(x)
        
        return x


def create_wrn16_4(num_classes: int = 10) -> WideResNet:
    """Create a Wide ResNet 16-4 model with GroupNorm.
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        WideResNet 16-4 model (depth=16, widen_factor=4)
    """
    return WideResNet(depth=16, widen_factor=4, num_classes=num_classes)


def init_wrn16_4(rng, num_classes: int = 10, input_shape=(1, 32, 32, 3)):
    """Initialize WRN-16-4 model and return (model, variables).
    
    Args:
        rng: JAX random key
        num_classes: Number of output classes
        input_shape: Input shape (batch, height, width, channels)
    
    Returns:
        Tuple of (model, variables dict with 'params')
    """
    model = create_wrn16_4(num_classes)
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input, train=True)
    return model, variables
