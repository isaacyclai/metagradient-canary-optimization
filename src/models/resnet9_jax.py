"""ResNet-9 model in Flax for CIFAR-10 with DP-SGD support.

Uses GroupNorm instead of BatchNorm for compatibility with per-sample
gradient computation (vmap). BatchNorm computes statistics across the
batch which is incompatible with DP-SGD's per-sample gradient clipping.

Architecture: conv(64) -> conv(128, pool) -> res(128) -> conv(256, pool) -> 
              conv(256, pool) -> res(256) -> pool -> fc(10)
"""

import jax.numpy as jnp
from flax import linen as nn


class ConvBlock(nn.Module):
    """Convolution block: Conv -> GroupNorm -> ReLU -> optional MaxPool."""
    features: int
    pool: bool = False
    num_groups: int = 8
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        # Use GroupNorm instead of BatchNorm for DP-SGD compatibility
        # GroupNorm normalizes within each sample, not across the batch
        num_groups = min(self.num_groups, self.features)
        x = nn.GroupNorm(num_groups=num_groups)(x)
        x = nn.relu(x)
        if self.pool:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class ResBlock(nn.Module):
    """Residual block: two conv layers with skip connection."""
    features: int
    num_groups: int = 8
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        num_groups = min(self.num_groups, self.features)
        
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.GroupNorm(num_groups=num_groups)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.GroupNorm(num_groups=num_groups)(x)
        x = nn.relu(x + residual)
        return x


class ResNet9(nn.Module):
    """ResNet-9 for CIFAR-10 with GroupNorm (DP-SGD compatible)."""
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # conv1: 3 -> 64
        x = ConvBlock(64)(x, train)
        
        # conv2: 64 -> 128, pool
        x = ConvBlock(128, pool=True)(x, train)
        
        # res1: 128 -> 128
        x = ResBlock(128)(x, train)
        
        # conv3: 128 -> 256, pool
        x = ConvBlock(256, pool=True)(x, train)
        
        # conv4: 256 -> 256, pool
        x = ConvBlock(256, pool=True)(x, train)
        
        # res2: 256 -> 256
        x = ResBlock(256)(x, train)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classifier
        x = nn.Dense(self.num_classes)(x)
        
        return x


def create_resnet9(num_classes: int = 10):
    """Create a ResNet-9 model with GroupNorm."""
    return ResNet9(num_classes=num_classes)


def init_resnet9(rng, num_classes: int = 10, input_shape=(1, 32, 32, 3)):
    """Initialize ResNet-9 model and return (model, variables).
    
    Args:
        rng: JAX random key
        num_classes: Number of output classes
        input_shape: Input shape (batch, height, width, channels)
    
    Returns:
        Tuple of (model, variables dict with 'params')
    """
    model = create_resnet9(num_classes)
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input, train=True)
    return model, variables
