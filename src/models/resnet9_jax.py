"""ResNet-9 model in Flax for CIFAR-10.

Standard ResNet-9 architecture matching the PyTorch version.
Architecture: conv(64) -> conv(128, pool) -> res(128) -> conv(256, pool) -> 
              conv(256, pool) -> res(256) -> pool -> fc(10)
"""

import jax.numpy as jnp
from flax import linen as nn


class ConvBlock(nn.Module):
    """Convolution block: Conv -> BatchNorm -> ReLU -> optional MaxPool."""
    features: int
    pool: bool = False
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.pool:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class ResBlock(nn.Module):
    """Residual block: two conv layers with skip connection."""
    features: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x + residual)
        return x


class ResNet9(nn.Module):
    """ResNet-9 for CIFAR-10."""
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
    """Create a ResNet-9 model."""
    return ResNet9(num_classes=num_classes)


def init_resnet9(rng, num_classes: int = 10, input_shape=(1, 32, 32, 3)):
    """Initialize ResNet-9 model and return (model, params).
    
    Args:
        rng: JAX random key
        num_classes: Number of output classes
        input_shape: Input shape (batch, height, width, channels)
    
    Returns:
        Tuple of (model, params)
    """
    model = create_resnet9(num_classes)
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input, train=False)
    return model, variables
