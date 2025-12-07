"""
Stage A: Spatial Visual Encoder (Perception Layer)

This module implements a Spatial ResNet encoder that preserves spatial information
critical for solving RAVEN "Position" rules.

Modifications from standard ResNet:
1. CoordConv: Adds x, y coordinate channels to the input (1 channel image -> 3 channel input).
2. No Global Pooling: output is (B, 512, 5, 5) instead of (B, 512).
3. Adapters: Optional 1x1 conv to reduce channel dimension.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class SpatialResNet(nn.Module):
    """
    ResNet-18 that preserves spatial dimensions (5x5 grid).
    Input: (B, 1, 160, 160) grayscale image
    Output: (B, 512, 5, 5) spatial features
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Standard ResNet-18 expects 3 channels. 
        # We will feed it (Image, X-coord, Y-coord).
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Remove the classification head (avgpool + fc)
        del self.resnet.avgpool
        del self.resnet.fc
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 160, 160) grayscale images
        Returns:
            features: (B, 512, 5, 5)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Add Coordinate Channels (CoordConv)
        # X coordinates (vary across W dim)
        xx = torch.linspace(-1, 1, W, device=device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Y coordinates (vary across H dim)
        yy = torch.linspace(-1, 1, H, device=device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        
        # Concatenate: (B, 1, H, W) + (B, 1, H, W) + (B, 1, H, W) -> (B, 3, H, W)
        x_coords = torch.cat([x, xx, yy], dim=1)
        
        # 2. ResNet Forward Pass
        # We use the sub-modules directly to skip avgpool
        x = self.resnet.conv1(x_coords)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Output is (B, 512, 5, 5) for 160x160 input
        return x


class SpatialFeatureExtractor(nn.Module):
    """
    Wrapper for handling RAVEN panel logic (Context + Choices).
    Includes a Spatial Adapter to compress 5x5 spatial features into a vector 
    while preserving positional information (unlike avgpool).
    """
    def __init__(self, pretrained: bool = True, freeze: bool = False, feature_dim: int = 512):
        super().__init__()
        self.encoder = SpatialResNet(pretrained=pretrained)
        self.out_channels = 512
        self.grid_size = 5
        
        # Spatial Adapter: Project (512, 5, 5) -> (512) vector
        # 1. Reduce channels 512 -> 64 to save parameters
        self.compress_conv = nn.Conv2d(512, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Flatten 64 * 5 * 5 = 1600 spatial features
        # 3. Project to target feature_dim (e.g. 512)
        self.fc = nn.Linear(64 * 5 * 5, feature_dim)
        
        self.feature_dim = feature_dim
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (B, 16, 160, 160) or (B, 16, 1, 160, 160)
        Output: 
            context: (B, 8, feature_dim)
            choices: (B, 8, feature_dim)
        """
        if x.dim() == 4:
            B, N, H, W = x.shape
            # Add channel dimension: (B*N, 1, H, W)
            x_flat = x.view(B * N, 1, H, W)
        else:
            B, N, C, H, W = x.shape
            x_flat = x.view(B * N, C, H, W)
        
        # Extract spatial features: (B*N, 512, 5, 5)
        features = self.encoder(x_flat) 
        
        # Spatial Adapter
        x = self.compress_conv(features) # (B*N, 64, 5, 5)
        x = self.relu(x)
        x = x.view(x.size(0), -1)        # (B*N, 1600)
        x = self.fc(x)                   # (B*N, 512)
        
        # Reshape to (B, N, D)
        features = x.view(B, N, -1)
        return features[:, :8], features[:, 8:]

