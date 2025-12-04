"""
Stage A: Visual Encoder (Perception Layer)

Processes the visual content of RPM puzzles using ResNet-18.
Each image produces a 512-dimensional feature vector.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetVisualEncoder(nn.Module):
    """
    Visual Encoder using ResNet-18 backbone.
    Converts each 160x160 grayscale image to a 512-dim feature vector.
    """
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        
        # Load pretrained ResNet-18
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Modify first conv layer to accept 1-channel grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize with mean of pretrained RGB weights
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)
                )
        
        # Copy rest of ResNet layers (excluding final FC)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) grayscale image tensor
        Returns:
            features: (B, 512) feature vector
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class RAVENFeatureExtractor(nn.Module):
    """
    Full Stage A: Extract features from all 16 panels (8 context + 8 choices)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetVisualEncoder(pretrained=pretrained)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 16, H, W) - 16 panels per puzzle
        Returns:
            context_features: (B, 8, 512) - features for context panels
            choice_features: (B, 8, 512) - features for choice panels
        """
        B, N, H, W = x.shape
        
        # Reshape to process all panels at once
        x = x.view(B * N, 1, H, W)  # (B*16, 1, H, W)
        features = self.encoder(x)  # (B*16, 512)
        features = features.view(B, N, -1)  # (B, 16, 512)
        
        context_features = features[:, :8, :]   # (B, 8, 512)
        choice_features = features[:, 8:, :]    # (B, 8, 512)
        
        return context_features, choice_features
    
    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

