"""
Stage A: Visual Encoder (Perception Layer)

Processes the visual content of RPM puzzles.
Each image produces a 512-dimensional feature vector.

Key insight: RAVEN puzzles have abstract geometric shapes - they need
a simpler CNN trained from scratch, not ImageNet-pretrained features.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SimpleConvEncoder(nn.Module):
    """
    CNN encoder for RAVEN abstract geometric puzzles.
    Designed to produce discriminative features for each panel.
    
    Key: Avoid BatchNorm which can cause feature collapse when processing
    similar-looking images. Use GroupNorm instead.
    """
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
        # Conv stack: 160x160 -> feature_dim
        # Using GroupNorm instead of BatchNorm to preserve feature diversity
        self.conv_layers = nn.Sequential(
            # 160 -> 80
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            
            # 80 -> 40
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            
            # 40 -> 20
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            
            # 20 -> 10
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            
            # 10 -> 5
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            
            # Global average pool
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use larger initialization for better gradient flow
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
    def __init__(self, pretrained: bool = True, freeze: bool = False, use_simple_encoder: bool = True, feature_dim: int = 256):
        super().__init__()
        
        # Use simple encoder for RAVEN (abstract shapes) - trains from scratch
        # ResNet pretrained on ImageNet doesn't work well for this domain
        if use_simple_encoder:
            self.encoder = SimpleConvEncoder(feature_dim=feature_dim)
            print(f"Using SimpleConvEncoder (feature_dim={feature_dim})")
        else:
            self.encoder = ResNetVisualEncoder(pretrained=pretrained)
            print(f"Using ResNetVisualEncoder (pretrained={pretrained})")
        
        # Freeze encoder weights to prevent overfitting on small datasets
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen (not trainable)")
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen (trainable)")
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 16, H, W) - 16 panels per puzzle
        Returns:
            context_features: (B, 8, feature_dim) - features for context panels
            choice_features: (B, 8, feature_dim) - features for choice panels
        """
        B, N, H, W = x.shape
        
        # Reshape to process all panels at once
        x = x.view(B * N, 1, H, W)  # (B*16, 1, H, W)
        features = self.encoder(x)  # (B*16, feature_dim)
        features = features.view(B, N, -1)  # (B, 16, feature_dim)
        
        context_features = features[:, :8, :]   # (B, 8, feature_dim)
        choice_features = features[:, 8:, :]    # (B, 8, feature_dim)
        
        return context_features, choice_features
    
    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

