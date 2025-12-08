"""
Stage A: Visual Encoders (Perception Layer)

All encoder implementations for RAVEN puzzles:
- SimpleConvEncoder: Custom CNN for abstract shapes
- ResNetVisualEncoder: ResNet-18 pretrained on ImageNet
- EfficientNetEncoder: EfficientNet-B0 
- DINOv2Encoder: Self-supervised ViT
- CLIPEncoder: CLIP ViT (image-text pretrained)

Best encoder for RAVEN: ResNet-18 with end-to-end fine-tuning (~28% accuracy)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class SimpleConvEncoder(nn.Module):
    """
    CNN encoder for RAVEN abstract geometric puzzles.
    Uses GroupNorm instead of BatchNorm to avoid feature collapse.
    """
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
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
    ResNet-18 encoder. Best performing encoder for RAVEN.
    Modified first conv layer for grayscale input.
    """
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Modify for grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)
                )
        
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


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder. Faster than ResNet, ~21% accuracy."""
    def __init__(self, freeze: bool = False):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()
        
        self.proj = nn.Sequential(nn.Linear(1280, 512), nn.ReLU())
        self.feature_dim = 512
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("EfficientNet encoder frozen")
        else:
            print("EfficientNet encoder trainable")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        x = x.view(B * 16, 3, x.shape[-2], x.shape[-1])
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        features = self.proj(self.model(x)).view(B, 16, -1)
        return features[:, :8], features[:, 8:]


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT encoder. Self-supervised, didn't work well for RAVEN (~14%)."""
    def __init__(self, model_name: str = "dinov2_vits14", freeze: bool = False):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        dim_map = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024}
        self.proj = nn.Linear(dim_map.get(model_name, 384), 512)
        self.feature_dim = 512
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"DINOv2 encoder frozen")
        else:
            print(f"DINOv2 encoder trainable")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        x = F.interpolate(x.view(B * 16, 3, x.shape[-2], x.shape[-1]), 
                          size=(224, 224), mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        features = self.proj(self.model(x)).view(B, 16, -1)
        return features[:, :8], features[:, 8:]


class CLIPEncoder(nn.Module):
    """CLIP ViT encoder. Requires: pip install git+https://github.com/openai/CLIP.git"""
    def __init__(self, model_name: str = "ViT-B/32", freeze: bool = False):
        super().__init__()
        try:
            import clip
        except ImportError:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model_name, device=self.device)
        self.feature_dim = 512
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("CLIP encoder frozen")
        else:
            for param in self.model.visual.parameters():
                param.requires_grad = True
            print("CLIP visual encoder trainable")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        x = F.interpolate(x.view(B * 16, 3, x.shape[-2], x.shape[-1]), 
                          size=(224, 224), mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        features = self.model.encode_image(x).view(B, 16, -1)
        return features[:, :8].float(), features[:, 8:].float()


class RAVENFeatureExtractor(nn.Module):
    """
    Main feature extractor. Processes all 16 panels (8 context + 8 choices).
    Default: ResNet-18 with fine-tuning (best results).
    """
    def __init__(self, pretrained: bool = True, freeze: bool = False, 
                 use_simple_encoder: bool = True, feature_dim: int = 256):
        super().__init__()
        
        if use_simple_encoder:
            self.encoder = SimpleConvEncoder(feature_dim=feature_dim)
            print(f"Using SimpleConvEncoder (feature_dim={feature_dim})")
        else:
            self.encoder = ResNetVisualEncoder(pretrained=pretrained)
            print(f"Using ResNetVisualEncoder (pretrained={pretrained})")
        
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen (not trainable)")
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen (trainable)")
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H, W = x.shape
        x = x.view(B * N, 1, H, W)
        features = self.encoder(x).view(B, N, -1)
        return features[:, :8, :], features[:, 8:, :]
    
    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim


def create_encoder(encoder_type: str = "resnet", freeze: bool = False):
    """
    Factory function for creating encoders.
    
    Args:
        encoder_type: 'resnet' (default/best), 'efficientnet', 'dinov2', 'clip'
        freeze: Whether to freeze encoder weights
    """
    if encoder_type == "resnet":
        return RAVENFeatureExtractor(pretrained=True, freeze=freeze, use_simple_encoder=False)
    elif encoder_type == "efficientnet":
        return EfficientNetEncoder(freeze=freeze)
    elif encoder_type == "dinov2":
        return DINOv2Encoder(freeze=freeze)
    elif encoder_type == "clip":
        return CLIPEncoder(freeze=freeze)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
