"""
Full RAVEN Model

Combines the visual encoder with a reasoning module for end-to-end training.
"""
import torch
import torch.nn as nn
from typing import Optional

from .encoder import RAVENFeatureExtractor
from .reasoner import TransformerReasoner, MLPRelationalReasoner
from .baselines import CNNDirectBaseline, RelationNetwork, HybridReasoner


class FullRAVENModel(nn.Module):
    """
    Complete end-to-end model: Encoder + Reasoner
    """
    def __init__(self, encoder: nn.Module, reasoner: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.reasoner = reasoner
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, H, W) - 16 panels per puzzle
        Returns:
            logits: (B, 8) - score for each choice
        """
        ctx_feat, choice_feat = self.encoder(x)
        logits = self.reasoner(ctx_feat, choice_feat)
        return logits
    
    def get_features(self, x: torch.Tensor) -> tuple:
        """Extract features without reasoning (useful for analysis)"""
        return self.encoder(x)


def create_model(
    model_type: str = 'transformer',
    pretrained_encoder: bool = True,
    freeze_encoder: bool = False,
    feature_dim: int = 512,
    hidden_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
) -> FullRAVENModel:
    """
    Factory function to create different model configurations.
    
    Args:
        model_type: One of 'transformer', 'mlp', 'cnn_direct', 'relation_net', 'hybrid'
        pretrained_encoder: Whether to use pretrained ResNet weights
        freeze_encoder: Whether to freeze encoder weights (prevents overfitting)
        feature_dim: Dimension of visual features (default 512 for ResNet-18)
        hidden_dim: Hidden dimension for reasoning modules
        num_heads: Number of attention heads (for transformer)
        num_layers: Number of transformer layers
        dropout: Dropout rate
        
    Returns:
        FullRAVENModel instance
    """
    # Create encoder (optionally frozen for small datasets)
    encoder = RAVENFeatureExtractor(pretrained=pretrained_encoder, freeze=freeze_encoder)
    
    # Create reasoner based on type
    if model_type == 'transformer':
        reasoner = TransformerReasoner(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )
    elif model_type == 'mlp':
        reasoner = MLPRelationalReasoner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
    elif model_type == 'cnn_direct':
        reasoner = CNNDirectBaseline(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
    elif model_type == 'relation_net':
        reasoner = RelationNetwork(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim // 2
        )
    elif model_type == 'hybrid':
        reasoner = HybridReasoner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim // 2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return FullRAVENModel(encoder, reasoner)


def load_model(
    checkpoint_path: str,
    model_type: str = 'transformer',
    device: str = 'cuda'
) -> FullRAVENModel:
    """
    Load a saved model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_type: Type of model architecture
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model architecture
    model = create_model(model_type=model_type, pretrained_encoder=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

