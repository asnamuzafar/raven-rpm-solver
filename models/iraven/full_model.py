"""
Full RAVEN Model

Combines all stages as per goal.md:
- Stage A: Visual encoder (ResNet-18)
- Stage B: Tokenizer (symbolic attribute extraction)
- Stage C/D: Reasoning modules
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from .encoder import RAVENFeatureExtractor
from .encoder_spatial import SpatialFeatureExtractor
from .tokenizer import SymbolicTokenizer
from .reasoner import TransformerReasoner, MLPRelationalReasoner
from .rule_reasoner import RuleAwareReasoner, NeuroSymbolicModel
from .rule_reasoner_pos import RuleAwareReasonerPos
from .reasoner_spatial_conv import SpatialConvolutionalReasoner
from .baselines import CNNDirectBaseline, RelationNetwork, HybridReasoner


class FullRAVENModel(nn.Module):
    """
    Complete end-to-end model: Encoder + Reasoner
    Implements Stages A + C/D from goal.md
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
    
    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features without reasoning (useful for analysis)"""
        return self.encoder(x)


class FullRAVENModelWithTokenizer(nn.Module):
    """
    Complete model including Stage B tokenizer for symbolic extraction.
    This model can:
    1. Extract visual features (Stage A)
    2. Convert to symbolic attributes (Stage B)
    3. Perform reasoning (Stage C/D)
    """
    def __init__(
        self, 
        encoder: nn.Module, 
        tokenizer: nn.Module,
        reasoner: nn.Module,
        use_symbolic: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.reasoner = reasoner
        self.use_symbolic = use_symbolic
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, H, W) - 16 panels per puzzle
        Returns:
            logits: (B, 8) - score for each choice
        """
        # Stage A: Visual encoding
        ctx_feat, choice_feat = self.encoder(x)
        
        # Stage B: Tokenization (symbolic attributes)
        # Concatenate features for tokenization
        all_features = torch.cat([ctx_feat, choice_feat], dim=1)  # (B, 16, 512)
        symbolic_attrs = self.tokenizer(all_features)  # dict of (B, 16, num_classes)
        
        # Stage C/D: Reasoning
        if isinstance(self.reasoner, HybridReasoner) and self.use_symbolic:
            # Hybrid model can use symbolic attributes
            context_attrs, choice_attrs = self._convert_to_attr_lists(symbolic_attrs)
            logits = self.reasoner(
                ctx_feat, choice_feat,
                context_attrs=context_attrs,
                choice_attrs=choice_attrs
            )
        else:
            logits = self.reasoner(ctx_feat, choice_feat)
        
        return logits
    
    def _convert_to_attr_lists(
        self, 
        symbolic_attrs: Dict[str, torch.Tensor]
    ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """Convert symbolic attribute tensors to lists for symbolic reasoner."""
        B = symbolic_attrs['shape'].shape[0]
        
        context_attrs = []
        choice_attrs = []
        
        attr_names = {
            'shape': self.tokenizer.shape_names,
            'size': self.tokenizer.size_names,
            'color': self.tokenizer.color_names,
            'count': self.tokenizer.count_names,
            'position': self.tokenizer.position_names
        }
        
        for b in range(B):
            ctx_list = []
            ch_list = []
            
            for i in range(16):
                attrs = {}
                for attr, names in attr_names.items():
                    idx = symbolic_attrs[attr][b, i].argmax().item()
                    attrs[attr] = names[idx]
                
                if i < 8:
                    ctx_list.append(attrs)
                else:
                    ch_list.append(attrs)
            
            context_attrs.append(ctx_list)
            choice_attrs.append(ch_list)
        
        return context_attrs, choice_attrs
    
    def get_symbolic_representation(self, x: torch.Tensor) -> Dict[str, List]:
        """
        Get human-readable symbolic representation of all panels.
        Useful for visualization and debugging.
        """
        ctx_feat, choice_feat = self.encoder(x)
        all_features = torch.cat([ctx_feat, choice_feat], dim=1)
        return self.tokenizer.to_symbolic(all_features)
    
    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features without reasoning"""
        return self.encoder(x)


def create_model(
    model_type: str = 'transformer',
    pretrained_encoder: bool = True,
    freeze_encoder: bool = False,
    use_simple_encoder: bool = True,
    encoder_type: str = 'resnet',  # 'resnet', 'simple', 'spatial'
    feature_dim: int = 256,
    hidden_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    include_tokenizer: bool = False
) -> nn.Module:
    """
    Factory function to create different model configurations.
    
    Args:
        model_type: One of 'transformer', 'mlp', 'cnn_direct', 'relation_net', 'hybrid', 'neuro_symbolic'
        pretrained_encoder: Whether to use pretrained ResNet weights (only for ResNet encoder)
        freeze_encoder: Whether to freeze encoder weights (prevents overfitting)
        use_simple_encoder: Use SimpleConvEncoder (better for RAVEN) vs ResNet
        feature_dim: Dimension of visual features (256 for simple encoder, 512 for ResNet)
        hidden_dim: Hidden dimension for reasoning modules
        num_heads: Number of attention heads (for transformer)
        num_layers: Number of transformer layers
        dropout: Dropout rate
        include_tokenizer: If True, return FullRAVENModelWithTokenizer
        
    Returns:
        FullRAVENModel or FullRAVENModelWithTokenizer instance
    """
    # Stage A: Create encoder
    if encoder_type == 'spatial':
        # New Spatial ResNet with CoordConv (Phase 4)
        encoder = SpatialFeatureExtractor(
            pretrained=pretrained_encoder,
            freeze=freeze_encoder,
            feature_dim=feature_dim,
            flatten_output=(model_type != 'spatial_conv') # Don't flatten for Conv Reasoner
        )
    else:
        # Backward compatibility for use_simple_encoder flag
        use_simple = use_simple_encoder or (encoder_type == 'simple')
        
        encoder = RAVENFeatureExtractor(
            pretrained=pretrained_encoder, 
            freeze=freeze_encoder,
            use_simple_encoder=use_simple,
            feature_dim=feature_dim
        )
    
    # Stage B: Create tokenizer (optional)
    tokenizer = SymbolicTokenizer(feature_dim=feature_dim, hidden_dim=hidden_dim // 2) if include_tokenizer else None
    
    # Stage C/D: Create reasoner based on type
    if model_type == 'transformer':
        reasoner = TransformerReasoner(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
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
    elif model_type == 'neuro_symbolic':
        # SOTA-inspired: Rule-aware neuro-symbolic reasoner (BEST: 32.9%)
        reasoner = RuleAwareReasoner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif model_type == 'neuro_symbolic_pos':
        # Enhanced reasoner with Position supervision (Phase 4)
        reasoner = RuleAwareReasonerPos(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif model_type == 'spatial_conv':
        # Phase 5: Convolutional Reasoner (No flattening)
        reasoner = SpatialConvolutionalReasoner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Return appropriate model class
    if include_tokenizer and tokenizer is not None:
        return FullRAVENModelWithTokenizer(
            encoder, tokenizer, reasoner,
            use_symbolic=(model_type == 'hybrid')
        )
    
    return FullRAVENModel(encoder, reasoner)


def load_model(
    checkpoint_path: str,
    model_type: str = 'transformer',
    device: str = 'cuda',
    include_tokenizer: bool = False
) -> nn.Module:
    """
    Load a saved model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_type: Type of model architecture
        device: Device to load model on
        include_tokenizer: Whether to include tokenizer module
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model architecture
    model = create_model(
        model_type=model_type, 
        pretrained_encoder=False,
        include_tokenizer=include_tokenizer
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    return model
