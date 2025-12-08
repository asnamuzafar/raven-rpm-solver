"""
RAVEN RPM Solver - Models Package

Organized modules:
- encoder.py: Visual encoders (ResNet, EfficientNet, DINO, CLIP)
- reasoner.py: Transformer and MLP reasoners
- baselines.py: RelationNet, CNN-Direct, Symbolic, Hybrid
- relation_network_clevr.py: CLEVR-specific Relation Network
- tokenizer.py: Symbolic attribute prediction
- rule_reasoner.py: Neuro-symbolic rule-aware reasoner
"""
from .encoder import ResNetVisualEncoder, RAVENFeatureExtractor, SimpleConvEncoder
from .tokenizer import SymbolicTokenizer, SymbolicEmbedding
from .reasoner import TransformerReasoner, MLPRelationalReasoner
from .rule_reasoner import RuleAwareReasoner, NeuroSymbolicModel, SupervisedAttributeHead
from .baselines import CNNDirectBaseline, RelationNetwork, SymbolicReasoner, HybridReasoner
from .full_model import FullRAVENModel, FullRAVENModelWithTokenizer, create_model, load_model
from .relation_network_clevr import SortOfCLEVRModel, BaselineCNN

__all__ = [
    # Encoders
    'ResNetVisualEncoder',
    'SimpleConvEncoder',
    'RAVENFeatureExtractor',
    # Reasoners
    'TransformerReasoner',
    'MLPRelationalReasoner',
    # Baselines
    'CNNDirectBaseline',
    'RelationNetwork',
    'SymbolicReasoner',
    'HybridReasoner',
    # Neuro-symbolic
    'RuleAwareReasoner',
    'NeuroSymbolicModel',
    'SupervisedAttributeHead',
    # Tokenizer
    'SymbolicTokenizer',
    'SymbolicEmbedding',
    # Full models
    'FullRAVENModel',
    'FullRAVENModelWithTokenizer',
    'create_model',
    'load_model',
    # CLEVR
    'SortOfCLEVRModel',
    'BaselineCNN',
]

