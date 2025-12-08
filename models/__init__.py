"""
RAVEN RPM Solver - Models Package

Organized into subpackages:
- clevr/: Sort-of-CLEVR models (Relation Network)
- iraven/: I-RAVEN models (encoders, reasoners, baselines)
"""
# Re-export from subpackages for convenience
from .clevr import SortOfCLEVRModel, BaselineCNN
from .iraven import (
    # Encoders
    ResNetVisualEncoder,
    SimpleConvEncoder,
    RAVENFeatureExtractor,
    # Reasoners
    TransformerReasoner,
    MLPRelationalReasoner,
    # Baselines
    CNNDirectBaseline,
    RelationNetwork,
    SymbolicReasoner,
    HybridReasoner,
    # Neuro-symbolic
    RuleAwareReasoner,
    NeuroSymbolicModel,
    SupervisedAttributeHead,
    # Tokenizer
    SymbolicTokenizer,
    SymbolicEmbedding,
    # Full models
    FullRAVENModel,
    FullRAVENModelWithTokenizer,
    create_model,
    load_model,
)

__all__ = [
    # CLEVR
    'SortOfCLEVRModel',
    'BaselineCNN',
    # I-RAVEN Encoders
    'ResNetVisualEncoder',
    'SimpleConvEncoder',
    'RAVENFeatureExtractor',
    # I-RAVEN Reasoners
    'TransformerReasoner',
    'MLPRelationalReasoner',
    # I-RAVEN Baselines
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
]
