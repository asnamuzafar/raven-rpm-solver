"""
CLEVR Models Package
- relation_network.py: Relation Network for Sort-of-CLEVR
"""
from .relation_network import SortOfCLEVRModel, BaselineCNN, CNNEncoder, RelationNetwork

__all__ = [
    'SortOfCLEVRModel',
    'BaselineCNN',
    'CNNEncoder',
    'RelationNetwork',
]
