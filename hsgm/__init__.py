"""
HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics

A novel framework that decomposes long documents into segments, constructs
local semantic graphs, and maintains a hierarchical global memory for
efficient semantic understanding.
"""

__version__ = "1.0.0"
__author__ = "Dong Liu, Yanxuan Yu"

from .core import HSGMModel, HSGMForClassification, HSGMForGeneration
from .components import (
    LocalSemanticGraph,
    HierarchicalMemory,
    IncrementalUpdater,
    QueryProcessor
)
from .utils import (
    DocumentSegmenter,
    SimilarityComputer,
    GraphAggregator
)

try:
    from .hsgm_ops import HSGMOps, get_hsgm_ops
    __all_hsgm_ops__ = ["HSGMOps", "get_hsgm_ops"]
except ImportError:
    __all_hsgm_ops__ = []

__all__ = [
    "HSGMModel",
    "HSGMForClassification",
    "HSGMForGeneration",
    "LocalSemanticGraph", 
    "HierarchicalMemory",
    "IncrementalUpdater",
    "QueryProcessor",
    "DocumentSegmenter",
    "SimilarityComputer",
    "GraphAggregator"
] + __all_hsgm_ops__
