"""
HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics

A novel framework that decomposes long documents into segments, constructs
local semantic graphs, and maintains a hierarchical global memory for
efficient semantic understanding.
"""

__version__ = "1.0.0"
__author__ = "Dong Liu, Yanxuan Yu"

from .core import HSGMModel
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

__all__ = [
    "HSGMModel",
    "LocalSemanticGraph", 
    "HierarchicalMemory",
    "IncrementalUpdater",
    "QueryProcessor",
    "DocumentSegmenter",
    "SimilarityComputer",
    "GraphAggregator"
]
