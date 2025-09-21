"""
Core components of the HSGM framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import math

@dataclass
class GraphNode:
    """Represents a node in the semantic graph."""
    id: int
    embedding: torch.Tensor
    position: int
    token: str
    segment_id: int

@dataclass
class GraphEdge:
    """Represents an edge in the semantic graph."""
    source: int
    target: int
    weight: float
    similarity: float

class LocalSemanticGraph:
    """
    Constructs local semantic graphs for document segments.
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 local_threshold: float = 0.2,
                 adaptive_threshold: bool = True,
                 alpha: float = 1.0,
                 beta: float = 0.5):
        """
        Initialize Local Semantic Graph constructor.
        
        Args:
            hidden_dim: Embedding dimension
            local_threshold: Similarity threshold for edge creation
            adaptive_threshold: Whether to use adaptive thresholding
            alpha: Mean weight for adaptive threshold
            beta: Std weight for adaptive threshold
        """
        self.hidden_dim = hidden_dim
        self.local_threshold = local_threshold
        self.adaptive_threshold = adaptive_threshold
        self.alpha = alpha
        self.beta = beta
        
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between embeddings."""
        # Normalize embeddings
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        return torch.dot(emb1_norm, emb2_norm).item()
    
    def adaptive_thresholding(self, similarities: List[float]) -> float:
        """
        Compute adaptive threshold based on similarity distribution.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            Adaptive threshold value
        """
        if not similarities:
            return self.local_threshold
            
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        return self.alpha * mean_sim + self.beta * std_sim
    
    def build_graph(self, 
                   tokens: List[str],
                   embeddings: torch.Tensor,
                   segment_id: int) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Build local semantic graph for a segment.
        
        Args:
            tokens: List of tokens in the segment
            embeddings: Token embeddings [seq_len, hidden_dim]
            segment_id: ID of the segment
            
        Returns:
            Tuple of (nodes, edges)
        """
        seq_len = embeddings.size(0)
        
        # Create nodes
        nodes = []
        for i, (token, emb) in enumerate(zip(tokens, embeddings)):
            node = GraphNode(
                id=segment_id * 10000 + i,  # Unique node ID
                embedding=emb,
                position=i,
                token=token,
                segment_id=segment_id
            )
            nodes.append(node)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                sim = self.compute_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Determine threshold
        if self.adaptive_threshold:
            threshold = self.adaptive_thresholding(similarities)
        else:
            threshold = self.local_threshold
        
        # Create edges based on threshold
        edges = []
        edge_idx = 0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if edge_idx < len(similarities) and i < len(nodes) and j < len(nodes):
                    sim = similarities[edge_idx]
                    if sim >= threshold:
                        edge = GraphEdge(
                            source=nodes[i].id,
                            target=nodes[j].id,
                            weight=sim,
                            similarity=sim
                        )
                        edges.append(edge)
                    edge_idx += 1
        
        return nodes, edges

class HierarchicalMemory:
    """
    Maintains hierarchical graph memory with summary nodes.
    """
    
    def __init__(self,
                 hidden_dim: int = 768,
                 global_threshold: float = 0.1,
                 summary_dim: int = 256,
                 num_attention_heads: int = 8):
        """
        Initialize Hierarchical Memory.
        
        Args:
            hidden_dim: Embedding dimension
            global_threshold: Threshold for global graph edges
            summary_dim: Dimension of summary nodes
            num_attention_heads: Number of attention heads
        """
        self.hidden_dim = hidden_dim
        self.global_threshold = global_threshold
        self.summary_dim = summary_dim
        self.num_attention_heads = num_attention_heads
        
        # Summary node aggregator
        self.summary_aggregator = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Summary projection
        self.summary_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, summary_dim),
            nn.LayerNorm(summary_dim)
        )
        
        # Global graph components
        self.global_nodes = []
        self.global_edges = []
        
    def create_summary_node(self, 
                          local_graph: Tuple[List[GraphNode], List[GraphEdge]],
                          previous_summaries: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Create summary node from local graph.
        
        Args:
            local_graph: Tuple of (nodes, edges) from local graph
            previous_summaries: Previous summary nodes for cross-attention
            
        Returns:
            Summary node embedding
        """
        nodes, edges = local_graph
        
        if not nodes:
            return torch.zeros(self.summary_dim)
        
        # Extract embeddings
        embeddings = torch.stack([node.embedding for node in nodes])
        
        # Compute mean and max pooling
        mean_pool = torch.mean(embeddings, dim=0)
        max_pool, _ = torch.max(embeddings, dim=0)
        
        # Cross-attention with previous summaries
        if previous_summaries is not None and len(previous_summaries) > 0:
            prev_summaries = torch.stack(previous_summaries).unsqueeze(0)
            embeddings_expanded = embeddings.unsqueeze(0)
            
            attended, _ = self.summary_aggregator(
                embeddings_expanded, 
                prev_summaries, 
                prev_summaries
            )
            cross_attn = attended.squeeze(0).mean(dim=0)
        else:
            cross_attn = torch.zeros(self.hidden_dim)
        
        # Combine representations
        combined = torch.cat([mean_pool, max_pool], dim=-1)
        
        # Project to summary dimension
        summary = self.summary_projection(combined)
        
        return summary
    
    def update_global_graph(self, new_summary: torch.Tensor):
        """
        Update global graph with new summary node.
        
        Args:
            new_summary: New summary node embedding
        """
        # Add new summary node
        self.global_nodes.append(new_summary)
        
        # Compute similarities with existing nodes
        new_edges = []
        for i, existing_summary in enumerate(self.global_nodes[:-1]):
            similarity = F.cosine_similarity(
                new_summary.unsqueeze(0), 
                existing_summary.unsqueeze(0)
            ).item()
            
            if similarity >= self.global_threshold:
                new_edges.append((i, len(self.global_nodes) - 1, similarity))
        
        # Add new edges
        self.global_edges.extend(new_edges)
    
    def get_top_k_similar(self, query: torch.Tensor, k: int = 5) -> List[int]:
        """
        Get top-k most similar summary nodes.
        
        Args:
            query: Query embedding
            k: Number of top similar nodes to return
            
        Returns:
            List of indices of top-k similar nodes
        """
        if not self.global_nodes:
            return []
        
        similarities = []
        for i, summary in enumerate(self.global_nodes):
            sim = F.cosine_similarity(
                query.unsqueeze(0),
                summary.unsqueeze(0)
            ).item()
            similarities.append((i, sim))
        
        # Sort by similarity and return top-k indices
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:k]]

class IncrementalUpdater:
    """
    Handles incremental updates for streaming documents.
    """
    
    def __init__(self,
                 cache_size: int = 1000,
                 cache_eviction_policy: str = "lru"):
        """
        Initialize Incremental Updater.
        
        Args:
            cache_size: Maximum cache size
            cache_eviction_policy: Cache eviction policy
        """
        self.cache_size = cache_size
        self.cache_eviction_policy = cache_eviction_policy
        self.edge_cache = {}
        self.node_cache = {}
        self.access_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_cache(self, segment_id: int, nodes: List[GraphNode], edges: List[GraphEdge]):
        """
        Update cache with new segment data.
        
        Args:
            segment_id: ID of the segment
            nodes: List of nodes
            edges: List of edges
        """
        # Store in cache
        self.node_cache[segment_id] = nodes
        self.edge_cache[segment_id] = edges
        self.access_times[segment_id] = len(self.access_times)
        
        # Evict if cache is full
        if len(self.node_cache) > self.cache_size:
            self._evict_cache()
    
    def _evict_cache(self):
        """Evict cache entries based on policy."""
        if self.cache_eviction_policy == "lru":
            # Remove least recently used
            oldest_segment = min(self.access_times.keys(), 
                               key=lambda x: self.access_times[x])
            del self.node_cache[oldest_segment]
            del self.edge_cache[oldest_segment]
            del self.access_times[oldest_segment]
    
    def get_cached_segment(self, segment_id: int) -> Optional[Tuple[List[GraphNode], List[GraphEdge]]]:
        """
        Get cached segment data.
        
        Args:
            segment_id: ID of the segment
            
        Returns:
            Cached (nodes, edges) or None if not cached
        """
        if segment_id in self.node_cache:
            self.cache_hits += 1
            self.access_times[segment_id] = len(self.access_times)
            return self.node_cache[segment_id], self.edge_cache[segment_id]
        else:
            self.cache_misses += 1
            return None
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses == 0:
            return {"hit_rate": 0.0, "miss_rate": 0.0}
        
        return {
            "hit_rate": self.cache_hits / total_accesses,
            "miss_rate": self.cache_misses / total_accesses,
            "cache_size": len(self.node_cache)
        }

class QueryProcessor:
    """
    Processes queries using hierarchical retrieval and local reasoning.
    """
    
    def __init__(self,
                 hidden_dim: int = 768,
                 num_gcn_layers: int = 3,
                 gcn_dropout: float = 0.1):
        """
        Initialize Query Processor.
        
        Args:
            hidden_dim: Embedding dimension
            num_gcn_layers: Number of GCN layers
            gcn_dropout: Dropout for GCN layers
        """
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        
        # GCN layers for local reasoning
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        
        self.dropout = nn.Dropout(gcn_dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def local_graph_reasoning(self, 
                            nodes: List[GraphNode],
                            edges: List[GraphEdge],
                            query: torch.Tensor) -> torch.Tensor:
        """
        Perform local graph reasoning using GCN.
        
        Args:
            nodes: List of nodes in the graph
            edges: List of edges in the graph
            query: Query embedding
            
        Returns:
            Updated node representations
        """
        if not nodes:
            return torch.zeros(self.hidden_dim)
        
        # Initialize node representations
        node_embeddings = torch.stack([node.embedding for node in nodes])
        
        # Build adjacency matrix
        num_nodes = len(nodes)
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        
        for edge in edges:
            source_idx = next(i for i, node in enumerate(nodes) if node.id == edge.source)
            target_idx = next(i for i, node in enumerate(nodes) if node.id == edge.target)
            adj_matrix[source_idx, target_idx] = edge.weight
            adj_matrix[target_idx, source_idx] = edge.weight
        
        # Normalize adjacency matrix
        degree = torch.sum(adj_matrix, dim=1)
        degree[degree == 0] = 1  # Avoid division by zero
        degree_inv_sqrt = torch.pow(degree, -0.5)
        norm_adj = torch.diag(degree_inv_sqrt) @ adj_matrix @ torch.diag(degree_inv_sqrt)
        
        # GCN layers
        h = node_embeddings
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Message passing
            h_new = norm_adj @ h
            h_new = gcn_layer(h_new)
            
            # Residual connection
            if i > 0:
                h_new = h_new + h
            
            h_new = self.layer_norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            h = h_new
        
        # Aggregate node representations
        return torch.mean(h, dim=0)
    
    def hierarchical_query_processing(self,
                                    query: torch.Tensor,
                                    hierarchical_memory: HierarchicalMemory,
                                    local_graphs: Dict[int, Tuple[List[GraphNode], List[GraphEdge]]],
                                    top_k: int = 5) -> torch.Tensor:
        """
        Process query using hierarchical retrieval and local reasoning.
        
        Args:
            query: Query embedding
            hierarchical_memory: Hierarchical memory instance
            local_graphs: Dictionary mapping segment_id to (nodes, edges)
            top_k: Number of top segments to retrieve
            
        Returns:
            Query result embedding
        """
        # Retrieve top-k similar summary nodes
        top_k_indices = hierarchical_memory.get_top_k_similar(query, top_k)
        
        if not top_k_indices:
            return torch.zeros(self.hidden_dim)
        
        # Local reasoning on retrieved segments
        results = []
        attention_weights = []
        
        for segment_idx in top_k_indices:
            if segment_idx in local_graphs:
                nodes, edges = local_graphs[segment_idx]
                local_result = self.local_graph_reasoning(nodes, edges, query)
                results.append(local_result)
                
                # Compute attention weight
                if segment_idx < len(hierarchical_memory.global_nodes):
                    summary_node = hierarchical_memory.global_nodes[segment_idx]
                    attention_weight = F.cosine_similarity(
                        query.unsqueeze(0),
                        summary_node.unsqueeze(0)
                    ).item()
                    attention_weights.append(attention_weight)
        
        if not results:
            return torch.zeros(self.hidden_dim)
        
        # Combine results with attention weights
        results_tensor = torch.stack(results)
        
        if attention_weights:
            attention_weights = torch.softmax(torch.tensor(attention_weights), dim=0)
            final_result = torch.sum(results_tensor * attention_weights.unsqueeze(-1), dim=0)
        else:
            final_result = torch.mean(results_tensor, dim=0)
        
        return final_result
