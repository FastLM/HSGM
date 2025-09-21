"""
Core HSGM model implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
import time
import numpy as np
from dataclasses import dataclass

from .components import (
    LocalSemanticGraph, 
    HierarchicalMemory, 
    IncrementalUpdater, 
    QueryProcessor
)
from .utils import (
    DocumentSegmenter, 
    SimilarityComputer, 
    GraphAggregator,
    TextEncoder,
    MemoryProfiler,
    compute_complexity_metrics,
    compute_approximation_error_bound
)

@dataclass
class HSGMOutput:
    """Output from HSGM model."""
    embeddings: torch.Tensor
    summary_nodes: List[torch.Tensor]
    local_graphs: Dict[int, Tuple[List, List]]  # segment_id -> (nodes, edges)
    global_graph: Tuple[List[torch.Tensor], List[Tuple]]  # (nodes, edges)
    processing_time: float
    memory_usage: Dict[str, float]
    complexity_metrics: Dict[str, float]

class HSGMModel(nn.Module):
    """
    Main HSGM model that orchestrates all components.
    """
    
    def __init__(self, 
                 config,
                 hidden_dim: int = 768,
                 segment_size: int = 256,
                 local_threshold: float = 0.2,
                 global_threshold: float = 0.1,
                 top_k_retrieval: int = 5,
                 device: str = "cuda"):
        """
        Initialize HSGM Model.
        
        Args:
            config: Configuration object
            hidden_dim: Hidden dimension for embeddings
            segment_size: Size of document segments
            local_threshold: Threshold for local graph edges
            global_threshold: Threshold for global graph edges
            top_k_retrieval: Number of top segments to retrieve
            device: Device to run on
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.top_k_retrieval = top_k_retrieval
        self.device = device
        
        # Initialize components
        self.text_encoder = TextEncoder(
            model_name=config.model_name,
            device=device
        )
        
        self.document_segmenter = DocumentSegmenter(
            segment_size=segment_size,
            segment_method="sliding_window"
        )
        
        self.similarity_computer = SimilarityComputer(
            similarity_type="cosine"
        )
        
        self.local_graph_builder = LocalSemanticGraph(
            hidden_dim=hidden_dim,
            local_threshold=local_threshold
        )
        
        self.hierarchical_memory = HierarchicalMemory(
            hidden_dim=hidden_dim,
            global_threshold=global_threshold
        )
        
        self.incremental_updater = IncrementalUpdater()
        
        self.query_processor = QueryProcessor(
            hidden_dim=hidden_dim
        )
        
        self.graph_aggregator = GraphAggregator(
            aggregation_method="attention"
        )
        
        self.memory_profiler = MemoryProfiler()
        
        # Model parameters
        self.num_segments_processed = 0
        self.total_processing_time = 0.0
        
    def forward(self, 
                documents: Union[str, List[str]],
                return_graphs: bool = True) -> HSGMOutput:
        """
        Forward pass through HSGM.
        
        Args:
            documents: Input document(s)
            return_graphs: Whether to return graph structures
            
        Returns:
            HSGMOutput with results
        """
        start_time = time.time()
        self.memory_profiler.start_profiling()
        
        if isinstance(documents, str):
            documents = [documents]
        
        all_embeddings = []
        all_summary_nodes = []
        all_local_graphs = {}
        global_graph_nodes = []
        global_graph_edges = []
        
        for doc_idx, document in enumerate(documents):
            # Step 1: Document segmentation
            segments = self.document_segmenter.segment_document(document)
            
            # Step 2: Process each segment
            for seg_idx, segment in enumerate(segments):
                segment_id = doc_idx * 1000 + seg_idx
                
                # Encode segment tokens
                embeddings = self.text_encoder.encode_tokens(segment)
                
                # Build local semantic graph
                nodes, edges = self.local_graph_builder.build_graph(
                    tokens=segment,
                    embeddings=embeddings,
                    segment_id=segment_id
                )
                
                # Create summary node
                summary_node = self.hierarchical_memory.create_summary_node(
                    local_graph=(nodes, edges),
                    previous_summaries=global_graph_nodes
                )
                
                # Update hierarchical memory
                self.hierarchical_memory.update_global_graph(summary_node)
                
                # Update incremental cache
                self.incremental_updater.update_cache(segment_id, nodes, edges)
                
                # Store results
                all_embeddings.append(embeddings)
                all_summary_nodes.append(summary_node)
                if return_graphs:
                    all_local_graphs[segment_id] = (nodes, edges)
                
                self.num_segments_processed += 1
            
            # Collect global graph from hierarchical memory
            global_graph_nodes.extend(self.hierarchical_memory.global_nodes)
            global_graph_edges.extend(self.hierarchical_memory.global_edges)
        
        # Compute final embeddings
        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            final_embeddings = torch.zeros(1, self.hidden_dim)
        
        # Record memory usage
        self.memory_profiler.record_memory()
        
        # Compute processing time
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        # Compute complexity metrics
        total_tokens = sum(len(doc.split()) for doc in documents)
        complexity_metrics = compute_complexity_metrics(total_tokens, self.segment_size)
        
        return HSGMOutput(
            embeddings=final_embeddings,
            summary_nodes=all_summary_nodes,
            local_graphs=all_local_graphs if return_graphs else {},
            global_graph=(global_graph_nodes, global_graph_edges),
            processing_time=processing_time,
            memory_usage=self.memory_profiler.get_stats(),
            complexity_metrics=complexity_metrics
        )
    
    def query(self, 
              query_text: str,
              top_k: Optional[int] = None) -> torch.Tensor:
        """
        Process a query using hierarchical retrieval.
        
        Args:
            query_text: Query text
            top_k: Number of top segments to retrieve (uses default if None)
            
        Returns:
            Query result embedding
        """
        if top_k is None:
            top_k = self.top_k_retrieval
        
        # Encode query
        query_embedding = self.text_encoder.encode_text(query_text)
        
        # Get cached local graphs
        local_graphs = {}
        for segment_id in range(self.num_segments_processed):
            cached_data = self.incremental_updater.get_cached_segment(segment_id)
            if cached_data is not None:
                local_graphs[segment_id] = cached_data
        
        # Process query hierarchically
        result = self.query_processor.hierarchical_query_processing(
            query=query_embedding,
            hierarchical_memory=self.hierarchical_memory,
            local_graphs=local_graphs,
            top_k=top_k
        )
        
        return result
    
    def incremental_update(self, new_segments: List[str]) -> Dict[str, Any]:
        """
        Perform incremental update with new segments.
        
        Args:
            new_segments: List of new document segments
            
        Returns:
            Update statistics
        """
        start_time = time.time()
        self.memory_profiler.start_profiling()
        
        update_stats = {
            "segments_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time": 0.0,
            "memory_usage": {}
        }
        
        for segment in new_segments:
            segment_id = self.num_segments_processed
            
            # Check cache first
            cached_data = self.incremental_updater.get_cached_segment(segment_id)
            if cached_data is not None:
                update_stats["cache_hits"] += 1
                continue
            
            update_stats["cache_misses"] += 1
            
            # Process new segment
            tokens = segment.split()
            embeddings = self.text_encoder.encode_tokens(tokens)
            
            # Build local graph
            nodes, edges = self.local_graph_builder.build_graph(
                tokens=tokens,
                embeddings=embeddings,
                segment_id=segment_id
            )
            
            # Create summary node
            summary_node = self.hierarchical_memory.create_summary_node(
                local_graph=(nodes, edges),
                previous_summaries=self.hierarchical_memory.global_nodes
            )
            
            # Update hierarchical memory
            self.hierarchical_memory.update_global_graph(summary_node)
            
            # Update cache
            self.incremental_updater.update_cache(segment_id, nodes, edges)
            
            self.num_segments_processed += 1
            update_stats["segments_processed"] += 1
        
        # Record final statistics
        update_stats["processing_time"] = time.time() - start_time
        update_stats["memory_usage"] = self.memory_profiler.get_stats()
        
        # Add cache statistics
        cache_stats = self.incremental_updater.get_cache_stats()
        update_stats.update(cache_stats)
        
        return update_stats
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        return {
            "num_segments_processed": self.num_segments_processed,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time_per_segment": (
                self.total_processing_time / max(1, self.num_segments_processed)
            ),
            "memory_stats": self.memory_profiler.get_stats(),
            "cache_stats": self.incremental_updater.get_cache_stats(),
            "hierarchical_memory_size": len(self.hierarchical_memory.global_nodes),
            "approximation_error_bound": compute_approximation_error_bound(
                self.local_threshold, self.global_threshold
            )
        }
    
    def reset_model(self):
        """Reset model state."""
        self.num_segments_processed = 0
        self.total_processing_time = 0.0
        self.hierarchical_memory.global_nodes = []
        self.hierarchical_memory.global_edges = []
        self.incremental_updater.edge_cache = {}
        self.incremental_updater.node_cache = {}
        self.incremental_updater.access_times = {}
        self.incremental_updater.cache_hits = 0
        self.incremental_updater.cache_misses = 0
        self.memory_profiler.peak_memory = 0
        self.memory_profiler.memory_history = []

class HSGMForClassification(nn.Module):
    """
    HSGM model adapted for classification tasks.
    """
    
    def __init__(self, 
                 base_model: HSGMModel,
                 num_classes: int,
                 dropout: float = 0.1):
        """
        Initialize HSGM for classification.
        
        Args:
            base_model: Base HSGM model
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_model.hidden_dim // 2, num_classes)
        )
    
    def forward(self, documents: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            documents: Input document(s)
            
        Returns:
            Classification logits
        """
        # Get HSGM output
        hsgm_output = self.base_model(documents, return_graphs=False)
        
        # Aggregate embeddings
        if hsgm_output.embeddings.size(0) > 0:
            # Use mean pooling over all embeddings
            aggregated = torch.mean(hsgm_output.embeddings, dim=0)
        else:
            aggregated = torch.zeros(self.base_model.hidden_dim)
        
        # Classification
        logits = self.classifier(aggregated)
        
        return logits

class HSGMForGeneration(nn.Module):
    """
    HSGM model adapted for text generation tasks.
    """
    
    def __init__(self, 
                 base_model: HSGMModel,
                 vocab_size: int,
                 hidden_dim: int = 768,
                 num_layers: int = 6):
        """
        Initialize HSGM for generation.
        
        Args:
            base_model: Base HSGM model
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of decoder layers
        """
        super().__init__()
        
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, 
                documents: Union[str, List[str]],
                target_sequences: Optional[List[str]] = None) -> torch.Tensor:
        """
        Forward pass for generation.
        
        Args:
            documents: Input document(s)
            target_sequences: Target sequences for training
            
        Returns:
            Generation logits
        """
        # Get HSGM output
        hsgm_output = self.base_model(documents, return_graphs=False)
        
        # Use summary nodes as memory
        if hsgm_output.summary_nodes:
            memory = torch.stack(hsgm_output.summary_nodes).unsqueeze(0)
        else:
            memory = torch.zeros(1, 1, self.hidden_dim)
        
        # For now, return a placeholder
        # In a full implementation, this would generate sequences
        batch_size = 1
        seq_len = 10  # Placeholder
        
        # Create target embeddings (placeholder)
        target_embeddings = torch.randn(batch_size, seq_len, self.hidden_dim)
        
        # Decoder forward pass
        decoder_output = target_embeddings
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, memory)
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return logits
