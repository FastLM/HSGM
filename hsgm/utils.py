"""
Utility functions for HSGM framework.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
# Optional imports
try:
    import spacy
except ImportError:
    spacy = None

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = AutoModel = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import networkx as nx
try:
    from scipy.spatial.distance import cosine
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine = cosine_similarity = None
    TfidfVectorizer = None

class DocumentSegmenter:
    """
    Segments documents into semantically coherent chunks.
    """
    
    def __init__(self, 
                 segment_size: int = 256,
                 overlap_size: int = 50,
                 segment_method: str = "sliding_window"):
        """
        Initialize Document Segmenter.
        
        Args:
            segment_size: Target size of each segment in tokens
            overlap_size: Overlap between consecutive segments
            segment_method: Method for segmentation ("sliding_window", "sentence_boundary", "semantic")
        """
        self.segment_size = segment_size
        self.overlap_size = overlap_size
        self.segment_method = segment_method
        
        # Load spacy model for sentence boundary detection
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
    
    def sliding_window_segmentation(self, tokens: List[str]) -> List[List[str]]:
        """
        Segment using sliding window approach.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token segments
        """
        segments = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.segment_size, len(tokens))
            segment = tokens[start:end]
            segments.append(segment)
            
            if end == len(tokens):
                break
                
            start = end - self.overlap_size
        
        return segments
    
    def sentence_boundary_segmentation(self, text: str) -> List[List[str]]:
        """
        Segment using sentence boundaries.
        
        Args:
            text: Input text
            
        Returns:
            List of token segments
        """
        if self.nlp is None:
            return self.sliding_window_segmentation(text.split())
        
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        segments = []
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sent_tokens = sentence.split()
            
            if current_length + len(sent_tokens) <= self.segment_size:
                current_segment.extend(sent_tokens)
                current_length += len(sent_tokens)
            else:
                if current_segment:
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = sent_tokens
                current_length = len(sent_tokens)
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def semantic_segmentation(self, tokens: List[str], embeddings: torch.Tensor) -> List[List[str]]:
        """
        Segment based on semantic similarity.
        
        Args:
            tokens: List of tokens
            embeddings: Token embeddings
            
        Returns:
            List of token segments
        """
        if len(tokens) <= self.segment_size:
            return [tokens]
        
        # Compute similarities between consecutive tokens
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = torch.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[i + 1].unsqueeze(0)
            ).item()
            similarities.append(sim)
        
        # Find break points where similarity drops significantly
        break_points = [0]
        threshold = np.percentile(similarities, 20)  # Bottom 20% similarity
        
        for i, sim in enumerate(similarities):
            if sim < threshold and i - break_points[-1] >= self.segment_size // 2:
                break_points.append(i + 1)
        
        break_points.append(len(tokens))
        
        # Create segments
        segments = []
        for i in range(len(break_points) - 1):
            start = break_points[i]
            end = break_points[i + 1]
            segment = tokens[start:end]
            
            # If segment is too large, split it further
            if len(segment) > self.segment_size:
                sub_segments = self.sliding_window_segmentation(segment)
                segments.extend(sub_segments)
            else:
                segments.append(segment)
        
        return segments
    
    def segment_document(self, 
                        text: str, 
                        embeddings: Optional[torch.Tensor] = None) -> List[List[str]]:
        """
        Segment document into coherent chunks.
        
        Args:
            text: Input document text
            embeddings: Optional token embeddings for semantic segmentation
            
        Returns:
            List of token segments
        """
        tokens = text.split()
        
        if self.segment_method == "sliding_window":
            return self.sliding_window_segmentation(tokens)
        elif self.segment_method == "sentence_boundary":
            return self.sentence_boundary_segmentation(text)
        elif self.segment_method == "semantic" and embeddings is not None:
            return self.semantic_segmentation(tokens, embeddings)
        else:
            return self.sliding_window_segmentation(tokens)

class SimilarityComputer:
    """
    Computes various types of similarities for graph construction.
    """
    
    def __init__(self, similarity_type: str = "cosine"):
        """
        Initialize Similarity Computer.
        
        Args:
            similarity_type: Type of similarity ("cosine", "euclidean", "dot_product", "semantic")
        """
        self.similarity_type = similarity_type
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity."""
        return torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def euclidean_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute euclidean similarity (inverse of distance)."""
        distance = torch.norm(emb1 - emb2).item()
        return 1.0 / (1.0 + distance)
    
    def dot_product_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute dot product similarity."""
        return torch.dot(emb1, emb2).item()
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence transformers."""
        if SentenceTransformer is not None:
            # This would require loading a sentence transformer model
            # For now, return a placeholder
            return 0.5
        else:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
    
    def compute_similarity(self, 
                          emb1: torch.Tensor, 
                          emb2: torch.Tensor,
                          text1: Optional[str] = None,
                          text2: Optional[str] = None) -> float:
        """
        Compute similarity based on specified type.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            text1: Optional first text (for semantic similarity)
            text2: Optional second text (for semantic similarity)
            
        Returns:
            Similarity score
        """
        if self.similarity_type == "cosine":
            return self.cosine_similarity(emb1, emb2)
        elif self.similarity_type == "euclidean":
            return self.euclidean_similarity(emb1, emb2)
        elif self.similarity_type == "dot_product":
            return self.dot_product_similarity(emb1, emb2)
        elif self.similarity_type == "semantic" and text1 and text2:
            return self.semantic_similarity(text1, text2)
        else:
            return self.cosine_similarity(emb1, emb2)

class GraphAggregator:
    """
    Aggregates information from graph structures.
    """
    
    def __init__(self, aggregation_method: str = "mean"):
        """
        Initialize Graph Aggregator.
        
        Args:
            aggregation_method: Method for aggregation ("mean", "max", "sum", "attention")
        """
        self.aggregation_method = aggregation_method
    
    def mean_aggregation(self, embeddings: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean aggregation with optional weights."""
        if weights is not None:
            weighted_embeddings = embeddings * weights.unsqueeze(-1)
            return torch.sum(weighted_embeddings, dim=0) / torch.sum(weights)
        else:
            return torch.mean(embeddings, dim=0)
    
    def max_aggregation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Max aggregation."""
        return torch.max(embeddings, dim=0)[0]
    
    def sum_aggregation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Sum aggregation."""
        return torch.sum(embeddings, dim=0)
    
    def attention_aggregation(self, 
                            embeddings: torch.Tensor, 
                            query: torch.Tensor) -> torch.Tensor:
        """Attention-based aggregation."""
        # Compute attention scores
        attention_scores = torch.mm(embeddings, query.unsqueeze(-1)).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        # Weighted aggregation
        return torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=0)
    
    def aggregate(self, 
                 embeddings: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate embeddings using specified method.
        
        Args:
            embeddings: Node embeddings [num_nodes, hidden_dim]
            weights: Optional edge weights [num_nodes]
            query: Optional query for attention aggregation
            
        Returns:
            Aggregated embedding
        """
        if self.aggregation_method == "mean":
            return self.mean_aggregation(embeddings, weights)
        elif self.aggregation_method == "max":
            return self.max_aggregation(embeddings)
        elif self.aggregation_method == "sum":
            return self.sum_aggregation(embeddings)
        elif self.aggregation_method == "attention" and query is not None:
            return self.attention_aggregation(embeddings, query)
        else:
            return self.mean_aggregation(embeddings, weights)

class TextEncoder:
    """
    Encodes text using various pre-trained models.
    """
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 device: str = "cuda"):
        """
        Initialize Text Encoder.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        
        if AutoTokenizer is not None and AutoModel is not None:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
        else:
            # Fallback: create dummy tokenizer and model
            self.tokenizer = None
            self.model = None
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        if self.tokenizer is not None and self.model is not None:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
            return embedding
        else:
            # Fallback: return random embedding
            return torch.randn(768).to(self.device)
    
    def encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        """
        Encode list of tokens into embeddings.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Token embeddings [num_tokens, hidden_dim]
        """
        if self.tokenizer is not None and self.model is not None:
            text = " ".join(tokens)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Remove [CLS] and [SEP] tokens
                embeddings = outputs.last_hidden_state[0, 1:-1, :]
            
            return embeddings
        else:
            # Fallback: return random embeddings
            return torch.randn(len(tokens), 768).to(self.device)

class MemoryProfiler:
    """
    Profiles memory usage during HSGM operations.
    """
    
    def __init__(self):
        """Initialize Memory Profiler."""
        self.peak_memory = 0
        self.memory_history = []
        
    def start_profiling(self):
        """Start memory profiling."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def record_memory(self):
        """Record current memory usage."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_history.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            "peak_memory_gb": self.peak_memory,
            "current_memory_gb": self.memory_history[-1] if self.memory_history else 0,
            "avg_memory_gb": np.mean(self.memory_history) if self.memory_history else 0,
            "memory_growth_gb": self.memory_history[-1] - self.memory_history[0] if len(self.memory_history) > 1 else 0
        }

def compute_complexity_metrics(document_length: int, segment_size: int) -> Dict[str, float]:
    """
    Compute theoretical complexity metrics for HSGM.
    
    Args:
        document_length: Length of document in tokens
        segment_size: Size of each segment
        
    Returns:
        Dictionary of complexity metrics
    """
    num_segments = document_length / segment_size
    
    # HSGM complexity: O(N*k + (N/k)^2)
    hsgm_complexity = document_length * segment_size + (document_length / segment_size) ** 2
    
    # Full graph complexity: O(N^2)
    full_graph_complexity = document_length ** 2
    
    # Theoretical speedup
    speedup = full_graph_complexity / hsgm_complexity
    
    # Optimal segment size for minimal complexity
    optimal_segment_size = int(np.sqrt(document_length))
    
    return {
        "hsgm_complexity": hsgm_complexity,
        "full_graph_complexity": full_graph_complexity,
        "theoretical_speedup": speedup,
        "optimal_segment_size": optimal_segment_size,
        "num_segments": num_segments
    }

def compute_approximation_error_bound(local_threshold: float, 
                                    global_threshold: float) -> float:
    """
    Compute Frobenius-norm approximation error bound.
    
    Args:
        local_threshold: Local similarity threshold
        global_threshold: Global similarity threshold
        
    Returns:
        Error bound
    """
    # From paper: ||Afull - AHSGM||F <= f(γℓ, γg) * ||Afull||F
    # where f(γℓ, γg) = sqrt(2(1 - γℓ^2)) + sqrt(2(1 - γg^2))
    
    local_error = np.sqrt(2 * (1 - local_threshold**2))
    global_error = np.sqrt(2 * (1 - global_threshold**2))
    
    return local_error + global_error
