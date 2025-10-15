"""
Python bindings for HSGM kernels using PyTorch's C++ extension
"""
import torch
import os
from torch.utils.cpp_extension import load

# Get the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_kernel_path = os.path.join(current_dir, 'hsgm_kernels.cu')

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        # Load HSGM kernels
        cuda_ops = load(
            name='hsgm_ops',
            sources=[cuda_kernel_path],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-std=c++14',
                '--expt-relaxed-constexpr',
                '-gencode=arch=compute_70,code=sm_70',  # V100
                '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
                '-gencode=arch=compute_80,code=sm_80',  # A100
                '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
            ],
            verbose=False
        )
        CUDA_OPS_LOADED = True
    except Exception as e:
        print(f"Warning: Failed to load HSGM kernels: {e}")
        print("Falling back to PyTorch operations")
        CUDA_OPS_LOADED = False
else:
    CUDA_OPS_LOADED = False
    print("CUDA not available, using CPU operations")


class HSGMOps:
    """
    High-level interface for HSGM operations
    Falls back to PyTorch operations if CUDA is not available
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.use_cuda = CUDA_AVAILABLE and CUDA_OPS_LOADED and device.startswith('cuda')
    
    def pairwise_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity within each segment
        
        Args:
            embeddings: [M, k, D] tensor of token embeddings
            
        Returns:
            similarities: [M, k, k] tensor of pairwise similarities
        """
        M, k, D = embeddings.shape
        
        if self.use_cuda:
            # Use custom HSGM kernel
            similarities = torch.empty(M, k, k, device=embeddings.device)
            cuda_ops.pairwise_similarity(embeddings, similarities)
            return similarities
        else:
            # Fallback to PyTorch
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            similarities = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
            return similarities
    
    def adaptive_thresholding(self, similarities: torch.Tensor,
                            alpha: float = 1.0,
                            beta: float = 0.5) -> torch.Tensor:
        """
        Compute adaptive thresholds for each segment
        
        Args:
            similarities: [M, k, k] tensor of similarities
            alpha: Weight for mean
            beta: Weight for standard deviation
            
        Returns:
            thresholds: [M] tensor of threshold values
        """
        M, k, _ = similarities.shape
        
        if self.use_cuda:
            means = torch.empty(M, device=similarities.device)
            stds = torch.empty(M, device=similarities.device)
            cuda_ops.adaptive_thresholding(similarities, means, stds)
            thresholds = alpha * means + beta * stds
            return thresholds
        else:
            # Fallback to PyTorch
            flat_sims = similarities.view(M, -1)
            means = flat_sims.mean(dim=1)
            stds = flat_sims.std(dim=1)
            thresholds = alpha * means + beta * stds
            return thresholds
    
    def create_sparse_edges(self, similarities: torch.Tensor,
                          thresholds: torch.Tensor,
                          max_edges: int = 1024):
        """
        Create sparse edge lists based on thresholds
        
        Args:
            similarities: [M, k, k] tensor of similarities
            thresholds: [M] tensor of threshold values
            max_edges: Maximum number of edges per segment
            
        Returns:
            edge_indices: [M, max_edges, 2] tensor of edge indices (COO format)
            edge_weights: [M, max_edges] tensor of edge weights
            edge_counts: [M] tensor of actual edge counts
        """
        M, k, _ = similarities.shape
        
        if self.use_cuda:
            edge_indices = torch.zeros(M, max_edges, 2, dtype=torch.int32, device=similarities.device)
            edge_weights = torch.zeros(M, max_edges, device=similarities.device)
            edge_counts = torch.zeros(M, dtype=torch.int32, device=similarities.device)
            
            cuda_ops.create_sparse_edges(
                similarities, thresholds,
                edge_indices, edge_weights, edge_counts,
                max_edges
            )
            
            return edge_indices, edge_weights, edge_counts
        else:
            # Fallback to PyTorch
            edge_list = []
            weight_list = []
            count_list = []
            
            for i in range(M):
                # Get upper triangular part
                mask = similarities[i] >= thresholds[i]
                triu_mask = torch.triu(mask, diagonal=1)
                
                edges = triu_mask.nonzero()
                weights = similarities[i][triu_mask]
                
                # Pad or truncate to max_edges
                num_edges = min(edges.size(0), max_edges)
                
                padded_edges = torch.zeros(max_edges, 2, dtype=torch.int32, device=similarities.device)
                padded_weights = torch.zeros(max_edges, device=similarities.device)
                
                if num_edges > 0:
                    padded_edges[:num_edges] = edges[:num_edges]
                    padded_weights[:num_edges] = weights[:num_edges]
                
                edge_list.append(padded_edges)
                weight_list.append(padded_weights)
                count_list.append(num_edges)
            
            edge_indices = torch.stack(edge_list)
            edge_weights = torch.stack(weight_list)
            edge_counts = torch.tensor(count_list, dtype=torch.int32, device=similarities.device)
            
            return edge_indices, edge_weights, edge_counts
    
    def cross_segment_attention(self, query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               num_heads: int = 12) -> torch.Tensor:
        """
        Compute multi-head cross-segment attention
        
        Args:
            query: [M, seq_len, D] tensor of query vectors
            key: [M, num_summaries, D] tensor of key vectors
            value: [M, num_summaries, D] tensor of value vectors
            num_heads: Number of attention heads
            
        Returns:
            output: [M, seq_len, D] tensor of attention output
        """
        M, seq_len, D = query.shape
        _, num_summaries, _ = key.shape
        
        if self.use_cuda and num_summaries <= 128:  # HSGM kernel limit
            output = torch.empty_like(query)
            cuda_ops.cross_attention(query, key, value, output, num_heads)
            return output
        else:
            # Fallback to PyTorch multi-head attention
            head_dim = D // num_heads
            
            # Reshape for multi-head attention
            q = query.view(M, seq_len, num_heads, head_dim).transpose(1, 2)
            k = key.view(M, num_summaries, num_heads, head_dim).transpose(1, 2)
            v = value.view(M, num_summaries, num_heads, head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attended = torch.matmul(attn_weights, v)
            
            # Reshape back
            output = attended.transpose(1, 2).contiguous().view(M, seq_len, D)
            return output
    
    def aggregate_summary_nodes(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node embeddings into summary nodes using mean and max pooling
        
        Args:
            node_embeddings: [M, k, D] tensor of node embeddings
            
        Returns:
            summary_nodes: [M, 2*D] tensor of summary node embeddings
        """
        M, k, D = node_embeddings.shape
        
        if self.use_cuda:
            summary_nodes = torch.empty(M, 2 * D, device=node_embeddings.device)
            cuda_ops.summary_aggregation(node_embeddings, summary_nodes)
            return summary_nodes
        else:
            # Fallback to PyTorch
            mean_pool = node_embeddings.mean(dim=1)  # [M, D]
            max_pool = node_embeddings.max(dim=1)[0]  # [M, D]
            summary_nodes = torch.cat([mean_pool, max_pool], dim=-1)  # [M, 2*D]
            return summary_nodes
    
    def top_k_retrieval(self, query: torch.Tensor,
                       summary_nodes: torch.Tensor,
                       k: int = 5) -> tuple:
        """
        Find top-K most similar summary nodes
        
        Args:
            query: [D_summary] tensor of query embedding
            summary_nodes: [M, D_summary] tensor of summary embeddings
            k: Number of top results to return
            
        Returns:
            indices: [K] tensor of top-K indices
            scores: [K] tensor of top-K similarity scores
        """
        M, D_summary = summary_nodes.shape
        
        if self.use_cuda and M <= 128 and k <= 32:
            indices = torch.empty(k, dtype=torch.int32, device=query.device)
            scores = torch.empty(k, device=query.device)
            cuda_ops.top_k_retrieval(query, summary_nodes, indices, scores, k)
            return indices.long(), scores
        else:
            # Fallback to PyTorch
            # Compute cosine similarity
            query_norm = torch.nn.functional.normalize(query, p=2, dim=0)
            nodes_norm = torch.nn.functional.normalize(summary_nodes, p=2, dim=1)
            similarities = torch.mv(nodes_norm, query_norm)
            
            # Get top-K
            scores, indices = torch.topk(similarities, k=min(k, M))
            return indices, scores
    
    def benchmark_kernels(self, M=10, k=256, D=768, num_runs=100):
        """
        Benchmark HSGM kernels vs PyTorch operations
        
        Args:
            M: Number of segments
            k: Segment size
            D: Embedding dimension
            num_runs: Number of benchmark runs
            
        Returns:
            results: Dictionary of timing results
        """
        if not self.use_cuda:
            print("CUDA not available for benchmarking")
            return {}
        
        # Create dummy data
        embeddings = torch.randn(M, k, D, device=self.device)
        
        results = {}
        
        # Benchmark pairwise similarity
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_runs):
            _ = self.pairwise_similarity(embeddings)
        end.record()
        torch.cuda.synchronize()
        
        results['pairwise_similarity_ms'] = start.elapsed_time(end) / num_runs
        
        # Benchmark other operations similarly...
        print(f"Pairwise Similarity: {results['pairwise_similarity_ms']:.3f}ms per call")
        
        return results


# Singleton instance
_hsgm_ops_instance = None

def get_hsgm_ops(device='cuda'):
    """Get or create HSGM ops singleton instance"""
    global _hsgm_ops_instance
    if _hsgm_ops_instance is None:
        _hsgm_ops_instance = HSGMOps(device)
    return _hsgm_ops_instance
