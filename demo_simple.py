"""
Simplified demo script for HSGM framework without model downloads.
"""
import torch
import numpy as np
from typing import List, Dict
import logging

from config import HSGMConfig
from hsgm.components import LocalSemanticGraph, HierarchicalMemory, IncrementalUpdater, QueryProcessor
from hsgm.utils import compute_complexity_metrics, compute_approximation_error_bound
from visualization import HSGMVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_basic_components():
    """Demonstrate basic HSGM components."""
    print("=" * 60)
    print("HSGM DEMO: Basic Components")
    print("=" * 60)
    
    # Initialize configuration
    config = HSGMConfig()
    print(f"Configuration: segment_size={config.segment_size}, local_threshold={config.local_threshold}")
    
    # Create components
    local_graph = LocalSemanticGraph(hidden_dim=128, local_threshold=0.2)
    hierarchical_memory = HierarchicalMemory(hidden_dim=128, summary_dim=64, global_threshold=0.1)
    incremental_updater = IncrementalUpdater()
    query_processor = QueryProcessor(hidden_dim=128)
    
    print("✓ All HSGM components created successfully")
    
    return local_graph, hierarchical_memory, incremental_updater, query_processor

def demo_graph_construction():
    """Demonstrate graph construction."""
    print("\n" + "-" * 40)
    print("Graph Construction Demo")
    print("-" * 40)
    
    # Create local semantic graph
    local_graph = LocalSemanticGraph(hidden_dim=128, local_threshold=0.3)
    
    # Sample tokens and embeddings
    tokens = ["machine", "learning", "algorithms", "data", "analysis", "models"]
    embeddings = torch.randn(len(tokens), 128)
    
    print(f"Processing {len(tokens)} tokens...")
    
    # Build graph
    nodes, edges = local_graph.build_graph(tokens, embeddings, segment_id=0)
    
    print(f"✓ Local graph built:")
    print(f"  - Nodes: {len(nodes)}")
    print(f"  - Edges: {len(edges)}")
    
    # Show some edges
    for i, edge in enumerate(edges[:3]):
        source_token = next(node.token for node in nodes if node.id == edge.source)
        target_token = next(node.token for node in nodes if node.id == edge.target)
        print(f"  - Edge {i+1}: '{source_token}' -> '{target_token}' (weight: {edge.weight:.3f})")
    
    return nodes, edges

def demo_hierarchical_memory(nodes, edges):
    """Demonstrate hierarchical memory."""
    print("\n" + "-" * 40)
    print("Hierarchical Memory Demo")
    print("-" * 40)
    
    # Create hierarchical memory
    hierarchical_memory = HierarchicalMemory(hidden_dim=128, summary_dim=64, global_threshold=0.1)
    
    # Create summary node
    summary_node = hierarchical_memory.create_summary_node((nodes, edges))
    print(f"✓ Summary node created: shape={summary_node.shape}")
    
    # Update global graph
    hierarchical_memory.update_global_graph(summary_node)
    print(f"✓ Global graph updated: {len(hierarchical_memory.global_nodes)} nodes")
    
    # Test retrieval
    query = torch.randn(64)  # Match summary dimension
    top_k = hierarchical_memory.get_top_k_similar(query, k=1)
    print(f"✓ Top-K retrieval: found {len(top_k)} similar nodes")
    
    return hierarchical_memory

def demo_incremental_updates():
    """Demonstrate incremental updates."""
    print("\n" + "-" * 40)
    print("Incremental Updates Demo")
    print("-" * 40)
    
    # Create incremental updater
    updater = IncrementalUpdater()
    
    # Simulate streaming documents
    documents = [
        "First document about machine learning.",
        "Second document about deep learning algorithms.",
        "Third document about natural language processing."
    ]
    
    print(f"Processing {len(documents)} documents incrementally...")
    
    for i, doc in enumerate(documents):
        # Create dummy nodes and edges for each document
        tokens = doc.split()
        embeddings = torch.randn(len(tokens), 128)
        
        # Build local graph (simplified)
        nodes = []
        edges = []
        for j, (token, emb) in enumerate(zip(tokens, embeddings)):
            from hsgm.components import GraphNode
            node = GraphNode(
                id=i * 100 + j,
                embedding=emb,
                position=j,
                token=token,
                segment_id=i
            )
            nodes.append(node)
        
        # Update cache
        updater.update_cache(i, nodes, edges)
        
        # Get cache stats
        stats = updater.get_cache_stats()
        print(f"  Document {i+1}: Cache hit rate: {stats['hit_rate']:.3f}")
    
    return updater

def demo_query_processing():
    """Demonstrate query processing."""
    print("\n" + "-" * 40)
    print("Query Processing Demo")
    print("-" * 40)
    
    # Create query processor
    processor = QueryProcessor(hidden_dim=128)
    
    # Create sample local graph
    from hsgm.components import GraphNode, GraphEdge
    nodes = [
        GraphNode(id=0, embedding=torch.randn(128), position=0, token="machine", segment_id=0),
        GraphNode(id=1, embedding=torch.randn(128), position=1, token="learning", segment_id=0),
        GraphNode(id=2, embedding=torch.randn(128), position=2, token="algorithm", segment_id=0)
    ]
    
    edges = [
        GraphEdge(source=0, target=1, weight=0.8, similarity=0.8),
        GraphEdge(source=1, target=2, weight=0.7, similarity=0.7)
    ]
    
    # Process query
    query = torch.randn(128)
    result = processor.local_graph_reasoning(nodes, edges, query)
    
    print(f"✓ Query processed: result shape={result.shape}")
    
    return processor

def demo_theoretical_analysis():
    """Demonstrate theoretical analysis."""
    print("\n" + "-" * 40)
    print("Theoretical Analysis Demo")
    print("-" * 40)
    
    # Test different document lengths
    document_lengths = [1000, 5000, 10000, 20000]
    segment_size = 256
    
    print(f"Analyzing complexity for segment size: {segment_size}")
    print("\nDocument Length | HSGM Complexity | Full Graph Complexity | Speedup")
    print("-" * 75)
    
    for length in document_lengths:
        metrics = compute_complexity_metrics(length, segment_size)
        
        hsgm_complexity = metrics["hsgm_complexity"]
        full_graph_complexity = metrics["full_graph_complexity"]
        speedup = metrics["theoretical_speedup"]
        
        print(f"{length:>15} | {hsgm_complexity:>15.0f} | {full_graph_complexity:>20.0f} | {speedup:>6.1f}x")
    
    # Test approximation error bounds
    print(f"\nApproximation Error Analysis:")
    print("Local Threshold | Global Threshold | Error Bound")
    print("-" * 55)
    
    local_thresholds = [0.1, 0.2, 0.3]
    global_thresholds = [0.05, 0.1, 0.15]
    
    for local_thresh in local_thresholds:
        for global_thresh in global_thresholds:
            error_bound = compute_approximation_error_bound(local_thresh, global_thresh)
            print(f"{local_thresh:>14} | {global_thresh:>15} | {error_bound:>11.4f}")

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "-" * 40)
    print("Visualization Demo")
    print("-" * 40)
    
    visualizer = HSGMVisualizer()
    
    # Create sample data for visualization
    document_lengths = [1000, 5000, 10000, 20000]
    hsgm_times = [0.3, 1.2, 2.8, 5.2]
    baseline_times = [1.2, 8.5, 25.3, 85.7]
    
    hsgm_memory = [6.2, 8.5, 11.2, 15.8]
    baseline_memory = [12.5, 25.3, 45.7, 78.2]
    
    model_names = ['HSGM', 'Longformer', 'BigBird', 'Full Graph']
    accuracies = [0.785, 0.768, 0.771, 0.782]
    f1_scores = [0.856, 0.845, 0.848, 0.851]
    
    print("Visualization components created successfully!")
    print(f"Available colors: {list(visualizer.colors.keys())}")
    print("\nSample data prepared for plotting:")
    print(f"- Document lengths: {document_lengths}")
    print(f"- HSGM processing times: {hsgm_times}")
    print(f"- Baseline processing times: {baseline_times}")
    print(f"- Model names: {model_names}")
    print(f"- Accuracies: {accuracies}")
    
    # Note: In a full demo, we would generate actual plots here
    print("\nNote: To generate actual plots, run visualization functions with matplotlib enabled")

def main():
    """Run all simplified demos."""
    print("HSGM Framework Simplified Demonstration")
    print("This demo showcases the core components of HSGM without requiring model downloads.")
    
    try:
        # Basic components demo
        components = demo_basic_components()
        
        # Graph construction demo
        nodes, edges = demo_graph_construction()
        
        # Hierarchical memory demo
        hierarchical_memory = demo_hierarchical_memory(nodes, edges)
        
        # Incremental updates demo
        updater = demo_incremental_updates()
        
        # Query processing demo
        processor = demo_query_processing()
        
        # Theoretical analysis demo
        demo_theoretical_analysis()
        
        # Visualization demo
        demo_visualization()
        
        print("\n" + "=" * 60)
        print("SIMPLIFIED DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The HSGM framework core components have been demonstrated:")
        print("- Local semantic graph construction")
        print("- Hierarchical memory management")
        print("- Incremental update mechanism")
        print("- Query processing capabilities")
        print("- Theoretical complexity analysis")
        print("- Visualization components")
        print("\nFor full functionality with pre-trained models, run: python demo.py")
        print("For comprehensive experiments, run: python experiments.py")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
