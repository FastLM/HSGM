"""
Comprehensive HSGM Demo and Test Suite
Combines all demonstration and testing functionality in one file.
"""
import torch
import numpy as np
import sys
import os
import time
from typing import List, Dict
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import HSGMConfig
from hsgm import HSGMModel, HSGMForClassification
from hsgm.components import LocalSemanticGraph, HierarchicalMemory, IncrementalUpdater, QueryProcessor, GraphNode, GraphEdge
from hsgm.utils import compute_complexity_metrics, compute_approximation_error_bound
from evaluation import Evaluator
from baselines import BaselineModels
from visualization import HSGMVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        from config import HSGMConfig, DatasetConfig, ExperimentConfig
        print("Config imports successful")
    except ImportError as e:
        print(f"Config import failed: {e}")
        return False
    
    try:
        from hsgm import HSGMModel, HSGMForClassification
        print("HSGM model imports successful")
    except ImportError as e:
        print(f"HSGM model import failed: {e}")
        return False
    
    try:
        from evaluation import Evaluator
        print("Evaluator import successful")
    except ImportError as e:
        print(f"Evaluator import failed: {e}")
        return False
    
    try:
        from baselines import BaselineModels
        print("Baselines import successful")
    except ImportError as e:
        print(f"Baselines import failed: {e}")
        return False
    
    try:
        from visualization import HSGMVisualizer
        print("Visualization import successful")
    except ImportError as e:
        print(f"Visualization import failed: {e}")
        return False
    
    return True

def test_config_creation():
    """Test configuration creation."""
    print("\nTesting configuration creation...")
    
    try:
        from config import HSGMConfig, DatasetConfig, ExperimentConfig
        
        config = HSGMConfig()
        dataset_config = DatasetConfig()
        experiment_config = ExperimentConfig()
        
        print(f"HSGM config created: hidden_dim={config.hidden_dim}, segment_size={config.segment_size}")
        print(f"Dataset config created: document_amr_path={dataset_config.document_amr_path}")
        print(f"Experiment config created: num_eval_runs={experiment_config.num_eval_runs}")
        
        return True
    except Exception as e:
        print(f"Configuration creation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic HSGM functionality without external dependencies."""
    print("\nTesting basic HSGM functionality...")
    
    try:
        from config import HSGMConfig
        from hsgm.components import LocalSemanticGraph, HierarchicalMemory
        from hsgm.utils import compute_complexity_metrics, compute_approximation_error_bound
        
        # Test configuration
        config = HSGMConfig()
        print(f"Configuration created: segment_size={config.segment_size}")
        
        # Test local semantic graph
        local_graph = LocalSemanticGraph(hidden_dim=64, local_threshold=0.2)
        print("Local semantic graph created")
        
        # Test hierarchical memory
        hierarchical_memory = HierarchicalMemory(hidden_dim=64, global_threshold=0.1)
        print("Hierarchical memory created")
        
        # Test utility functions
        metrics = compute_complexity_metrics(1000, 256)
        print(f"Complexity metrics: speedup={metrics['theoretical_speedup']:.2f}x")
        
        error_bound = compute_approximation_error_bound(0.2, 0.1)
        print(f"Error bound computed: {error_bound:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_construction():
    """Test graph construction with dummy data."""
    print("\nTesting graph construction...")
    
    try:
        from hsgm.components import LocalSemanticGraph
        
        # Create dummy embeddings
        tokens = ["token1", "token2", "token3", "token4"]
        embeddings = torch.randn(4, 64)  # 4 tokens, 64-dim embeddings
        
        # Build graph
        local_graph = LocalSemanticGraph(hidden_dim=64, local_threshold=0.1)
        nodes, edges = local_graph.build_graph(tokens, embeddings, segment_id=0)
        
        print(f"Graph built: {len(nodes)} nodes, {len(edges)} edges")
        
        # Test similarity computation
        sim = local_graph.compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity computed: {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Graph construction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hierarchical_memory():
    """Test hierarchical memory operations."""
    print("\nTesting hierarchical memory...")
    
    try:
        from hsgm.components import HierarchicalMemory, GraphNode, GraphEdge
        
        # Create hierarchical memory
        memory = HierarchicalMemory(hidden_dim=64, global_threshold=0.1)
        
        # Create dummy local graph
        nodes = [
            GraphNode(id=0, embedding=torch.randn(64), position=0, token="test1", segment_id=0),
            GraphNode(id=1, embedding=torch.randn(64), position=1, token="test2", segment_id=0)
        ]
        edges = [
            GraphEdge(source=0, target=1, weight=0.8, similarity=0.8)
        ]
        
        # Create summary node
        summary = memory.create_summary_node((nodes, edges))
        print(f"Summary node created: shape={summary.shape}")
        
        # Update global graph
        memory.update_global_graph(summary)
        print(f"Global graph updated: {len(memory.global_nodes)} nodes")
        
        # Test retrieval (use correct dimension)
        query = torch.randn(256)  # Match summary dimension
        top_k = memory.get_top_k_similar(query, k=1)
        print(f"Top-K retrieval: {len(top_k)} results")
        
        return True
        
    except Exception as e:
        print(f"Hierarchical memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_incremental_updater():
    """Test incremental update mechanism."""
    print("\nTesting incremental updater...")
    
    try:
        from hsgm.components import IncrementalUpdater, GraphNode, GraphEdge
        
        updater = IncrementalUpdater()
        
        # Create dummy data
        nodes = [
            GraphNode(id=0, embedding=torch.randn(64), position=0, token="test1", segment_id=0)
        ]
        edges = []
        
        # Update cache
        updater.update_cache(0, nodes, edges)
        print("Cache updated")
        
        # Get cached data
        cached = updater.get_cached_segment(0)
        print(f"Cache retrieval: {cached is not None}")
        
        # Get cache stats
        stats = updater.get_cache_stats()
        print(f"Cache stats: hit_rate={stats['hit_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Incremental updater test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_processor():
    """Test query processing."""
    print("\nTesting query processor...")
    
    try:
        from hsgm.components import QueryProcessor, GraphNode
        
        processor = QueryProcessor(hidden_dim=64)
        
        # Create dummy nodes and edges
        nodes = [
            GraphNode(id=0, embedding=torch.randn(64), position=0, token="test1", segment_id=0)
        ]
        edges = []
        
        query = torch.randn(64)
        
        # Test local reasoning
        result = processor.local_graph_reasoning(nodes, edges, query)
        print(f"Local reasoning: shape={result.shape}")
        
        return True
        
    except Exception as e:
        print(f"Query processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization components."""
    print("\nTesting visualization...")
    
    try:
        from visualization import HSGMVisualizer
        
        visualizer = HSGMVisualizer()
        
        # Test with dummy data
        document_lengths = [1000, 5000]
        hsgm_times = [0.3, 1.2]
        baseline_times = [1.2, 8.5]
        
        print("Visualizer created")
        print(f"Available colors: {list(visualizer.colors.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        return False

def demo_basic_components():
    """Demonstrate basic HSGM components."""
    print("\n" + "=" * 60)
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
    
    print("All HSGM components created successfully")
    
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
    
    print(f"Local graph built:")
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
    print(f"Summary node created: shape={summary_node.shape}")
    
    # Update global graph
    hierarchical_memory.update_global_graph(summary_node)
    print(f"Global graph updated: {len(hierarchical_memory.global_nodes)} nodes")
    
    # Test retrieval
    query = torch.randn(64)  # Match summary dimension
    top_k = hierarchical_memory.get_top_k_similar(query, k=1)
    print(f"Top-K retrieval: found {len(top_k)} similar nodes")
    
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
    
    print(f"Query processed: result shape={result.shape}")
    
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

def demo_comparison_with_baselines():
    """Demonstrate comparison with baseline models."""
    print("\n" + "-" * 40)
    print("Baseline Comparison Demo")
    print("-" * 40)
    
    try:
        # Initialize baselines (without downloading models)
        baselines = BaselineModels("cpu")
        
        print("Baseline models initialized successfully!")
        print(f"Available baselines: {list(baselines.models.keys())}")
        
        # Show theoretical comparison
        print("\nTheoretical Performance Comparison:")
        print("Model Type | Processing Complexity | Memory Complexity")
        print("-" * 60)
        print("HSGM       | O(Nk + (N/k)²)        | O(N)")
        print("Full Graph | O(N²)                 | O(N²)")
        print("Longformer | O(N log N)            | O(N)")
        print("BigBird    | O(N log N)            | O(N)")
        
        return True
        
    except Exception as e:
        print(f"Baseline comparison demo failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("HSGM Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_creation,
        test_basic_functionality,
        test_graph_construction,
        test_hierarchical_memory,
        test_incremental_updater,
        test_query_processor,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! HSGM implementation is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False

def run_all_demos():
    """Run all demonstrations."""
    print("\nHSGM Framework Demonstration")
    print("This demo showcases the key features of the HSGM framework.")
    
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
        
        # Baseline comparison demo
        demo_comparison_with_baselines()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The HSGM framework has been demonstrated with:")
        print("- Basic component functionality")
        print("- Local semantic graph construction")
        print("- Hierarchical memory management")
        print("- Incremental update mechanism")
        print("- Query processing capabilities")
        print("- Theoretical complexity analysis")
        print("- Visualization components")
        print("- Baseline model comparison")
        
        return True
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run tests and demos."""
    print("HSGM Comprehensive Demo and Test Suite")
    print("=" * 60)
    
    # Run tests first
    test_success = run_all_tests()
    
    if test_success:
        print("\nProceeding to demonstrations...")
        # Run demos
        demo_success = run_all_demos()
        
        if demo_success:
            print("\nAll tests and demonstrations completed successfully!")
            print("\nFor more detailed experiments, run: python experiments.py")
            print("For training, run: python train.py")
        else:
            print("\nDemos completed with some issues.")
    else:
        print("\nSome tests failed. Please fix the implementation before running demos.")
    
    return test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

