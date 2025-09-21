"""
Simple test script for HSGM implementation without downloading models.
"""
import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic HSGM functionality without external dependencies."""
    print("Testing basic HSGM functionality...")
    
    try:
        from config import HSGMConfig
        from hsgm.components import LocalSemanticGraph, HierarchicalMemory
        from hsgm.utils import compute_complexity_metrics, compute_approximation_error_bound
        
        # Test configuration
        config = HSGMConfig()
        print(f"✓ Configuration created: segment_size={config.segment_size}")
        
        # Test local semantic graph
        local_graph = LocalSemanticGraph(hidden_dim=64, local_threshold=0.2)
        print("✓ Local semantic graph created")
        
        # Test hierarchical memory
        hierarchical_memory = HierarchicalMemory(hidden_dim=64, global_threshold=0.1)
        print("✓ Hierarchical memory created")
        
        # Test utility functions
        metrics = compute_complexity_metrics(1000, 256)
        print(f"✓ Complexity metrics: speedup={metrics['theoretical_speedup']:.2f}x")
        
        error_bound = compute_approximation_error_bound(0.2, 0.1)
        print(f"✓ Error bound computed: {error_bound:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
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
        
        print(f"✓ Graph built: {len(nodes)} nodes, {len(edges)} edges")
        
        # Test similarity computation
        sim = local_graph.compute_similarity(embeddings[0], embeddings[1])
        print(f"✓ Similarity computed: {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Graph construction test failed: {e}")
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
        print(f"✓ Summary node created: shape={summary.shape}")
        
        # Update global graph
        memory.update_global_graph(summary)
        print(f"✓ Global graph updated: {len(memory.global_nodes)} nodes")
        
        # Test retrieval (use correct dimension)
        query = torch.randn(256)  # Match summary dimension
        top_k = memory.get_top_k_similar(query, k=1)
        print(f"✓ Top-K retrieval: {len(top_k)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Hierarchical memory test failed: {e}")
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
        print("✓ Cache updated")
        
        # Get cached data
        cached = updater.get_cached_segment(0)
        print(f"✓ Cache retrieval: {cached is not None}")
        
        # Get cache stats
        stats = updater.get_cache_stats()
        print(f"✓ Cache stats: hit_rate={stats['hit_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Incremental updater test failed: {e}")
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
        print(f"✓ Local reasoning: shape={result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Query processor test failed: {e}")
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
        
        print("✓ Visualizer created")
        print(f"✓ Available colors: {list(visualizer.colors.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def run_simple_tests():
    """Run all simple tests."""
    print("HSGM Simple Tests (No Model Downloads)")
    print("=" * 50)
    
    tests = [
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
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Simple Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple tests passed! Core HSGM functionality is working.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
