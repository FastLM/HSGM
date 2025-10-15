"""
Demo script for HSGM framework.
"""
import torch
import numpy as np
from typing import List, Dict
import logging

from config import HSGMConfig
from hsgm import HSGMModel, HSGMForClassification
from experiments.evaluation import Evaluator
from experiments.baselines import BaselineModels
from experiments.visualization import HSGMVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_basic_functionality():
    """Demonstrate basic HSGM functionality."""
    print("=" * 60)
    print("HSGM DEMO: Basic Functionality")
    print("=" * 60)
    
    # Initialize configuration
    config = HSGMConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {config.device}")
    
    # Create HSGM model
    hsgm_model = HSGMModel(
        config=config,
        hidden_dim=config.hidden_dim,
        segment_size=config.segment_size,
        local_threshold=config.local_threshold,
        global_threshold=config.global_threshold,
        top_k_retrieval=config.top_k_retrieval,
        device=config.device
    )
    
    # Sample documents
    sample_documents = [
        "This is a sample document for HSGM demonstration. It contains multiple sentences to show how the hierarchical segment-graph memory works.",
        "Another document with different content to test the model's ability to handle multiple documents and maintain separate semantic graphs.",
        "A longer document that will be segmented into multiple parts to demonstrate the local graph construction and hierarchical memory aggregation."
    ]
    
    print(f"\nProcessing {len(sample_documents)} sample documents...")
    
    # Process documents
    output = hsgm_model(sample_documents)
    
    print(f"Processed {len(sample_documents)} documents")
    print(f"Generated {len(output.summary_nodes)} summary nodes")
    print(f"Processing time: {output.processing_time:.4f} seconds")
    print(f"Memory usage: {output.memory_usage['current_memory_gb']:.4f} GB")
    
    # Demonstrate query processing
    print("\n" + "-" * 40)
    print("Query Processing Demo")
    print("-" * 40)
    
    query = "What is the content about?"
    query_result = hsgm_model.query(query)
    
    print(f"Query: '{query}'")
    print(f"Query result embedding shape: {query_result.shape}")
    
    return hsgm_model, output

def demo_comparison_with_baselines():
    """Demonstrate comparison with baseline models."""
    print("\n" + "=" * 60)
    print("HSGM DEMO: Baseline Comparison")
    print("=" * 60)
    
    config = HSGMConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    hsgm_model = HSGMModel(
        config=config,
        device=config.device
    )
    
    baselines = BaselineModels(config.device)
    
    # Sample documents
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
        "Natural language processing combines computational linguistics with machine learning to understand human language."
    ]
    
    print(f"Comparing HSGM with baseline models on {len(sample_documents)} documents...")
    
    # Evaluate HSGM
    import time
    start_time = time.time()
    hsgm_output = hsgm_model(sample_documents)
    hsgm_time = time.time() - start_time
    
    # Evaluate baselines
    baseline_results = {}
    baseline_names = ["longformer", "bigbird", "sliding_window_graph"]
    
    for baseline_name in baseline_names:
        try:
            baseline_model = baselines.get_model(baseline_name)
            start_time = time.time()
            baseline_output = baseline_model(sample_documents)
            baseline_time = time.time() - start_time
            
            baseline_results[baseline_name] = {
                "processing_time": baseline_time,
                "output_shape": baseline_output.shape
            }
            
            print(f"{baseline_name}: {baseline_time:.4f}s")
            
        except Exception as e:
            print(f"{baseline_name}: Error - {e}")
    
    print(f"HSGM: {hsgm_time:.4f}s")
    
    # Calculate speedup
    if baseline_results:
        avg_baseline_time = np.mean([r["processing_time"] for r in baseline_results.values()])
        speedup = avg_baseline_time / hsgm_time
        print(f"Average speedup: {speedup:.2f}x")
    
    return baseline_results

def demo_streaming_processing():
    """Demonstrate streaming document processing."""
    print("\n" + "=" * 60)
    print("HSGM DEMO: Streaming Processing")
    print("=" * 60)
    
    config = HSGMConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create HSGM model
    hsgm_model = HSGMModel(
        config=config,
        device=config.device
    )
    
    # Simulate streaming documents
    streaming_documents = [
        "First document arrives in the stream.",
        "Second document with new information comes next.",
        "Third document continues the stream of data.",
        "Fourth document adds more context to the processing.",
        "Fifth document completes this batch of streaming data."
    ]
    
    print(f"Processing {len(streaming_documents)} documents in streaming fashion...")
    
    # Process documents incrementally
    for i, doc in enumerate(streaming_documents):
        print(f"\nProcessing document {i+1}: {doc[:50]}...")
        
        # Incremental update
        update_stats = hsgm_model.incremental_update([doc])
        
        print(f"  Cache hit rate: {update_stats['hit_rate']:.3f}")
        print(f"  Memory usage: {update_stats['memory_usage']['current_memory_gb']:.4f} GB")
        print(f"  Processing time: {update_stats['processing_time']:.4f}s")
    
    # Get final model statistics
    model_stats = hsgm_model.get_model_stats()
    
    print(f"\nFinal Model Statistics:")
    print(f"  Segments processed: {model_stats['num_segments_processed']}")
    print(f"  Total processing time: {model_stats['total_processing_time']:.4f}s")
    print(f"  Average time per segment: {model_stats['avg_processing_time_per_segment']:.4f}s")
    print(f"  Hierarchical memory size: {model_stats['hierarchical_memory_size']}")
    
    return model_stats

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("HSGM DEMO: Visualization")
    print("=" * 60)
    
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
    
    print("Generating visualizations...")
    
    # Generate plots
    print("1. Complexity analysis plot")
    visualizer.plot_complexity_analysis(document_lengths, hsgm_times, baseline_times)
    
    print("2. Memory usage comparison plot")
    visualizer.plot_memory_usage(document_lengths, hsgm_memory, baseline_memory)
    
    print("3. Performance comparison plot")
    visualizer.plot_accuracy_comparison(model_names, accuracies, f1_scores)
    
    print("Visualizations generated successfully!")

def demo_theoretical_analysis():
    """Demonstrate theoretical complexity analysis."""
    print("\n" + "=" * 60)
    print("HSGM DEMO: Theoretical Analysis")
    print("=" * 60)
    
    from hsgm.utils import compute_complexity_metrics, compute_approximation_error_bound
    
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

def main():
    """Run all demos."""
    print("HSGM Framework Demonstration")
    print("This demo showcases the key features of the Hierarchical Segment-Graph Memory framework.")
    
    try:
        # Basic functionality demo
        hsgm_model, output = demo_basic_functionality()
        
        # Baseline comparison demo
        baseline_results = demo_comparison_with_baselines()
        
        # Streaming processing demo
        streaming_stats = demo_streaming_processing()
        
        # Visualization demo
        demo_visualization()
        
        # Theoretical analysis demo
        demo_theoretical_analysis()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The HSGM framework has been demonstrated with:")
        print("- Basic document processing and semantic graph construction")
        print("- Comparison with multiple baseline models")
        print("- Streaming document processing with incremental updates")
        print("- Visualization of performance metrics")
        print("- Theoretical complexity analysis")
        print("\nFor more detailed experiments, run: python experiments.py")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
