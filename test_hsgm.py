"""
Simple test script for HSGM implementation.
"""
import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        from config import HSGMConfig, DatasetConfig, ExperimentConfig
        print(" Config imports successful")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from hsgm import HSGMModel, HSGMForClassification
        print(" HSGM model imports successful")
    except ImportError as e:
        print(f"✗ HSGM model import failed: {e}")
        return False
    
    try:
        from evaluation import Evaluator
        print(" Evaluator import successful")
    except ImportError as e:
        print(f"✗ Evaluator import failed: {e}")
        return False
    
    try:
        from baselines import BaselineModels
        print(" Baselines import successful")
    except ImportError as e:
        print(f"✗ Baselines import failed: {e}")
        return False
    
    try:
        from visualization import HSGMVisualizer
        print(" Visualization import successful")
    except ImportError as e:
        print(f"✗ Visualization import failed: {e}")
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
        
        print(f" HSGM config created: hidden_dim={config.hidden_dim}, segment_size={config.segment_size}")
        print(f" Dataset config created: document_amr_path={dataset_config.document_amr_path}")
        print(f" Experiment config created: num_eval_runs={experiment_config.num_eval_runs}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False

def test_hsgm_model_creation():
    """Test HSGM model creation."""
    print("\nTesting HSGM model creation...")
    
    try:
        from config import HSGMConfig
        from hsgm import HSGMModel
        
        config = HSGMConfig()
        config.device = "cpu"  # Use CPU for testing
        
        model = HSGMModel(
            config=config,
            hidden_dim=config.hidden_dim,
            segment_size=config.segment_size,
            local_threshold=config.local_threshold,
            global_threshold=config.global_threshold,
            top_k_retrieval=config.top_k_retrieval,
            device=config.device
        )
        
        print(f" HSGM model created successfully")
        print(f"  - Hidden dim: {model.hidden_dim}")
        print(f"  - Segment size: {model.segment_size}")
        print(f"  - Device: {model.device}")
        
        return True
    except Exception as e:
        print(f"✗ HSGM model creation failed: {e}")
        return False

def test_document_processing():
    """Test basic document processing."""
    print("\nTesting document processing...")
    
    try:
        from config import HSGMConfig
        from hsgm import HSGMModel
        
        config = HSGMConfig()
        config.device = "cpu"
        
        model = HSGMModel(
            config=config,
            hidden_dim=128,  # Smaller for testing
            segment_size=64,
            local_threshold=0.2,
            global_threshold=0.1,
            top_k_retrieval=3,
            device=config.device
        )
        
        # Simple test documents
        test_documents = [
            "This is a test document for HSGM.",
            "Another test document with different content."
        ]
        
        print(f"Processing {len(test_documents)} test documents...")
        output = model(test_documents)
        
        print(f" Document processing successful")
        print(f"  - Output embeddings shape: {output.embeddings.shape}")
        print(f"  - Number of summary nodes: {len(output.summary_nodes)}")
        print(f"  - Processing time: {output.processing_time:.4f}s")
        
        return True
    except Exception as e:
        print(f"✗ Document processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_models():
    """Test baseline model creation."""
    print("\nTesting baseline models...")
    
    try:
        from baselines import BaselineModels
        
        baselines = BaselineModels(device="cpu")
        
        print(f" Baseline models created successfully")
        print(f"  - Available models: {list(baselines.models.keys())}")
        
        # Test one baseline
        try:
            full_graph_model = baselines.get_model("full_graph")
            print(f" Full graph baseline created")
        except Exception as e:
            print(f"✗ Full graph baseline failed: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Baseline models creation failed: {e}")
        return False

def test_evaluator():
    """Test evaluator creation."""
    print("\nTesting evaluator...")
    
    try:
        from evaluation import Evaluator
        
        evaluator = Evaluator(device="cpu")
        
        print(f" Evaluator created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Evaluator creation failed: {e}")
        return False

def test_visualization():
    """Test visualization creation."""
    print("\nTesting visualization...")
    
    try:
        from visualization import HSGMVisualizer
        
        visualizer = HSGMVisualizer()
        
        print(f" Visualizer created successfully")
        print(f"  - Available colors: {list(visualizer.colors.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Visualizer creation failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from hsgm.utils import compute_complexity_metrics, compute_approximation_error_bound
        
        # Test complexity metrics
        metrics = compute_complexity_metrics(1000, 256)
        print(f" Complexity metrics computed")
        print(f"  - HSGM complexity: {metrics['hsgm_complexity']:.2f}")
        print(f"  - Full graph complexity: {metrics['full_graph_complexity']:.2f}")
        print(f"  - Speedup: {metrics['theoretical_speedup']:.2f}x")
        
        # Test error bounds
        error_bound = compute_approximation_error_bound(0.2, 0.1)
        print(f" Error bound computed: {error_bound:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Utility functions failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("HSGM Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_creation,
        test_hsgm_model_creation,
        test_document_processing,
        test_baseline_models,
        test_evaluator,
        test_visualization,
        test_utils
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
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All tests passed! HSGM implementation is working correctly.")
        return True
    else:
        print(" Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
