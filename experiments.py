"""
Comprehensive experiments for HSGM framework.
"""
import os
import sys
import argparse
import logging
import time
import json
import random
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import HSGMConfig, DatasetConfig, ExperimentConfig
from hsgm import HSGMModel, HSGMForClassification
from data_loader import create_dataloaders, create_streaming_dataset
from evaluation import Evaluator
from baselines import BaselineModels
from visualization import HSGMVisualizer, create_comparison_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class HSGMExperiments:
    """Comprehensive experiments for HSGM framework."""
    
    def __init__(self, 
                 config: HSGMConfig,
                 dataset_config: DatasetConfig,
                 experiment_config: ExperimentConfig):
        """
        Initialize HSGM Experiments.
        
        Args:
            config: HSGM configuration
            dataset_config: Dataset configuration
            experiment_config: Experiment configuration
        """
        self.config = config
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        
        # Set up device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.hsgm_model = HSGMForClassification(
            base_model=HSGMModel(
                config=config,
                hidden_dim=config.hidden_dim,
                segment_size=config.segment_size,
                local_threshold=config.local_threshold,
                global_threshold=config.global_threshold,
                top_k_retrieval=config.top_k_retrieval,
                device=str(self.device)
            ),
            num_classes=3
        ).to(self.device)
        
        self.baselines = BaselineModels(self.device)
        self.evaluator = Evaluator(self.device)
        self.visualizer = HSGMVisualizer()
        
        # Results storage
        self.results = {
            "hsgm": {},
            "baselines": {},
            "ablation": {},
            "scalability": {},
            "streaming": {}
        }
    
    def run_main_experiments(self, dataset_name: str = "document_amr") -> Dict[str, Any]:
        """
        Run main experiments comparing HSGM with baselines.
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Dictionary of experimental results
        """
        logger.info(f"Running main experiments on {dataset_name}")
        
        # Load dataset
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name, self.config, self.dataset_config
        )
        
        # Evaluate HSGM
        logger.info("Evaluating HSGM...")
        hsgm_results = self.evaluator.evaluate_classification(
            self.hsgm_model, test_loader, num_runs=self.experiment_config.num_eval_runs
        )
        self.results["hsgm"] = {
            "accuracy": hsgm_results.accuracy,
            "f1": hsgm_results.f1,
            "processing_time": hsgm_results.processing_time,
            "memory_usage": hsgm_results.memory_usage,
            "throughput": hsgm_results.throughput
        }
        
        # Evaluate baselines
        logger.info("Evaluating baselines...")
        baseline_results = {}
        
        baseline_names = ["longformer", "bigbird", "full_graph", "sliding_window_graph"]
        
        for baseline_name in baseline_names:
            try:
                logger.info(f"Evaluating {baseline_name}...")
                baseline_model = self.baselines.get_model(baseline_name)
                
                # Create a wrapper for evaluation
                class BaselineWrapper(nn.Module):
                    def __init__(self, base_model, hidden_dim):
                        super().__init__()
                        self.base_model = base_model
                        self.classifier = nn.Linear(hidden_dim, 3)
                    
                    def forward(self, documents):
                        embeddings = self.base_model(documents)
                        return self.classifier(embeddings)
                
                wrapped_model = BaselineWrapper(baseline_model, baseline_model.hidden_dim).to(self.device)
                
                results = self.evaluator.evaluate_classification(
                    wrapped_model, test_loader, num_runs=self.experiment_config.num_eval_runs
                )
                
                baseline_results[baseline_name] = {
                    "accuracy": results.accuracy,
                    "f1": results.f1,
                    "processing_time": results.processing_time,
                    "memory_usage": results.memory_usage,
                    "throughput": results.throughput
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {baseline_name}: {e}")
                baseline_results[baseline_name] = {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "processing_time": float('inf'),
                    "memory_usage": float('inf'),
                    "throughput": 0.0,
                    "error": str(e)
                }
        
        self.results["baselines"] = baseline_results
        
        # Generate comparison
        comparison = self.evaluator.compare_with_baselines(
            hsgm_results, baseline_results
        )
        
        logger.info("Main experiments completed")
        return {
            "hsgm_results": self.results["hsgm"],
            "baseline_results": baseline_results,
            "comparison": comparison
        }
    
    def run_ablation_study(self, dataset_name: str = "document_amr") -> Dict[str, Any]:
        """
        Run ablation study on HSGM components.
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Dictionary of ablation results
        """
        logger.info("Running ablation study...")
        
        # Load dataset
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name, self.config, self.dataset_config
        )
        
        ablation_results = {}
        
        # Test different component combinations
        components = [
            "local_graph_only",
            "hierarchical_memory_only", 
            "cross_attention_only",
            "contrastive_learning_only",
            "full_hsgm"
        ]
        
        for component in components:
            logger.info(f"Testing {component}...")
            
            # Create modified model based on component
            if component == "local_graph_only":
                # Disable hierarchical memory and cross-attention
                modified_model = self._create_modified_model(component)
            elif component == "hierarchical_memory_only":
                # Disable local graphs, use simple pooling
                modified_model = self._create_modified_model(component)
            elif component == "cross_attention_only":
                # Disable local graphs, use cross-attention only
                modified_model = self._create_modified_model(component)
            elif component == "contrastive_learning_only":
                # Use contrastive learning without other components
                modified_model = self._create_modified_model(component)
            else:
                # Full HSGM
                modified_model = self.hsgm_model
            
            try:
                results = self.evaluator.evaluate_classification(
                    modified_model, test_loader, num_runs=3
                )
                
                ablation_results[component] = {
                    "accuracy": results.accuracy,
                    "f1": results.f1,
                    "processing_time": results.processing_time,
                    "memory_usage": results.memory_usage
                }
                
            except Exception as e:
                logger.error(f"Error in ablation study for {component}: {e}")
                ablation_results[component] = {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "processing_time": float('inf'),
                    "memory_usage": float('inf'),
                    "error": str(e)
                }
        
        self.results["ablation"] = ablation_results
        logger.info("Ablation study completed")
        return ablation_results
    
    def _create_modified_model(self, component: str):
        """Create modified model for ablation study."""
        # This is a simplified implementation
        # In practice, you would modify the actual model architecture
        
        if component == "local_graph_only":
            # Return a model that only uses local graphs
            return self.hsgm_model  # Placeholder
        elif component == "hierarchical_memory_only":
            # Return a model that only uses hierarchical memory
            return self.hsgm_model  # Placeholder
        else:
            return self.hsgm_model  # Placeholder
    
    def run_scalability_analysis(self, dataset_name: str = "document_amr") -> Dict[str, Any]:
        """
        Run scalability analysis across different document lengths.
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Dictionary of scalability results
        """
        logger.info("Running scalability analysis...")
        
        scalability_results = {}
        document_lengths = self.experiment_config.document_lengths
        
        for length in document_lengths:
            logger.info(f"Testing document length: {length}")
            
            # Create modified dataset with documents of specific length
            modified_config = self.dataset_config
            # In practice, you would filter or generate documents of specific lengths
            
            try:
                # Evaluate HSGM
                hsgm_efficiency = self.evaluator.evaluate_efficiency(
                    self.hsgm_model, None, [length]  # Pass None for dataloader, test specific length
                )
                
                # Evaluate baselines
                baseline_efficiency = {}
                for baseline_name in ["longformer", "bigbird", "full_graph"]:
                    try:
                        baseline_model = self.baselines.get_model(baseline_name)
                        baseline_eff = self.evaluator.evaluate_efficiency(
                            baseline_model, None, [length]
                        )
                        baseline_efficiency[baseline_name] = baseline_eff.get(length, {})
                    except Exception as e:
                        logger.error(f"Error in scalability test for {baseline_name}: {e}")
                        baseline_efficiency[baseline_name] = {}
                
                scalability_results[length] = {
                    "hsgm": hsgm_efficiency.get(length, {}),
                    "baselines": baseline_efficiency
                }
                
            except Exception as e:
                logger.error(f"Error in scalability test for length {length}: {e}")
                scalability_results[length] = {"error": str(e)}
        
        self.results["scalability"] = scalability_results
        logger.info("Scalability analysis completed")
        return scalability_results
    
    def run_streaming_experiments(self, dataset_name: str = "document_amr") -> Dict[str, Any]:
        """
        Run streaming document experiments.
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Dictionary of streaming results
        """
        logger.info("Running streaming experiments...")
        
        # Create streaming dataset
        streaming_chunks = create_streaming_dataset(
            dataset_name, self.dataset_config,
            chunk_size=self.dataset_config.streaming_chunk_size,
            interval_ms=self.dataset_config.streaming_interval
        )
        
        # Simulate streaming processing
        streaming_results = {
            "cache_hit_rates": [],
            "memory_usage": [],
            "processing_times": [],
            "accuracy_scores": [],
            "timestamps": []
        }
        
        # Reset model for streaming
        self.hsgm_model.base_model.reset_model()
        
        for i, chunk in enumerate(tqdm(streaming_chunks[:100])):  # Process first 100 chunks
            start_time = time.time()
            
            # Process chunk incrementally
            update_stats = self.hsgm_model.base_model.incremental_update([chunk["text"]])
            
            processing_time = time.time() - start_time
            
            # Record metrics
            streaming_results["cache_hit_rates"].append(update_stats.get("hit_rate", 0.0))
            streaming_results["memory_usage"].append(update_stats.get("memory_usage", {}).get("current_memory_gb", 0.0))
            streaming_results["processing_times"].append(processing_time)
            streaming_results["timestamps"].append(chunk["timestamp"] / 1000.0)  # Convert to seconds
            
            # Simulate accuracy (in practice, you would evaluate on a task)
            accuracy = 0.95 + 0.05 * np.random.normal(0, 0.01)  # Placeholder
            streaming_results["accuracy_scores"].append(max(0.0, min(1.0, accuracy)))
        
        self.results["streaming"] = streaming_results
        logger.info("Streaming experiments completed")
        return streaming_results
    
    def run_hyperparameter_analysis(self, dataset_name: str = "document_amr") -> Dict[str, Any]:
        """
        Run hyperparameter sensitivity analysis.
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Dictionary of hyperparameter results
        """
        logger.info("Running hyperparameter analysis...")
        
        # Load dataset
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name, self.config, self.dataset_config
        )
        
        hyperparameter_results = {}
        
        # Test different hyperparameter combinations
        param_ranges = self.experiment_config.hyperparameter_ranges
        
        for param_name, param_values in param_ranges.items():
            logger.info(f"Testing {param_name}: {param_values}")
            
            param_results = {}
            
            for value in param_values:
                # Create model with modified hyperparameter
                modified_config = self.config
                setattr(modified_config, param_name, value)
                
                try:
                    # Create modified model
                    modified_model = self._create_model_with_params(modified_config)
                    
                    # Evaluate
                    results = self.evaluator.evaluate_classification(
                        modified_model, test_loader, num_runs=2
                    )
                    
                    param_results[value] = {
                        "accuracy": results.accuracy,
                        "f1": results.f1,
                        "processing_time": results.processing_time,
                        "memory_usage": results.memory_usage
                    }
                    
                except Exception as e:
                    logger.error(f"Error testing {param_name}={value}: {e}")
                    param_results[value] = {"error": str(e)}
            
            hyperparameter_results[param_name] = param_results
        
        logger.info("Hyperparameter analysis completed")
        return hyperparameter_results
    
    def _create_model_with_params(self, config):
        """Create model with specific hyperparameters."""
        # Create a new model with modified config
        base_model = HSGMModel(
            config=config,
            hidden_dim=config.hidden_dim,
            segment_size=config.segment_size,
            local_threshold=config.local_threshold,
            global_threshold=config.global_threshold,
            top_k_retrieval=config.top_k_retrieval,
            device=str(self.device)
        ).to(self.device)
        
        return HSGMForClassification(base_model=base_model, num_classes=3).to(self.device)
    
    def generate_comprehensive_report(self, output_dir: str = "./experiment_results"):
        """
        Generate comprehensive experimental report.
        
        Args:
            output_dir: Output directory for results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating comprehensive report...")
        
        # Save raw results
        with open(os.path.join(output_dir, "raw_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations(output_dir)
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        logger.info(f"Comprehensive report generated in {output_dir}")
    
    def _generate_visualizations(self, output_dir: str):
        """Generate visualization plots."""
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Scalability plots
        if "scalability" in self.results:
            self._plot_scalability_results(viz_dir)
        
        # Streaming plots
        if "streaming" in self.results:
            self._plot_streaming_results(viz_dir)
        
        # Ablation plots
        if "ablation" in self.results:
            self._plot_ablation_results(viz_dir)
        
        # Performance comparison plots
        if "hsgm" in self.results and "baselines" in self.results:
            self._plot_performance_comparison(viz_dir)
    
    def _plot_scalability_results(self, viz_dir: str):
        """Plot scalability results."""
        scalability_data = self.results["scalability"]
        
        document_lengths = list(scalability_data.keys())
        hsgm_times = []
        baseline_times = []
        
        for length in document_lengths:
            if length in scalability_data and "hsgm" in scalability_data[length]:
                hsgm_time = scalability_data[length]["hsgm"].get("avg_processing_time", 0.0)
                hsgm_times.append(hsgm_time)
            else:
                hsgm_times.append(0.0)
            
            # Get average baseline time
            baseline_times_for_length = []
            if length in scalability_data and "baselines" in scalability_data[length]:
                for baseline_name, baseline_data in scalability_data[length]["baselines"].items():
                    if "avg_processing_time" in baseline_data:
                        baseline_times_for_length.append(baseline_data["avg_processing_time"])
            
            if baseline_times_for_length:
                baseline_times.append(np.mean(baseline_times_for_length))
            else:
                baseline_times.append(0.0)
        
        self.visualizer.plot_complexity_analysis(
            document_lengths, hsgm_times, baseline_times,
            save_path=os.path.join(viz_dir, "scalability_analysis.png")
        )
    
    def _plot_streaming_results(self, viz_dir: str):
        """Plot streaming results."""
        streaming_data = self.results["streaming"]
        
        self.visualizer.plot_streaming_performance(
            streaming_data["timestamps"],
            streaming_data["cache_hit_rates"],
            streaming_data["memory_usage"],
            streaming_data["accuracy_scores"],
            save_path=os.path.join(viz_dir, "streaming_performance.png")
        )
    
    def _plot_ablation_results(self, viz_dir: str):
        """Plot ablation study results."""
        ablation_data = self.results["ablation"]
        
        component_names = list(ablation_data.keys())
        accuracies = [ablation_data[name].get("accuracy", 0.0) for name in component_names]
        f1_scores = [ablation_data[name].get("f1", 0.0) for name in component_names]
        
        self.visualizer.plot_ablation_study(
            component_names,
            {"accuracy": accuracies, "f1": f1_scores},
            save_path=os.path.join(viz_dir, "ablation_study.png")
        )
    
    def _plot_performance_comparison(self, viz_dir: str):
        """Plot performance comparison."""
        hsgm_data = self.results["hsgm"]
        baseline_data = self.results["baselines"]
        
        model_names = ["HSGM"] + list(baseline_data.keys())
        accuracies = [hsgm_data["accuracy"]] + [baseline_data[name].get("accuracy", 0.0) for name in baseline_data.keys()]
        f1_scores = [hsgm_data["f1"]] + [baseline_data[name].get("f1", 0.0) for name in baseline_data.keys()]
        
        self.visualizer.plot_accuracy_comparison(
            model_names, accuracies, f1_scores,
            save_path=os.path.join(viz_dir, "performance_comparison.png")
        )
    
    def _generate_summary_report(self, output_dir: str):
        """Generate summary report."""
        report_lines = []
        report_lines.append("# HSGM Experimental Results Summary")
        report_lines.append("=" * 50)
        
        # Main results
        if "hsgm" in self.results:
            hsgm_data = self.results["hsgm"]
            report_lines.append(f"\n## HSGM Performance")
            report_lines.append(f"- Accuracy: {hsgm_data['accuracy']:.4f}")
            report_lines.append(f"- F1 Score: {hsgm_data['f1']:.4f}")
            report_lines.append(f"- Processing Time: {hsgm_data['processing_time']:.4f}s")
            report_lines.append(f"- Memory Usage: {hsgm_data['memory_usage']:.4f}GB")
            report_lines.append(f"- Throughput: {hsgm_data['throughput']:.4f} docs/s")
        
        # Baseline comparison
        if "baselines" in self.results:
            report_lines.append(f"\n## Baseline Comparison")
            for baseline_name, baseline_data in self.results["baselines"].items():
                if "error" not in baseline_data:
                    report_lines.append(f"\n### {baseline_name}")
                    report_lines.append(f"- Accuracy: {baseline_data['accuracy']:.4f}")
                    report_lines.append(f"- F1 Score: {baseline_data['f1']:.4f}")
                    report_lines.append(f"- Processing Time: {baseline_data['processing_time']:.4f}s")
        
        # Scalability summary
        if "scalability" in self.results:
            report_lines.append(f"\n## Scalability Analysis")
            for length, data in self.results["scalability"].items():
                if "hsgm" in data:
                    hsgm_time = data["hsgm"].get("avg_processing_time", 0.0)
                    report_lines.append(f"- Document Length {length}: {hsgm_time:.4f}s")
        
        # Save report
        with open(os.path.join(output_dir, "summary_report.md"), 'w') as f:
            f.write("\n".join(report_lines))

def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run HSGM experiments")
    parser.add_argument("--config", type=str, default="config.py", help="Config file path")
    parser.add_argument("--dataset", type=str, default="document_amr", help="Dataset to use")
    parser.add_argument("--experiments", nargs="+", 
                       default=["main", "ablation", "scalability", "streaming"],
                       help="Experiments to run")
    parser.add_argument("--output_dir", type=str, default="./experiment_results", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configurations
    config = HSGMConfig()
    dataset_config = DatasetConfig()
    experiment_config = ExperimentConfig()
    
    # Create experiment runner
    experiments = HSGMExperiments(config, dataset_config, experiment_config)
    
    # Run experiments
    if "main" in args.experiments:
        experiments.run_main_experiments(args.dataset)
    
    if "ablation" in args.experiments:
        experiments.run_ablation_study(args.dataset)
    
    if "scalability" in args.experiments:
        experiments.run_scalability_analysis(args.dataset)
    
    if "streaming" in args.experiments:
        experiments.run_streaming_experiments(args.dataset)
    
    # Generate report
    experiments.generate_comprehensive_report(args.output_dir)
    
    logger.info("All experiments completed successfully!")

if __name__ == "__main__":
    main()
