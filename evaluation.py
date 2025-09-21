"""
Evaluation utilities for HSGM framework.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, mean_squared_error, mean_absolute_error
)
from rouge import Rouge
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    auc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bleu: float = 0.0
    smatch_f1: float = 0.0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0

class Evaluator:
    """Comprehensive evaluator for HSGM models."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        self.device = device
        self.rouge = Rouge()
    
    def evaluate_classification(self,
                              model,
                              dataloader,
                              num_runs: int = 5) -> EvaluationMetrics:
        """
        Evaluate model on classification tasks.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            num_runs: Number of evaluation runs for statistical significance
            
        Returns:
            EvaluationMetrics object
        """
        all_metrics = []
        
        for run in range(num_runs):
            predictions = []
            labels = []
            processing_times = []
            memory_usages = []
            
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    start_time = time.time()
                    
                    # Move batch to device
                    documents = batch["documents"]
                    batch_labels = batch.get("labels", None)
                    
                    if batch_labels is not None:
                        batch_labels = batch_labels.to(self.device)
                    
                    # Forward pass
                    if hasattr(model, 'base_model'):
                        # Task-specific model
                        logits = model(documents)
                    else:
                        # Base HSGM model
                        output = model(documents)
                        # For classification, we'd need a classification head
                        logits = torch.randn(len(documents), 3).to(self.device)  # Placeholder
                    
                    # Record processing time
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # Record memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_usages.append(memory_usage)
                    
                    # Get predictions
                    batch_predictions = torch.argmax(logits, dim=-1)
                    predictions.extend(batch_predictions.cpu().numpy())
                    
                    if batch_labels is not None:
                        labels.extend(batch_labels.cpu().numpy())
            
            # Compute metrics for this run
            if labels:
                metrics = self._compute_classification_metrics(
                    predictions, labels, processing_times, memory_usages
                )
                all_metrics.append(metrics)
        
        # Average metrics across runs
        return self._average_metrics(all_metrics)
    
    def evaluate_generation(self,
                          model,
                          dataloader,
                          num_runs: int = 5) -> EvaluationMetrics:
        """
        Evaluate model on text generation tasks.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            num_runs: Number of evaluation runs
            
        Returns:
            EvaluationMetrics object
        """
        all_metrics = []
        
        for run in range(num_runs):
            generated_texts = []
            target_texts = []
            processing_times = []
            
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    start_time = time.time()
                    
                    documents = batch["documents"]
                    targets = batch.get("target_sequences", [])
                    
                    # Generate text (placeholder implementation)
                    # In a real implementation, this would use the model's generation capabilities
                    for i, doc in enumerate(documents):
                        # Placeholder generation
                        generated_text = f"Generated summary for document {i}"
                        generated_texts.append(generated_text)
                        
                        if i < len(targets):
                            target_texts.append(targets[i])
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
            
            # Compute generation metrics
            metrics = self._compute_generation_metrics(
                generated_texts, target_texts, processing_times
            )
            all_metrics.append(metrics)
        
        return self._average_metrics(all_metrics)
    
    def evaluate_semantic_parsing(self,
                                model,
                                dataloader,
                                num_runs: int = 5) -> EvaluationMetrics:
        """
        Evaluate model on semantic parsing tasks (AMR, SRL, etc.).
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            num_runs: Number of evaluation runs
            
        Returns:
            EvaluationMetrics object
        """
        all_metrics = []
        
        for run in range(num_runs):
            predictions = []
            targets = []
            processing_times = []
            
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    start_time = time.time()
                    
                    documents = batch["documents"]
                    
                    # Process documents with HSGM
                    output = model(documents, return_graphs=True)
                    
                    # Extract semantic structures (placeholder)
                    for i, doc in enumerate(documents):
                        # Placeholder semantic parsing
                        pred_structure = {"nodes": [], "edges": []}
                        target_structure = {"nodes": [], "edges": []}  # Would come from labels
                        
                        predictions.append(pred_structure)
                        targets.append(target_structure)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
            
            # Compute semantic parsing metrics
            metrics = self._compute_semantic_parsing_metrics(
                predictions, targets, processing_times
            )
            all_metrics.append(metrics)
        
        return self._average_metrics(all_metrics)
    
    def evaluate_efficiency(self,
                          model,
                          dataloader,
                          document_lengths: List[int] = None) -> Dict[str, Any]:
        """
        Evaluate computational efficiency across different document lengths.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            document_lengths: List of document lengths to test
            
        Returns:
            Dictionary of efficiency metrics
        """
        if document_lengths is None:
            document_lengths = [1000, 5000, 10000, 20000]
        
        efficiency_results = {}
        
        for length in document_lengths:
            # Filter documents by length
            filtered_batches = []
            for batch in dataloader:
                filtered_docs = []
                for doc in batch["documents"]:
                    if len(doc.split()) <= length:
                        filtered_docs.append(doc)
                
                if filtered_docs:
                    filtered_batch = {k: v for k, v in batch.items()}
                    filtered_batch["documents"] = filtered_docs
                    filtered_batches.append(filtered_batch)
            
            if not filtered_batches:
                continue
            
            # Measure performance
            processing_times = []
            memory_usages = []
            throughputs = []
            
            model.eval()
            with torch.no_grad():
                for batch in filtered_batches:
                    start_time = time.time()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    
                    # Process batch
                    output = model(batch["documents"])
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB
                        memory_usages.append(memory_usage)
                    
                    # Throughput (documents per second)
                    throughput = len(batch["documents"]) / processing_time
                    throughputs.append(throughput)
            
            efficiency_results[length] = {
                "avg_processing_time": np.mean(processing_times),
                "std_processing_time": np.std(processing_times),
                "avg_memory_usage": np.mean(memory_usages) if memory_usages else 0,
                "std_memory_usage": np.std(memory_usages) if memory_usages else 0,
                "avg_throughput": np.mean(throughputs),
                "std_throughput": np.std(throughputs),
                "num_documents": sum(len(batch["documents"]) for batch in filtered_batches)
            }
        
        return efficiency_results
    
    def _compute_classification_metrics(self,
                                      predictions: List[int],
                                      labels: List[int],
                                      processing_times: List[float],
                                      memory_usages: List[float]) -> EvaluationMetrics:
        """Compute classification metrics."""
        metrics = EvaluationMetrics()
        
        # Basic metrics
        metrics.accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        metrics.precision = precision
        metrics.recall = recall
        metrics.f1 = f1
        metrics.f1_macro = f1_macro
        metrics.f1_weighted = f1
        
        # Try to compute AUC (for binary classification)
        try:
            if len(set(labels)) == 2:
                metrics.auc = roc_auc_score(labels, predictions)
        except:
            metrics.auc = 0.0
        
        # Efficiency metrics
        metrics.processing_time = np.mean(processing_times)
        metrics.memory_usage = np.mean(memory_usages) if memory_usages else 0
        metrics.throughput = len(predictions) / np.sum(processing_times)
        
        return metrics
    
    def _compute_generation_metrics(self,
                                  generated_texts: List[str],
                                  target_texts: List[str],
                                  processing_times: List[float]) -> EvaluationMetrics:
        """Compute text generation metrics."""
        metrics = EvaluationMetrics()
        
        # ROUGE metrics
        try:
            rouge_scores = self.rouge.get_scores(generated_texts, target_texts)
            metrics.rouge_1 = np.mean([score['rouge-1']['f'] for score in rouge_scores])
            metrics.rouge_2 = np.mean([score['rouge-2']['f'] for score in rouge_scores])
            metrics.rouge_l = np.mean([score['rouge-l']['f'] for score in rouge_scores])
        except:
            metrics.rouge_1 = 0.0
            metrics.rouge_2 = 0.0
            metrics.rouge_l = 0.0
        
        # BLEU score (simplified)
        try:
            from nltk.translate.bleu_score import sentence_bleu
            bleu_scores = []
            for gen, target in zip(generated_texts, target_texts):
                gen_tokens = gen.split()
                target_tokens = target.split()
                bleu = sentence_bleu([target_tokens], gen_tokens)
                bleu_scores.append(bleu)
            metrics.bleu = np.mean(bleu_scores)
        except:
            metrics.bleu = 0.0
        
        # Efficiency metrics
        metrics.processing_time = np.mean(processing_times)
        metrics.throughput = len(generated_texts) / np.sum(processing_times)
        
        return metrics
    
    def _compute_semantic_parsing_metrics(self,
                                        predictions: List[Dict],
                                        targets: List[Dict],
                                        processing_times: List[float]) -> EvaluationMetrics:
        """Compute semantic parsing metrics."""
        metrics = EvaluationMetrics()
        
        # Smatch F1 for AMR parsing (simplified)
        try:
            smatch_scores = []
            for pred, target in zip(predictions, targets):
                # Simplified Smatch computation
                pred_nodes = len(pred.get("nodes", []))
                target_nodes = len(target.get("nodes", []))
                pred_edges = len(pred.get("edges", []))
                target_edges = len(target.get("edges", []))
                
                # Simple overlap-based score
                node_overlap = min(pred_nodes, target_nodes) / max(pred_nodes, target_nodes, 1)
                edge_overlap = min(pred_edges, target_edges) / max(pred_edges, target_edges, 1)
                smatch = (node_overlap + edge_overlap) / 2
                smatch_scores.append(smatch)
            
            metrics.smatch_f1 = np.mean(smatch_scores)
        except:
            metrics.smatch_f1 = 0.0
        
        # Efficiency metrics
        metrics.processing_time = np.mean(processing_times)
        metrics.throughput = len(predictions) / np.sum(processing_times)
        
        return metrics
    
    def _average_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Average metrics across multiple runs."""
        if not metrics_list:
            return EvaluationMetrics()
        
        avg_metrics = EvaluationMetrics()
        
        for field in avg_metrics.__dataclass_fields__:
            values = [getattr(metrics, field) for metrics in metrics_list]
            setattr(avg_metrics, field, np.mean(values))
        
        return avg_metrics
    
    def compare_with_baselines(self,
                             hsgm_metrics: EvaluationMetrics,
                             baseline_metrics: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """
        Compare HSGM metrics with baseline models.
        
        Args:
            hsgm_metrics: HSGM evaluation metrics
            baseline_metrics: Dictionary of baseline metrics
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for baseline_name, baseline_metric in baseline_metrics.items():
            comparison[baseline_name] = {
                "accuracy_improvement": hsgm_metrics.accuracy - baseline_metric.accuracy,
                "f1_improvement": hsgm_metrics.f1 - baseline_metric.f1,
                "speedup": baseline_metric.processing_time / hsgm_metrics.processing_time,
                "memory_reduction": baseline_metric.memory_usage - hsgm_metrics.memory_usage,
                "throughput_improvement": hsgm_metrics.throughput - baseline_metric.throughput
            }
        
        return comparison
    
    def generate_report(self,
                       metrics: EvaluationMetrics,
                       efficiency_results: Dict[str, Any],
                       comparison_results: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Main evaluation metrics
            efficiency_results: Efficiency evaluation results
            comparison_results: Baseline comparison results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("HSGM EVALUATION REPORT")
        report.append("=" * 60)
        
        # Performance metrics
        report.append("\nPERFORMANCE METRICS:")
        report.append("-" * 30)
        report.append(f"Accuracy: {metrics.accuracy:.4f}")
        report.append(f"Precision: {metrics.precision:.4f}")
        report.append(f"Recall: {metrics.recall:.4f}")
        report.append(f"F1 Score: {metrics.f1:.4f}")
        report.append(f"F1 Macro: {metrics.f1_macro:.4f}")
        report.append(f"F1 Weighted: {metrics.f1_weighted:.4f}")
        report.append(f"AUC: {metrics.auc:.4f}")
        
        # Generation metrics
        if metrics.rouge_1 > 0 or metrics.rouge_2 > 0:
            report.append(f"\nGENERATION METRICS:")
            report.append("-" * 30)
            report.append(f"ROUGE-1: {metrics.rouge_1:.4f}")
            report.append(f"ROUGE-2: {metrics.rouge_2:.4f}")
            report.append(f"ROUGE-L: {metrics.rouge_l:.4f}")
            report.append(f"BLEU: {metrics.bleu:.4f}")
        
        # Semantic parsing metrics
        if metrics.smatch_f1 > 0:
            report.append(f"\nSEMANTIC PARSING METRICS:")
            report.append("-" * 30)
            report.append(f"Smatch F1: {metrics.smatch_f1:.4f}")
        
        # Efficiency metrics
        report.append(f"\nEFFICIENCY METRICS:")
        report.append("-" * 30)
        report.append(f"Processing Time: {metrics.processing_time:.4f}s")
        report.append(f"Memory Usage: {metrics.memory_usage:.4f}GB")
        report.append(f"Throughput: {metrics.throughput:.4f} docs/s")
        
        # Scalability analysis
        if efficiency_results:
            report.append(f"\nSCALABILITY ANALYSIS:")
            report.append("-" * 30)
            for length, results in efficiency_results.items():
                report.append(f"Document Length {length}:")
                report.append(f"  Processing Time: {results['avg_processing_time']:.4f}±{results['std_processing_time']:.4f}s")
                report.append(f"  Memory Usage: {results['avg_memory_usage']:.4f}±{results['std_memory_usage']:.4f}GB")
                report.append(f"  Throughput: {results['avg_throughput']:.4f}±{results['std_throughput']:.4f} docs/s")
        
        # Baseline comparison
        if comparison_results:
            report.append(f"\nBASELINE COMPARISON:")
            report.append("-" * 30)
            for baseline_name, comparison in comparison_results.items():
                report.append(f"{baseline_name}:")
                report.append(f"  Accuracy Improvement: {comparison['accuracy_improvement']:+.4f}")
                report.append(f"  F1 Improvement: {comparison['f1_improvement']:+.4f}")
                report.append(f"  Speedup: {comparison['speedup']:.2f}x")
                report.append(f"  Memory Reduction: {comparison['memory_reduction']:.4f}GB")
                report.append(f"  Throughput Improvement: {comparison['throughput_improvement']:+.4f} docs/s")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_results(self,
                    metrics: EvaluationMetrics,
                    efficiency_results: Dict[str, Any],
                    comparison_results: Dict[str, Any] = None,
                    output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to file.
        
        Args:
            metrics: Main evaluation metrics
            efficiency_results: Efficiency evaluation results
            comparison_results: Baseline comparison results
            output_path: Output file path
        """
        results = {
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f1_macro": metrics.f1_macro,
                "f1_weighted": metrics.f1_weighted,
                "auc": metrics.auc,
                "rouge_1": metrics.rouge_1,
                "rouge_2": metrics.rouge_2,
                "rouge_l": metrics.rouge_l,
                "bleu": metrics.bleu,
                "smatch_f1": metrics.smatch_f1,
                "processing_time": metrics.processing_time,
                "memory_usage": metrics.memory_usage,
                "throughput": metrics.throughput
            },
            "efficiency_results": efficiency_results,
            "comparison_results": comparison_results or {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
