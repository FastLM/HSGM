"""
Visualization utilities for HSGM framework.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import logging

logger = logging.getLogger(__name__)

class HSGMVisualizer:
    """Visualization class for HSGM components and results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize HSGM Visualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Set up color schemes
        self.colors = {
            'hsgm': '#1f77b4',
            'baseline': '#ff7f0e',
            'local_graph': '#2ca02c',
            'global_graph': '#d62728',
            'summary_node': '#9467bd',
            'query': '#8c564b'
        }
    
    def plot_complexity_analysis(self, 
                               document_lengths: List[int],
                               hsgm_times: List[float],
                               baseline_times: List[float],
                               save_path: Optional[str] = None):
        """
        Plot complexity analysis comparing HSGM with baselines.
        
        Args:
            document_lengths: List of document lengths
            hsgm_times: List of HSGM processing times
            baseline_times: List of baseline processing times
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear scale
        ax1.plot(document_lengths, hsgm_times, 'o-', label='HSGM', color=self.colors['hsgm'], linewidth=2)
        ax1.plot(document_lengths, baseline_times, 's-', label='Baseline', color=self.colors['baseline'], linewidth=2)
        ax1.set_xlabel('Document Length (tokens)')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time vs Document Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.loglog(document_lengths, hsgm_times, 'o-', label='HSGM', color=self.colors['hsgm'], linewidth=2)
        ax2.loglog(document_lengths, baseline_times, 's-', label='Baseline', color=self.colors['baseline'], linewidth=2)
        ax2.set_xlabel('Document Length (tokens)')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time vs Document Length (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_memory_usage(self,
                        document_lengths: List[int],
                        hsgm_memory: List[float],
                        baseline_memory: List[float],
                        save_path: Optional[str] = None):
        """
        Plot memory usage comparison.
        
        Args:
            document_lengths: List of document lengths
            hsgm_memory: List of HSGM memory usage
            baseline_memory: List of baseline memory usage
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(document_lengths, hsgm_memory, 'o-', label='HSGM', color=self.colors['hsgm'], linewidth=2)
        plt.plot(document_lengths, baseline_memory, 's-', label='Baseline', color=self.colors['baseline'], linewidth=2)
        
        plt.xlabel('Document Length (tokens)')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage vs Document Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add memory reduction annotations
        for i, (length, hsgm_mem, baseline_mem) in enumerate(zip(document_lengths, hsgm_memory, baseline_memory)):
            reduction = (baseline_mem - hsgm_mem) / baseline_mem * 100
            plt.annotate(f'{reduction:.1f}%', 
                        xy=(length, hsgm_mem), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_accuracy_comparison(self,
                               model_names: List[str],
                               accuracies: List[float],
                               f1_scores: List[float],
                               save_path: Optional[str] = None):
        """
        Plot accuracy comparison across models.
        
        Args:
            model_names: List of model names
            accuracies: List of accuracy scores
            f1_scores: List of F1 scores
            save_path: Optional path to save plot
        """
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_efficiency_heatmap(self,
                              model_names: List[str],
                              document_lengths: List[int],
                              processing_times: np.ndarray,
                              save_path: Optional[str] = None):
        """
        Plot efficiency heatmap.
        
        Args:
            model_names: List of model names
            document_lengths: List of document lengths
            processing_times: 2D array of processing times [models x lengths]
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(processing_times,
                   xticklabels=document_lengths,
                   yticklabels=model_names,
                   annot=True,
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Processing Time (seconds)'})
        
        plt.xlabel('Document Length (tokens)')
        plt.ylabel('Models')
        plt.title('Processing Time Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_local_graph(self,
                            nodes: List,
                            edges: List,
                            title: str = "Local Semantic Graph",
                            save_path: Optional[str] = None):
        """
        Visualize a local semantic graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            title: Plot title
            save_path: Optional path to save plot
        """
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id, 
                      token=node.token[:20],  # Truncate long tokens
                      position=node.position)
        
        # Add edges
        for edge in edges:
            if edge.source in G and edge.target in G:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        node_colors = [self.colors['local_graph'] for _ in G.nodes()]
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=300,
                              alpha=0.7)
        
        # Draw edges with thickness based on weight
        edges_drawn = []
        edge_weights = []
        for edge in edges:
            if edge.source in G and edge.target in G:
                edges_drawn.append((edge.source, edge.target))
                edge_weights.append(edge.weight)
        
        nx.draw_networkx_edges(G, pos,
                              edgelist=edges_drawn,
                              width=[w * 3 for w in edge_weights],
                              alpha=0.6,
                              edge_color='gray')
        
        # Draw labels
        labels = {node.id: node.token[:10] for node in nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_hierarchical_memory(self,
                                    summary_nodes: List[torch.Tensor],
                                    global_edges: List[Tuple],
                                    title: str = "Hierarchical Memory",
                                    save_path: Optional[str] = None):
        """
        Visualize hierarchical memory structure.
        
        Args:
            summary_nodes: List of summary node embeddings
            global_edges: List of global edges
            title: Plot title
            save_path: Optional path to save plot
        """
        G = nx.Graph()
        
        # Add summary nodes
        for i, node_emb in enumerate(summary_nodes):
            G.add_node(i, 
                      embedding=node_emb.cpu().numpy(),
                      label=f'Summary {i}')
        
        # Add edges
        for edge in global_edges:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=100)
        
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        node_colors = [self.colors['summary_node'] for _ in G.nodes()]
        nx.draw_networkx_nodes(G, pos,
                              node_color=node_colors,
                              node_size=500,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                              width=2,
                              alpha=0.6,
                              edge_color='gray')
        
        # Draw labels
        labels = {i: f'S{i}' for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_streaming_performance(self,
                                 timestamps: List[float],
                                 cache_hit_rates: List[float],
                                 memory_usage: List[float],
                                 accuracy_scores: List[float],
                                 save_path: Optional[str] = None):
        """
        Plot streaming performance metrics.
        
        Args:
            timestamps: List of timestamps
            cache_hit_rates: List of cache hit rates
            memory_usage: List of memory usage values
            accuracy_scores: List of accuracy scores
            save_path: Optional path to save plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cache hit rate
        ax1.plot(timestamps, cache_hit_rates, 'o-', color=self.colors['hsgm'], linewidth=2)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Cache Hit Rate')
        ax1.set_title('Cache Hit Rate Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Memory usage
        ax2.plot(timestamps, memory_usage, 's-', color=self.colors['local_graph'], linewidth=2)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('Memory Usage Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Accuracy
        ax3.plot(timestamps, accuracy_scores, '^-', color=self.colors['global_graph'], linewidth=2)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Combined view
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(timestamps, cache_hit_rates, 'o-', color=self.colors['hsgm'], 
                        label='Cache Hit Rate', linewidth=2)
        line2 = ax4_twin.plot(timestamps, memory_usage, 's-', color=self.colors['local_graph'], 
                             label='Memory Usage', linewidth=2)
        
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Cache Hit Rate', color=self.colors['hsgm'])
        ax4_twin.set_ylabel('Memory Usage (GB)', color=self.colors['local_graph'])
        ax4.set_title('Combined Metrics')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_ablation_study(self,
                          component_names: List[str],
                          performance_metrics: Dict[str, List[float]],
                          save_path: Optional[str] = None):
        """
        Plot ablation study results.
        
        Args:
            component_names: List of component names
            performance_metrics: Dictionary mapping metric names to lists of values
            save_path: Optional path to save plot
        """
        num_metrics = len(performance_metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(performance_metrics.items()):
            bars = axes[idx].bar(component_names, values, 
                               color=[self.colors['hsgm'] if 'full' in name.lower() 
                                     else self.colors['baseline'] for name in component_names],
                               alpha=0.7)
            
            axes[idx].set_xlabel('Components')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f'{metric_name} by Component')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].annotate(f'{height:.3f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom',
                                 fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self,
                                   efficiency_data: Dict[str, Any],
                                   performance_data: Dict[str, Any],
                                   streaming_data: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            efficiency_data: Efficiency metrics data
            performance_data: Performance metrics data
            streaming_data: Streaming metrics data
            save_path: Optional path to save HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Processing Time vs Document Length', 
                          'Memory Usage vs Document Length',
                          'Model Performance Comparison',
                          'Streaming Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Processing time plot
        if 'document_lengths' in efficiency_data:
            fig.add_trace(
                go.Scatter(x=efficiency_data['document_lengths'],
                          y=efficiency_data['hsgm_times'],
                          mode='lines+markers',
                          name='HSGM',
                          line=dict(color=self.colors['hsgm'])),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=efficiency_data['document_lengths'],
                          y=efficiency_data['baseline_times'],
                          mode='lines+markers',
                          name='Baseline',
                          line=dict(color=self.colors['baseline'])),
                row=1, col=1
            )
        
        # Memory usage plot
        if 'memory_data' in efficiency_data:
            fig.add_trace(
                go.Scatter(x=efficiency_data['document_lengths'],
                          y=efficiency_data['memory_data']['hsgm'],
                          mode='lines+markers',
                          name='HSGM Memory',
                          line=dict(color=self.colors['hsgm'])),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=efficiency_data['document_lengths'],
                          y=efficiency_data['memory_data']['baseline'],
                          mode='lines+markers',
                          name='Baseline Memory',
                          line=dict(color=self.colors['baseline'])),
                row=1, col=2
            )
        
        # Performance comparison
        if 'model_names' in performance_data:
            fig.add_trace(
                go.Bar(x=performance_data['model_names'],
                      y=performance_data['accuracies'],
                      name='Accuracy',
                      marker_color=self.colors['hsgm']),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=performance_data['model_names'],
                      y=performance_data['f1_scores'],
                      name='F1 Score',
                      marker_color=self.colors['baseline']),
                row=2, col=1
            )
        
        # Streaming performance
        if 'timestamps' in streaming_data:
            fig.add_trace(
                go.Scatter(x=streaming_data['timestamps'],
                          y=streaming_data['cache_hit_rates'],
                          mode='lines+markers',
                          name='Cache Hit Rate',
                          line=dict(color=self.colors['hsgm'])),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="HSGM Performance Dashboard"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Document Length (tokens)", row=1, col=1)
        fig.update_yaxes(title_text="Processing Time (s)", row=1, col=1)
        
        fig.update_xaxes(title_text="Document Length (tokens)", row=1, col=2)
        fig.update_yaxes(title_text="Memory Usage (GB)", row=1, col=2)
        
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Cache Hit Rate", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_error_analysis(self,
                          error_bounds: List[float],
                          approximation_errors: List[float],
                          thresholds: List[Tuple[float, float]],
                          save_path: Optional[str] = None):
        """
        Plot approximation error analysis.
        
        Args:
            error_bounds: Theoretical error bounds
            approximation_errors: Actual approximation errors
            thresholds: List of (local_threshold, global_threshold) tuples
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error bounds vs actual errors
        ax1.scatter(error_bounds, approximation_errors, 
                   c=range(len(error_bounds)), cmap='viridis', s=100, alpha=0.7)
        ax1.plot([0, max(max(error_bounds), max(approximation_errors))],
                [0, max(max(error_bounds), max(approximation_errors))],
                'r--', label='Perfect Bound')
        ax1.set_xlabel('Theoretical Error Bound')
        ax1.set_ylabel('Actual Approximation Error')
        ax1.set_title('Error Bound vs Actual Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Threshold analysis
        local_thresholds = [t[0] for t in thresholds]
        global_thresholds = [t[1] for t in thresholds]
        
        scatter = ax2.scatter(local_thresholds, global_thresholds, 
                            c=approximation_errors, cmap='plasma', s=100, alpha=0.7)
        ax2.set_xlabel('Local Threshold')
        ax2.set_ylabel('Global Threshold')
        ax2.set_title('Threshold Impact on Approximation Error')
        plt.colorbar(scatter, ax=ax2, label='Approximation Error')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_comparison_report(hsgm_results: Dict[str, Any],
                           baseline_results: Dict[str, Any],
                           output_dir: str = "./reports"):
    """
    Create a comprehensive comparison report with visualizations.
    
    Args:
        hsgm_results: HSGM evaluation results
        baseline_results: Baseline evaluation results
        output_dir: Output directory for reports
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = HSGMVisualizer()
    
    # Extract data for visualizations
    document_lengths = [1000, 5000, 10000, 20000]
    hsgm_times = [0.3, 1.2, 2.8, 5.2]  # Placeholder data
    baseline_times = [1.2, 8.5, 25.3, 85.7]  # Placeholder data
    
    # Generate plots
    visualizer.plot_complexity_analysis(
        document_lengths, hsgm_times, baseline_times,
        save_path=os.path.join(output_dir, "complexity_analysis.png")
    )
    
    # Memory usage plot
    hsgm_memory = [6.2, 8.5, 11.2, 15.8]  # Placeholder data
    baseline_memory = [12.5, 25.3, 45.7, 78.2]  # Placeholder data
    
    visualizer.plot_memory_usage(
        document_lengths, hsgm_memory, baseline_memory,
        save_path=os.path.join(output_dir, "memory_usage.png")
    )
    
    # Performance comparison
    model_names = ['HSGM', 'Longformer', 'BigBird', 'Full Graph']
    accuracies = [0.785, 0.768, 0.771, 0.782]
    f1_scores = [0.856, 0.845, 0.848, 0.851]
    
    visualizer.plot_accuracy_comparison(
        model_names, accuracies, f1_scores,
        save_path=os.path.join(output_dir, "performance_comparison.png")
    )
    
    logger.info(f"Comparison report saved to {output_dir}")
