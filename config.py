"""
Configuration settings for HSGM implementation.
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class HSGMConfig:
    """Configuration class for HSGM model."""
    
    # Model architecture
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    
    # HSGM specific parameters
    segment_size: int = 256
    local_threshold: float = 0.2
    global_threshold: float = 0.1
    top_k_retrieval: int = 5
    
    # Training parameters
    learning_rate: float = 3e-5
    batch_size: int = 8
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000
    
    # Data parameters
    max_length: int = 4096
    tokenizer_name: str = "roberta-base"
    model_name: str = "roberta-base"
    
    # Evaluation parameters
    num_workers: int = 4
    seed: int = 42
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hsgm"
    log_interval: int = 100
    
    # Device
    device: str = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

@dataclass
class DatasetConfig:
    """Configuration for specific datasets."""
    
    # Document-AMR
    document_amr_path: str = "./data/document_amr"
    document_amr_max_docs: int = 1000
    
    # OntoNotes-SRL
    onto_notes_path: str = "./data/onto_notes"
    onto_notes_max_segments: int = 5000
    
    # Legal-ECHR
    legal_eghr_path: str = "./data/legal_eghr"
    legal_eghr_max_docs: int = 1000
    
    # Downstream tasks
    narrative_qa_path: str = "./data/narrative_qa"
    gov_report_path: str = "./data/gov_report"
    
    # Streaming simulation
    streaming_chunk_size: int = 256
    streaming_interval: int = 100  # ms

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Ablation studies
    ablation_components: List[str] = None
    hyperparameter_ranges: Dict[str, List[Any]] = None
    
    # Scalability tests
    document_lengths: List[int] = None
    memory_profiling: bool = True
    
    # Evaluation settings
    eval_batch_size: int = 16
    num_eval_runs: int = 5
    confidence_interval: float = 0.95
    
    def __post_init__(self):
        if self.ablation_components is None:
            self.ablation_components = [
                "local_graph", "hierarchical_memory", 
                "cross_attention", "contrastive_learning"
            ]
        
        if self.hyperparameter_ranges is None:
            self.hyperparameter_ranges = {
                "local_threshold": [0.1, 0.15, 0.2, 0.25, 0.3],
                "global_threshold": [0.05, 0.1, 0.15, 0.2],
                "segment_size": [128, 256, 512, 1024],
                "top_k_retrieval": [3, 5, 7, 10]
            }
        
        if self.document_lengths is None:
            self.document_lengths = [1000, 5000, 10000, 20000]

# Default configurations
DEFAULT_HSGM_CONFIG = HSGMConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
