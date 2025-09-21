# HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Authors:** Dong Liu, Yanxuan Yu  
**Institutions:** Yale University, Columbia University

## Abstract

Semantic parsing of long documents remains challenging due to quadratic growth in pairwise composition and memory requirements. We introduce **Hierarchical Segment-Graph Memory (HSGM)**, a novel framework that decomposes an input of length N into M meaningful segments, constructs Local Semantic Graphs on each segment, and extracts compact summary nodes to form a Global Graph Memory. HSGM supports incremental updates—only newly arrived segments incur local graph construction and summary-node integration—while Hierarchical Query Processing locates relevant segments via top-K retrieval over summary nodes and then performs fine-grained reasoning within their local graphs.

## Key Features

- 🚀 **Efficient Processing**: Reduces complexity from O(N²) to O(Nk + (N/k)²)
- 🧠 **Hierarchical Memory**: Maintains both local and global semantic structures
- 📈 **Incremental Updates**: Supports streaming document processing
- 🔍 **Hierarchical Querying**: Top-K retrieval with fine-grained reasoning
- 📊 **Comprehensive Evaluation**: Extensive comparison with state-of-the-art baselines
- 🎯 **Multiple Tasks**: Supports AMR parsing, SRL, event extraction, QA, and summarization

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Additional Models

```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### Basic Usage

```python
from hsgm import HSGMModel
from config import HSGMConfig

# Initialize configuration
config = HSGMConfig()

# Create HSGM model
model = HSGMModel(
    config=config,
    hidden_dim=768,
    segment_size=256,
    local_threshold=0.2,
    global_threshold=0.1,
    top_k_retrieval=5,
    device="cuda"
)

# Process documents
documents = [
    "Your long document text here...",
    "Another document to process..."
]

output = model(documents)
print(f"Generated {len(output.summary_nodes)} summary nodes")
print(f"Processing time: {output.processing_time:.4f}s")
```

### Query Processing

```python
# Process a query
query = "What is the main topic?"
result = model.query(query)
print(f"Query result shape: {result.shape}")
```

### Incremental Updates

```python
# Add new documents incrementally
new_documents = ["New document content..."]
update_stats = model.incremental_update(new_documents)
print(f"Cache hit rate: {update_stats['hit_rate']:.3f}")
```

## Demo

Run the comprehensive demo to see HSGM in action:

```bash
python demo.py
```

This will demonstrate:
- Basic document processing
- Comparison with baseline models
- Streaming document processing
- Visualization capabilities
- Theoretical complexity analysis

## Training

### Train HSGM Model

```bash
python train.py --dataset document_amr --task classification --epochs 10
```

### Available Datasets

- `document_amr`: Document-level AMR parsing
- `onto_notes_srl`: Semantic Role Labeling
- `legal_eghr`: Legal event extraction
- `narrative_qa`: Question answering
- `gov_report`: Text summarization

### Training Options

```bash
python train.py \
    --dataset document_amr \
    --task classification \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --max_epochs 10 \
    --segment_size 256 \
    --local_threshold 0.2 \
    --global_threshold 0.1
```

## Experiments

### Run Comprehensive Experiments

```bash
python experiments.py \
    --dataset document_amr \
    --experiments main ablation scalability streaming \
    --output_dir ./results
```

### Available Experiments

- **main**: Compare HSGM with baseline models
- **ablation**: Component ablation study
- **scalability**: Performance across document lengths
- **streaming**: Incremental processing simulation
- **hyperparameter**: Sensitivity analysis

### Individual Experiment Types

```bash
# Main comparison experiment
python experiments.py --experiments main

# Ablation study
python experiments.py --experiments ablation

# Scalability analysis
python experiments.py --experiments scalability

# Streaming simulation
python experiments.py --experiments streaming
```

## Architecture

### HSGM Components

```
Input Document
     ↓
Document Segmentation
     ↓
Local Semantic Graph Construction
     ↓
Hierarchical Memory Building
     ↓
Global Graph Memory
     ↓
Query Processing (Hierarchical Retrieval + Local Reasoning)
```

### Key Components

1. **Document Segmenter**: Splits documents into coherent segments
2. **Local Semantic Graph**: Builds graphs within each segment
3. **Hierarchical Memory**: Maintains summary nodes and global structure
4. **Incremental Updater**: Handles streaming document updates
5. **Query Processor**: Performs hierarchical retrieval and reasoning

## Performance

### Theoretical Complexity

- **HSGM**: O(Nk + (N/k)²) where k is segment size
- **Full Graph**: O(N²)
- **Speedup**: Up to 59x on 20k-token documents

### Empirical Results

| Model | Accuracy | F1 Score | Speed | Memory |
|-------|----------|----------|-------|--------|
| HSGM | 78.5% | 85.6% | 380ms | 8.2GB |
| Longformer | 76.8% | 84.5% | 700ms | 6.8GB |
| BigBird | 77.1% | 84.8% | 650ms | 6.5GB |
| Full Graph | 78.2% | 85.1% | 1200ms | 12.5GB |

### Scalability Analysis

| Document Length | HSGM Time | Baseline Time | Speedup | Memory Reduction |
|----------------|-----------|---------------|---------|------------------|
| 1k tokens | 0.3s | 1.2s | 4.0x | 48% |
| 5k tokens | 1.2s | 8.5s | 7.1x | 65% |
| 10k tokens | 2.8s | 25.3s | 9.0x | 72% |
| 20k tokens | 5.2s | 85.7s | 16.5x | 78% |

## Configuration

### Main Configuration Options

```python
@dataclass
class HSGMConfig:
    # Model architecture
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_layers: int = 6
    
    # HSGM specific parameters
    segment_size: int = 256
    local_threshold: float = 0.2
    global_threshold: float = 0.1
    top_k_retrieval: int = 5
    
    # Training parameters
    learning_rate: float = 3e-5
    batch_size: int = 8
    max_epochs: int = 10
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
```

## Datasets

### Supported Datasets

1. **Document-AMR**: 500 training, 100 validation, 100 test documents
2. **OntoNotes-SRL**: 20k training, 2k validation, 2k test segments
3. **Legal-ECHR**: European Court of Human Rights cases
4. **NarrativeQA**: Long-form narrative question answering
5. **GovReport**: Government report summarization

### Data Format

```json
{
    "text": "Document content...",
    "labels": [0, 1, 2],
    "metadata": {
        "doc_id": "unique_id",
        "length": 1234,
        "source": "dataset_name"
    }
}
```

## Baseline Models

### Transformer-based Baselines

- **Full Graph**: Single global semantic graph
- **Sliding Window Graph**: Fixed-size windows with overlap
- **Longformer**: Sparse transformer with local+global attention
- **BigBird**: Sparse attention with random, window, global patterns
- **LongT5**: Encoder-decoder with local attention and global memory
- **Hierarchical Transformer**: Two-level segment and document attention
- **Graph Transformer**: Graph-structured data transformer
- **Reformer**: Efficient transformer with LSH attention

### Retrieval-augmented Baselines

- **BM25 + T5**: BM25 retrieval with T5 generation
- **FiD**: Fusion-in-Decoder with dense retrieval
- **SGPT**: SGPT-1.3B with semantic similarity retrieval
- **RAG**: DPR retriever with BART generator
- **REPLUG**: Retrieval-enhanced language models

## Evaluation Metrics

### Performance Metrics

- **Accuracy**: Classification accuracy
- **F1 Score**: Precision, Recall, F1 (macro/weighted)
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L for generation
- **BLEU**: BLEU score for text generation
- **Smatch F1**: For AMR parsing

### Efficiency Metrics

- **Processing Time**: End-to-end inference time
- **Memory Usage**: Peak GPU memory consumption
- **Throughput**: Documents processed per second
- **Cache Hit Rate**: Fraction of reused computations

## Visualization

### Available Plots

- **Complexity Analysis**: Processing time vs document length
- **Memory Usage**: Memory consumption comparison
- **Performance Comparison**: Accuracy and F1 scores
- **Efficiency Heatmap**: Model performance across document lengths
- **Streaming Performance**: Cache hit rates and memory over time
- **Ablation Study**: Component contribution analysis

### Generate Visualizations

```python
from visualization import HSGMVisualizer

visualizer = HSGMVisualizer()

# Plot complexity analysis
visualizer.plot_complexity_analysis(
    document_lengths, hsgm_times, baseline_times
)

# Plot memory usage
visualizer.plot_memory_usage(
    document_lengths, hsgm_memory, baseline_memory
)
```

## Theoretical Analysis

### Complexity Reduction

HSGM reduces worst-case complexity from O(N²) to O(Nk + (N/k)²):

- **Local Graph Construction**: O(Nk) for N tokens with segment size k
- **Global Memory Building**: O((N/k)²) for M = N/k segments
- **Optimal Segment Size**: k = √N achieves O(N^(3/2)) complexity

### Approximation Error Bounds

Given thresholds δℓ ≥ γℓ and δg ≥ γg, the approximation error is bounded by:

||Afull - AHSGM||F ≤ f(γℓ, γg) · ||Afull||F

where f(γℓ, γg) = √(2(1 - γℓ²)) + √(2(1 - γg²))

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/NoakLiu/HSGM.git
cd HSGM
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## Citation

If you use HSGM in your research, please cite our paper:

```bibtex
@inproceedings{liu2024hsgm,
  title={HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics},
  author={Liu, Dong and Yu, Yanxuan},
  booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  year={2024}
}
```

## License

This project is licensed under the MIT License

## Acknowledgments

- We thank the developers of PyTorch, Transformers, and other open-source libraries
- Special thanks to the research communities for the baseline models and datasets
- This work was supported by [Funding Information]

## Contact

- **Dong Liu**: dong.liu.dl2367@yale.edu
- **Yanxuan Yu**: yy3523@columbia.edu

For questions and support, please open an issue on GitHub or contact the authors directly.

---

**Note**: This implementation is based on the paper "HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics" by Dong Liu and Yanxuan Yu.
