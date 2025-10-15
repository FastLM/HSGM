# HSGM: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Authors:** Dong Liu, Yanxuan Yu  
**Institutions:** Yale University, Columbia University

## Abstract

Semantic parsing of long documents remains challenging due to quadratic growth in pairwise composition and memory requirements. We introduce **Hierarchical Segment-Graph Memory (HSGM)**, a novel framework that decomposes an input of length N into M meaningful segments, constructs Local Semantic Graphs on each segment, and extracts compact summary nodes to form a Global Graph Memory. HSGM reduces complexity from O(N²) to O(Nk + (N/k)²), achieving 2-4× speedup with custom CUDA kernels and 60%+ memory reduction. The framework supports incremental updates for streaming documents, hierarchical query processing via top-K retrieval, and multiple semantic tasks including AMR parsing, SRL, event extraction, QA, and summarization.

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

Run the comprehensive demo and test suite to see HSGM in action:

```bash
python run_demo.py
```



## Training

### Train HSGM Model

```bash
python experiments/train.py --dataset document_amr --task classification --epochs 10
```

### Available Datasets

- `document_amr`: Document-level AMR parsing
- `onto_notes_srl`: Semantic Role Labeling
- `legal_eghr`: Legal event extraction
- `narrative_qa`: Question answering
- `gov_report`: Text summarization

### Training Options

```bash
python experiments/train.py \
    --dataset document_amr \
    --task classification \
    --batch_size 8 \
    --learning_rate 3e-5
```

## Experiments

### Run Comprehensive Experiments

```bash
python experiments/experiments.py \
    --dataset document_amr \
    --experiments main ablation scalability streaming
```

## Architecture

### Pipeline

```
Input Document → Segmentation → Local Graphs → Hierarchical Memory → Query Processing
```

### Components & Acceleration

The framework consists of four core components: a Document Segmenter that splits documents into coherent segments, a Local Semantic Graph builder that constructs graphs within each segment, a Hierarchical Memory that maintains summary nodes with global structure, and a Query Processor for hierarchical retrieval with local reasoning. HSGM leverages custom CUDA kernels for GPU acceleration including vectorized pairwise similarity computation (O(Nk)), parallel adaptive thresholding with reduction, efficient sparse edge creation in COO format, multi-head cross-segment attention with shared memory optimization, and bitonic sort for fast top-K retrieval.

```python
from hsgm import get_hsgm_ops
ops = get_hsgm_ops(device='cuda')  # Auto-detects CUDA
similarities = ops.pairwise_similarity(embeddings)
```

## Performance

### Complexity

- **HSGM**: O(Nk + (N/k)²) with k ≪ N
- **Full Graph**: O(N²)
- **Optimal k**: √N achieves O(N^(3/2))

### Benchmark Results

| Model | Accuracy | F1 | Time (ms) | Memory (GB) |
|-------|----------|----|-----------| ------------|
| HSGM | 78.5% | 85.6% | 380 | 8.2 |
| Longformer | 76.8% | 84.5% | 700 | 6.8 |
| BigBird | 77.1% | 84.8% | 650 | 6.5 |
| Full Graph | 78.2% | 85.1% | 1200 | 12.5 |

### Scalability (20k tokens)

- **Speedup**: 16.5× faster than baseline
- **Memory**: 78% reduction vs Full Graph
- **Accuracy**: 95%+ of baseline performance

## Configuration

```python
from config import HSGMConfig

config = HSGMConfig(
    hidden_dim=768,
    segment_size=256,
    local_threshold=0.2,
    global_threshold=0.1,
    top_k_retrieval=5,
    learning_rate=3e-5
)
```

## Datasets

Supported: Document-AMR, OntoNotes-SRL, Legal-ECHR, NarrativeQA, GovReport

## Baselines

**Transformer**: Longformer, BigBird, LongT5, Reformer  
**Graph**: Full Graph, Sliding Window, Graph Transformer  
**Retrieval**: BM25+T5, FiD, SGPT, RAG, REPLUG



## Development

```bash
git clone https://github.com/FastLM/HSGM.git
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
@inproceedings{
liu2025hsgm,
title={{HSGM}: Hierarchical Segment-Graph Memory for Scalable Long-Text Semantics},
author={Liu, Dong and Yu, Yanxuan},
booktitle={The 14th Joint Conference on Lexical and Computational Semantics},
year={2025},
url={https://openreview.net/forum?id=NgbsJ3umjn}
}
```

