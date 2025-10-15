"""
Baseline models for comparison with HSGM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import numpy as np
# Optional imports
try:
    from transformers import (
        AutoTokenizer, AutoModel, 
        LongformerModel, LongformerTokenizer,
        BigBirdModel, BigBirdTokenizer,
        T5ForConditionalGeneration, T5Tokenizer,
        ReformerModel, ReformerTokenizer
    )
except ImportError:
    AutoTokenizer = AutoModel = None
    LongformerModel = LongformerTokenizer = None
    BigBirdModel = BigBirdTokenizer = None
    T5ForConditionalGeneration = T5Tokenizer = None
    ReformerModel = ReformerTokenizer = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
import logging

logger = logging.getLogger(__name__)

class BaselineModel(nn.Module):
    """Base class for baseline models."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

class FullGraphBaseline(BaselineModel):
    """Full Graph baseline - constructs a single global semantic graph."""
    
    def __init__(self, hidden_dim: int = 768, device: str = "cuda"):
        super().__init__("full_graph", device)
        
        self.hidden_dim = hidden_dim
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.encoder = AutoModel.from_pretrained("roberta-base").to(device)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(3)
        ])
        
        self.dropout = nn.Dropout(0.1)
    
    def build_global_graph(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build global graph for entire document."""
        # Encode tokens
        text = " ".join(tokens)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[0, 1:-1, :]  # Remove [CLS] and [SEP]
        
        # Build adjacency matrix based on similarity
        num_tokens = embeddings.size(0)
        adj_matrix = torch.zeros(num_tokens, num_tokens)
        
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                similarity = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                ).item()
                
                if similarity > 0.2:  # Threshold
                    adj_matrix[i, j] = similarity
                    adj_matrix[j, i] = similarity
        
        return embeddings, adj_matrix
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for full graph baseline."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            tokens = doc.split()
            
            # Build global graph
            embeddings, adj_matrix = self.build_global_graph(tokens)
            
            # Apply GNN layers
            h = embeddings
            for layer in self.gnn_layers:
                h = layer(torch.mm(adj_matrix, h))
                h = F.relu(h)
                h = self.dropout(h)
            
            # Aggregate
            result = torch.mean(h, dim=0)
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class SlidingWindowGraphBaseline(BaselineModel):
    """Sliding Window Graph baseline."""
    
    def __init__(self, 
                 window_size: int = 256,
                 overlap: int = 128,
                 hidden_dim: int = 768,
                 device: str = "cuda"):
        super().__init__("sliding_window_graph", device)
        
        self.window_size = window_size
        self.overlap = overlap
        self.hidden_dim = hidden_dim
        
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.encoder = AutoModel.from_pretrained("roberta-base").to(device)
        
        # Local graph layers
        self.local_gnn = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(2)
        ])
        
        # Global aggregation
        self.global_aggregator = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def create_windows(self, tokens: List[str]) -> List[List[str]]:
        """Create sliding windows."""
        windows = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.window_size, len(tokens))
            window = tokens[start:end]
            windows.append(window)
            
            if end == len(tokens):
                break
            start = end - self.overlap
        
        return windows
    
    def process_window(self, window_tokens: List[str]) -> torch.Tensor:
        """Process a single window."""
        text = " ".join(window_tokens)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[0, 1:-1, :]
        
        # Build local graph
        num_tokens = embeddings.size(0)
        adj_matrix = torch.eye(num_tokens)
        
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                similarity = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                ).item()
                
                if similarity > 0.3:
                    adj_matrix[i, j] = similarity
                    adj_matrix[j, i] = similarity
        
        # Apply local GNN
        h = embeddings
        for layer in self.local_gnn:
            h = layer(torch.mm(adj_matrix, h))
            h = F.relu(h)
            h = self.dropout(h)
        
        # Aggregate window
        return torch.mean(h, dim=0)
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for sliding window graph."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            tokens = doc.split()
            windows = self.create_windows(tokens)
            
            # Process each window
            window_embeddings = []
            for window in windows:
                window_emb = self.process_window(window)
                window_embeddings.append(window_emb)
            
            if window_embeddings:
                # Aggregate windows
                window_tensor = torch.stack(window_embeddings)
                global_emb = self.global_aggregator(window_tensor)
                result = torch.mean(global_emb, dim=0)
            else:
                result = torch.zeros(self.hidden_dim)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class LongformerBaseline(BaselineModel):
    """Longformer baseline model."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("longformer", device)
        
        self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.hidden_dim = self.model.config.hidden_size
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for Longformer."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            inputs = self.tokenizer(
                doc, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token
                result = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class BigBirdBaseline(BaselineModel):
    """BigBird baseline model."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("bigbird", device)
        
        self.model = BigBirdModel.from_pretrained("google/bigbird-roberta-base").to(device)
        self.tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        self.hidden_dim = self.model.config.hidden_size
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for BigBird."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            inputs = self.tokenizer(
                doc,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                result = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class LongT5Baseline(BaselineModel):
    """LongT5 baseline model."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("longt5", device)
        
        self.model = T5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base").to(device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/long-t5-tglobal-base")
        self.hidden_dim = self.model.config.d_model
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for LongT5."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            inputs = self.tokenizer(
                doc,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.encoder(**inputs)
                # Use mean pooling
                result = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class HierarchicalTransformerBaseline(BaselineModel):
    """Hierarchical Transformer baseline."""
    
    def __init__(self, hidden_dim: int = 768, device: str = "cuda"):
        super().__init__("hierarchical_transformer", device)
        
        self.hidden_dim = hidden_dim
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.sentence_encoder = AutoModel.from_pretrained("roberta-base").to(device)
        
        # Document-level transformer
        self.doc_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for hierarchical transformer."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            # Split into sentences (simplified)
            sentences = doc.split('. ')
            
            # Encode each sentence
            sentence_embeddings = []
            for sentence in sentences:
                if sentence.strip():
                    inputs = self.tokenizer(
                        sentence, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.sentence_encoder(**inputs)
                        sentence_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
                        sentence_embeddings.append(sentence_emb)
            
            if sentence_embeddings:
                # Document-level encoding
                sentence_tensor = torch.stack(sentence_embeddings).unsqueeze(0)
                doc_output = self.doc_transformer(sentence_tensor)
                result = torch.mean(doc_output, dim=1).squeeze(0)
            else:
                result = torch.zeros(self.hidden_dim)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class GraphTransformerBaseline(BaselineModel):
    """Graph Transformer baseline."""
    
    def __init__(self, hidden_dim: int = 768, device: str = "cuda"):
        super().__init__("graph_transformer", device)
        
        self.hidden_dim = hidden_dim
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.encoder = AutoModel.from_pretrained("roberta-base").to(device)
        
        # Graph attention layers
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for graph transformer."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            tokens = doc.split()
            text = " ".join(tokens)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings = outputs.last_hidden_state[0, 1:-1, :]  # Remove [CLS] and [SEP]
            
            # Graph attention
            embeddings_expanded = embeddings.unsqueeze(0)
            attended, _ = self.graph_attention(
                embeddings_expanded, 
                embeddings_expanded, 
                embeddings_expanded
            )
            
            # Residual connection and layer norm
            attended = attended.squeeze(0)
            attended = self.layer_norm(attended + embeddings)
            attended = self.dropout(attended)
            
            # Aggregate
            result = torch.mean(attended, dim=0)
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class ReformerBaseline(BaselineModel):
    """Reformer baseline model."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("reformer", device)
        
        # Use a simpler transformer for Reformer-like behavior
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.encoder = AutoModel.from_pretrained("roberta-base").to(device)
        
        # Efficient attention simulation
        self.efficient_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        self.hidden_dim = 768
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for Reformer."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings = outputs.last_hidden_state
                
                # Efficient attention
                attended, _ = self.efficient_attention(
                    embeddings, embeddings, embeddings
                )
                
                result = torch.mean(attended, dim=1).squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class RetrievalAugmentedBaseline(BaselineModel):
    """Base class for retrieval-augmented baselines."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self.retrieval_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        self.top_k = 5
    
    def retrieve_documents(self, query: str, corpus: List[str]) -> List[Tuple[str, float]]:
        """Retrieve top-k most similar documents."""
        query_embedding = self.retrieval_model.encode(query)
        corpus_embeddings = self.retrieval_model.encode(corpus)
        
        similarities = np.dot(corpus_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        return [(corpus[i], similarities[i]) for i in top_indices]

class BM25T5Baseline(RetrievalAugmentedBaseline):
    """BM25 + T5 baseline."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("bm25_t5", device)
        
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.hidden_dim = 768
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for BM25 + T5."""
        # Simplified implementation - in practice would use BM25 for retrieval
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            # Placeholder: encode document
            inputs = self.tokenizer(
                f"summarize: {doc}",
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generator.encoder(**inputs)
                result = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class FiDBaseline(RetrievalAugmentedBaseline):
    """Fusion-in-Decoder baseline."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("fid", device)
        
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.hidden_dim = 768
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for FiD."""
        # Simplified FiD implementation
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            # Split into passages
            passages = [doc[i:i+512] for i in range(0, len(doc), 256)]
            
            # Encode passages
            passage_embeddings = []
            for passage in passages[:5]:  # Limit to 5 passages
                inputs = self.tokenizer(
                    passage,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.generator.encoder(**inputs)
                    passage_emb = torch.mean(outputs.last_hidden_state, dim=1)
                    passage_embeddings.append(passage_emb)
            
            if passage_embeddings:
                # Concatenate and aggregate
                combined = torch.cat(passage_embeddings, dim=1)
                result = torch.mean(combined, dim=1).squeeze(0)
            else:
                result = torch.zeros(self.hidden_dim)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class SGPTBaseline(RetrievalAugmentedBaseline):
    """SGPT baseline."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("sgpt", device)
        
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
        self.hidden_dim = 384  # MiniLM dimension
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for SGPT."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            embedding = self.model.encode(doc, convert_to_tensor=True)
            results.append(embedding)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class RAGBaseline(RetrievalAugmentedBaseline):
    """RAG baseline."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("rag", device)
        
        from transformers import BartForConditionalGeneration, BartTokenizer
        self.generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.hidden_dim = 768
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for RAG."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            inputs = self.tokenizer(
                doc,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generator.model.encoder(**inputs)
                result = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class REPLUGBaseline(RetrievalAugmentedBaseline):
    """REPLUG baseline."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("replug", device)
        
        self.encoder = AutoModel.from_pretrained("roberta-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.hidden_dim = 768
    
    def forward(self, documents: Union[str, List[str]]):
        """Forward pass for REPLUG."""
        if isinstance(documents, str):
            documents = [documents]
        
        results = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                result = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
            results.append(result)
        
        return torch.stack(results) if results else torch.zeros(1, self.hidden_dim)

class BaselineModels:
    """Container for all baseline models."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize baseline models.
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self.models = {}
        
        # Transformer-based baselines
        self.models["full_graph"] = FullGraphBaseline(device=device)
        self.models["sliding_window_graph"] = SlidingWindowGraphBaseline(device=device)
        self.models["longformer"] = LongformerBaseline(device=device)
        self.models["bigbird"] = BigBirdBaseline(device=device)
        self.models["longt5"] = LongT5Baseline(device=device)
        self.models["hierarchical_transformer"] = HierarchicalTransformerBaseline(device=device)
        self.models["graph_transformer"] = GraphTransformerBaseline(device=device)
        self.models["reformer"] = ReformerBaseline(device=device)
        
        # Retrieval-augmented baselines
        self.models["bm25_t5"] = BM25T5Baseline(device=device)
        self.models["fid"] = FiDBaseline(device=device)
        self.models["sgpt"] = SGPTBaseline(device=device)
        self.models["rag"] = RAGBaseline(device=device)
        self.models["replug"] = REPLUGBaseline(device=device)
    
    def get_model(self, model_name: str) -> BaselineModel:
        """Get a specific baseline model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown baseline model: {model_name}")
        return self.models[model_name]
    
    def get_all_models(self) -> Dict[str, BaselineModel]:
        """Get all baseline models."""
        return self.models
    
    def evaluate_baseline(self, 
                         model_name: str,
                         documents: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate a specific baseline model.
        
        Args:
            model_name: Name of the baseline model
            documents: Input documents
            
        Returns:
            Dictionary with model output and metrics
        """
        model = self.get_model(model_name)
        
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            output = model(documents)
        
        processing_time = time.time() - start_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            memory_usage = 0.0
        
        # Throughput
        num_docs = len(documents) if isinstance(documents, list) else 1
        throughput = num_docs / processing_time
        
        return {
            "model_name": model_name,
            "output": output,
            "processing_time": processing_time,
            "memory_usage": memory_usage,
            "throughput": throughput,
            "output_shape": output.shape if isinstance(output, torch.Tensor) else "N/A"
        }
    
    def compare_all_baselines(self, 
                            documents: Union[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Compare all baseline models.
        
        Args:
            documents: Input documents
            
        Returns:
            Dictionary with results for all baselines
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.evaluate_baseline(model_name, documents)
                results[model_name] = result
                logger.info(f"Evaluated {model_name}: {result['processing_time']:.4f}s, {result['memory_usage']:.4f}GB")
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "processing_time": float('inf'),
                    "memory_usage": float('inf'),
                    "throughput": 0.0
                }
        
        return results
    
    def get_efficiency_ranking(self, results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Rank models by efficiency (processing time).
        
        Args:
            results: Results from compare_all_baselines
            
        Returns:
            List of (model_name, processing_time) tuples sorted by efficiency
        """
        valid_results = [
            (name, result["processing_time"]) 
            for name, result in results.items() 
            if "error" not in result
        ]
        
        return sorted(valid_results, key=lambda x: x[1])
    
    def get_memory_ranking(self, results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Rank models by memory usage.
        
        Args:
            results: Results from compare_all_baselines
            
        Returns:
            List of (model_name, memory_usage) tuples sorted by memory efficiency
        """
        valid_results = [
            (name, result["memory_usage"]) 
            for name, result in results.items() 
            if "error" not in result
        ]
        
        return sorted(valid_results, key=lambda x: x[1])
    
    def get_throughput_ranking(self, results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Rank models by throughput.
        
        Args:
            results: Results from compare_all_baselines
            
        Returns:
            List of (model_name, throughput) tuples sorted by throughput
        """
        valid_results = [
            (name, result["throughput"]) 
            for name, result in results.items() 
            if "error" not in result
        ]
        
        return sorted(valid_results, key=lambda x: x[1], reverse=True)
