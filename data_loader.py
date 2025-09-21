"""
Data loading utilities for HSGM framework.
"""
import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset, load_dataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class HSGMDataset(Dataset):
    """Base dataset class for HSGM."""
    
    def __init__(self, 
                 documents: List[str],
                 labels: Optional[List[int]] = None,
                 target_sequences: Optional[List[str]] = None,
                 metadata: Optional[List[Dict]] = None):
        """
        Initialize HSGM Dataset.
        
        Args:
            documents: List of document texts
            labels: Optional list of labels for classification
            target_sequences: Optional target sequences for generation
            metadata: Optional metadata for each document
        """
        self.documents = documents
        self.labels = labels
        self.target_sequences = target_sequences
        self.metadata = metadata or [{}] * len(documents)
        
        assert len(self.documents) == len(self.metadata), "Documents and metadata must have same length"
        if self.labels is not None:
            assert len(self.documents) == len(self.labels), "Documents and labels must have same length"
        if self.target_sequences is not None:
            assert len(self.documents) == len(self.target_sequences), "Documents and target sequences must have same length"
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        item = {
            "documents": self.documents[idx],
            "metadata": self.metadata[idx]
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.target_sequences is not None:
            item["target_sequences"] = self.target_sequences[idx]
        
        return item

class DocumentAMRDataset(HSGMDataset):
    """Dataset for Document-AMR parsing task."""
    
    def __init__(self, 
                 data_path: str,
                 split: str = "train",
                 max_docs: Optional[int] = None):
        """
        Initialize Document-AMR dataset.
        
        Args:
            data_path: Path to dataset
            split: Dataset split ("train", "val", "test")
            max_docs: Maximum number of documents to load
        """
        self.data_path = data_path
        self.split = split
        
        # Load data
        documents, labels, metadata = self._load_data()
        
        # Limit number of documents if specified
        if max_docs is not None:
            documents = documents[:max_docs]
            labels = labels[:max_docs] if labels else None
            metadata = metadata[:max_docs]
        
        super().__init__(documents, labels, metadata=metadata)
    
    def _load_data(self) -> Tuple[List[str], List[Dict], List[Dict]]:
        """Load Document-AMR data."""
        documents = []
        amr_graphs = []  # Labels for AMR parsing
        metadata = []
        
        # Try to load from different formats
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                documents.append(item["text"])
                amr_graphs.append(item.get("amr", {}))
                metadata.append({
                    "doc_id": item.get("id", ""),
                    "length": len(item["text"]),
                    "source": "document_amr"
                })
        
        else:
            # Create dummy data if file doesn't exist
            logger.warning(f"Data file {data_file} not found. Creating dummy data.")
            for i in range(100):
                documents.append(f"This is a sample document {i} for AMR parsing. " * 50)
                amr_graphs.append({"nodes": [], "edges": []})
                metadata.append({
                    "doc_id": f"doc_{i}",
                    "length": len(documents[-1]),
                    "source": "document_amr_dummy"
                })
        
        return documents, amr_graphs, metadata

class OntoNotesSRLDataset(HSGMDataset):
    """Dataset for OntoNotes Semantic Role Labeling."""
    
    def __init__(self,
                 data_path: str,
                 split: str = "train",
                 max_segments: Optional[int] = None):
        """
        Initialize OntoNotes-SRL dataset.
        
        Args:
            data_path: Path to dataset
            split: Dataset split
            max_segments: Maximum number of segments to load
        """
        self.data_path = data_path
        self.split = split
        
        documents, labels, metadata = self._load_data()
        
        if max_segments is not None:
            documents = documents[:max_segments]
            labels = labels[:max_segments] if labels else None
            metadata = metadata[:max_segments]
        
        super().__init__(documents, labels, metadata=metadata)
    
    def _load_data(self) -> Tuple[List[str], List[Dict], List[Dict]]:
        """Load OntoNotes-SRL data."""
        documents = []
        srl_labels = []
        metadata = []
        
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                documents.append(item["text"])
                srl_labels.append(item.get("srl", []))
                metadata.append({
                    "segment_id": item.get("id", ""),
                    "length": len(item["text"]),
                    "source": "onto_notes_srl"
                })
        
        else:
            # Create dummy data
            logger.warning(f"Data file {data_file} not found. Creating dummy data.")
            for i in range(500):
                documents.append(f"Segment {i} contains semantic role information. " * 30)
                srl_labels.append([{"predicate": "contains", "arguments": []}])
                metadata.append({
                    "segment_id": f"seg_{i}",
                    "length": len(documents[-1]),
                    "source": "onto_notes_srl_dummy"
                })
        
        return documents, srl_labels, metadata

class LegalECHRDataset(HSGMDataset):
    """Dataset for Legal ECHR event extraction."""
    
    def __init__(self,
                 data_path: str,
                 split: str = "train",
                 max_docs: Optional[int] = None):
        """
        Initialize Legal-ECHR dataset.
        
        Args:
            data_path: Path to dataset
            split: Dataset split
            max_docs: Maximum number of documents to load
        """
        self.data_path = data_path
        self.split = split
        
        documents, labels, metadata = self._load_data()
        
        if max_docs is not None:
            documents = documents[:max_docs]
            labels = labels[:max_docs] if labels else None
            metadata = metadata[:max_docs]
        
        super().__init__(documents, labels, metadata=metadata)
    
    def _load_data(self) -> Tuple[List[str], List[Dict], List[Dict]]:
        """Load Legal-ECHR data."""
        documents = []
        events = []
        metadata = []
        
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                documents.append(item["text"])
                events.append(item.get("events", []))
                metadata.append({
                    "case_id": item.get("case_id", ""),
                    "length": len(item["text"]),
                    "source": "legal_eghr"
                })
        
        else:
            # Create dummy data
            logger.warning(f"Data file {data_file} not found. Creating dummy data.")
            for i in range(200):
                documents.append(f"Legal case {i} involves various events and decisions. " * 40)
                events.append([{"type": "decision", "text": "case decided"}])
                metadata.append({
                    "case_id": f"case_{i}",
                    "length": len(documents[-1]),
                    "source": "legal_eghr_dummy"
                })
        
        return documents, events, metadata

class NarrativeQADataset(HSGMDataset):
    """Dataset for NarrativeQA question answering."""
    
    def __init__(self,
                 data_path: str,
                 split: str = "train",
                 max_examples: Optional[int] = None):
        """
        Initialize NarrativeQA dataset.
        
        Args:
            data_path: Path to dataset
            split: Dataset split
            max_examples: Maximum number of examples to load
        """
        self.data_path = data_path
        self.split = split
        
        documents, answers, metadata = self._load_data()
        
        if max_examples is not None:
            documents = documents[:max_examples]
            answers = answers[:max_examples] if answers else None
            metadata = metadata[:max_examples]
        
        super().__init__(documents, target_sequences=answers, metadata=metadata)
    
    def _load_data(self) -> Tuple[List[str], List[str], List[Dict]]:
        """Load NarrativeQA data."""
        documents = []
        answers = []
        metadata = []
        
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                documents.append(item["story"])
                answers.append(item.get("answers", [""])[0] if item.get("answers") else "")
                metadata.append({
                    "question": item.get("question", ""),
                    "story_id": item.get("story_id", ""),
                    "length": len(item["story"]),
                    "source": "narrative_qa"
                })
        
        else:
            # Create dummy data
            logger.warning(f"Data file {data_file} not found. Creating dummy data.")
            for i in range(300):
                documents.append(f"Narrative story {i} with a question to answer. " * 60)
                answers.append(f"Answer {i}")
                metadata.append({
                    "question": f"What happens in story {i}?",
                    "story_id": f"story_{i}",
                    "length": len(documents[-1]),
                    "source": "narrative_qa_dummy"
                })
        
        return documents, answers, metadata

class GovReportDataset(HSGMDataset):
    """Dataset for GovReport summarization."""
    
    def __init__(self,
                 data_path: str,
                 split: str = "train",
                 max_docs: Optional[int] = None):
        """
        Initialize GovReport dataset.
        
        Args:
            data_path: Path to dataset
            split: Dataset split
            max_docs: Maximum number of documents to load
        """
        self.data_path = data_path
        self.split = split
        
        documents, summaries, metadata = self._load_data()
        
        if max_docs is not None:
            documents = documents[:max_docs]
            summaries = summaries[:max_docs] if summaries else None
            metadata = metadata[:max_docs]
        
        super().__init__(documents, target_sequences=summaries, metadata=metadata)
    
    def _load_data(self) -> Tuple[List[str], List[str], List[Dict]]:
        """Load GovReport data."""
        documents = []
        summaries = []
        metadata = []
        
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                documents.append(item["report"])
                summaries.append(item.get("summary", ""))
                metadata.append({
                    "report_id": item.get("id", ""),
                    "length": len(item["report"]),
                    "source": "gov_report"
                })
        
        else:
            # Create dummy data
            logger.warning(f"Data file {data_file} not found. Creating dummy data.")
            for i in range(150):
                documents.append(f"Government report {i} with detailed analysis. " * 80)
                summaries.append(f"Summary of report {i}")
                metadata.append({
                    "report_id": f"report_{i}",
                    "length": len(documents[-1]),
                    "source": "gov_report_dummy"
                })
        
        return documents, summaries, metadata

def create_dataloaders(dataset_name: str,
                      config,
                      dataset_config,
                      batch_size: Optional[int] = None,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of the dataset
        config: HSGM configuration
        dataset_config: Dataset configuration
        batch_size: Batch size (uses config if None)
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.batch_size
    
    # Select dataset class
    dataset_classes = {
        "document_amr": DocumentAMRDataset,
        "onto_notes_srl": OntoNotesSRLDataset,
        "legal_eghr": LegalECHRDataset,
        "narrative_qa": NarrativeQADataset,
        "gov_report": GovReportDataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_class = dataset_classes[dataset_name]
    
    # Get data path
    data_paths = {
        "document_amr": dataset_config.document_amr_path,
        "onto_notes_srl": dataset_config.onto_notes_path,
        "legal_eghr": dataset_config.legal_eghr_path,
        "narrative_qa": dataset_config.narrative_qa_path,
        "gov_report": dataset_config.gov_report_path
    }
    
    data_path = data_paths[dataset_name]
    
    # Get max examples
    max_examples = {
        "document_amr": dataset_config.document_amr_max_docs,
        "onto_notes_srl": dataset_config.onto_notes_max_segments,
        "legal_eghr": dataset_config.legal_eghr_max_docs,
        "narrative_qa": None,
        "gov_report": None
    }
    
    max_examples = max_examples[dataset_name]
    
    # Create datasets
    train_dataset = dataset_class(data_path, "train", max_examples)
    val_dataset = dataset_class(data_path, "val", max_examples)
    test_dataset = dataset_class(data_path, "test", max_examples)
    
    # Create data loaders
    def collate_fn(batch):
        """Custom collate function for HSGM datasets."""
        documents = [item["documents"] for item in batch]
        metadata = [item["metadata"] for item in batch]
        
        result = {
            "documents": documents,
            "metadata": metadata
        }
        
        # Add labels if present
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            result["labels"] = labels
        
        # Add target sequences if present
        if "target_sequences" in batch[0]:
            target_sequences = [item["target_sequences"] for item in batch]
            result["target_sequences"] = target_sequences
        
        return result
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders for {dataset_name}")
    logger.info(f"Train: {len(train_dataset)} examples")
    logger.info(f"Val: {len(val_dataset)} examples")
    logger.info(f"Test: {len(test_dataset)} examples")
    
    return train_loader, val_loader, test_loader

def create_streaming_dataset(dataset_name: str,
                           dataset_config,
                           chunk_size: int = 256,
                           interval_ms: int = 100) -> List[Dict]:
    """
    Create a streaming dataset for simulating real-time document arrival.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        chunk_size: Size of each chunk
        interval_ms: Interval between chunks in milliseconds
        
    Returns:
        List of streaming chunks
    """
    # Load full dataset
    _, _, test_loader = create_dataloaders(dataset_name, None, dataset_config)
    
    streaming_chunks = []
    
    for batch in test_loader:
        documents = batch["documents"]
        
        for doc in documents:
            # Split document into chunks
            tokens = doc.split()
            
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = " ".join(chunk_tokens)
                
                chunk_data = {
                    "text": chunk_text,
                    "chunk_id": len(streaming_chunks),
                    "document_id": hash(doc) % 10000,
                    "timestamp": len(streaming_chunks) * interval_ms,
                    "position": i,
                    "total_length": len(tokens)
                }
                
                streaming_chunks.append(chunk_data)
    
    logger.info(f"Created streaming dataset with {len(streaming_chunks)} chunks")
    return streaming_chunks
