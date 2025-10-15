"""
HSGM Experiments Package

Contains all experimental code, training, evaluation, baselines, and visualization.
"""

from .train import HSGMTrainer
from .evaluation import Evaluator, EvaluationMetrics
from .baselines import BaselineModels
from .visualization import HSGMVisualizer
from .data_loader import (
    HSGMDataset,
    DocumentAMRDataset,
    OntoNotesSRLDataset,
    LegalECHRDataset,
    NarrativeQADataset,
    GovReportDataset,
    create_dataloaders
)

__all__ = [
    "HSGMTrainer",
    "Evaluator",
    "EvaluationMetrics",
    "BaselineModels",
    "HSGMVisualizer",
    "HSGMDataset",
    "DocumentAMRDataset",
    "OntoNotesSRLDataset",
    "LegalECHRDataset",
    "NarrativeQADataset",
    "GovReportDataset",
    "create_dataloaders"
]

