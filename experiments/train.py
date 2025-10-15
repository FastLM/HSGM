"""
Training script for HSGM model.
"""
import os
import sys
import argparse
import logging
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HSGMConfig, DatasetConfig, ExperimentConfig
from hsgm import HSGMModel, HSGMForClassification, HSGMForGeneration
from experiments.data_loader import HSGMDataset, create_dataloaders
from experiments.evaluation import Evaluator
from experiments.baselines import BaselineModels

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

class HSGMTrainer:
    """Trainer class for HSGM model."""
    
    def __init__(self, 
                 config: HSGMConfig,
                 dataset_config: DatasetConfig,
                 experiment_config: ExperimentConfig):
        """
        Initialize HSGM Trainer.
        
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
        
        # Initialize model
        self.base_model = HSGMModel(
            config=config,
            hidden_dim=config.hidden_dim,
            segment_size=config.segment_size,
            local_threshold=config.local_threshold,
            global_threshold=config.global_threshold,
            top_k_retrieval=config.top_k_retrieval,
            device=str(self.device)
        ).to(self.device)
        
        # Initialize task-specific model
        self.task_type = "classification"  # Can be "classification", "generation", "semantic_parsing"
        if self.task_type == "classification":
            self.model = HSGMForClassification(
                base_model=self.base_model,
                num_classes=3,  # Example: positive, negative, neutral
                dropout=config.dropout
            ).to(self.device)
        elif self.task_type == "generation":
            self.model = HSGMForGeneration(
                base_model=self.base_model,
                vocab_size=30000,  # Example vocabulary size
                hidden_dim=config.hidden_dim
            ).to(self.device)
        else:
            self.model = self.base_model
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize evaluator
        self.evaluator = Evaluator(self.device)
        
        # Initialize baselines for comparison
        self.baselines = BaselineModels(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.training_history = []
        
    def setup_wandb(self):
        """Set up Weights & Biases logging."""
        if self.config.use_wandb and wandb is not None:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "segment_size": self.config.segment_size,
                    "local_threshold": self.config.local_threshold,
                    "global_threshold": self.config.global_threshold,
                    "hidden_dim": self.config.hidden_dim
                }
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics for classification
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Extract batch data
            documents = batch["documents"]
            labels = batch.get("labels", None)
            
            # Forward pass
            if self.task_type == "classification":
                logits = self.model(documents)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Collect predictions for metrics
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            elif self.task_type == "generation":
                logits = self.model(documents, batch.get("target_sequences", None))
                # For generation, we'd need target sequences
                loss = torch.tensor(0.0, requires_grad=True)
                
            else:
                # Semantic parsing task
                output = self.model(documents)
                # Define loss based on specific task
                loss = torch.tensor(0.0, requires_grad=True)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}"
            })
            
            # Log to wandb
            if self.config.use_wandb and wandb is not None and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": self.current_epoch,
                    "batch": batch_idx
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        metrics = {
            "loss": avg_loss,
            "epoch": self.current_epoch
        }
        
        # Add task-specific metrics
        if self.task_type == "classification" and all_predictions:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average="weighted")
            precision, recall, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average="macro"
            )
            
            metrics.update({
                "accuracy": accuracy,
                "f1_weighted": f1,
                "f1_macro": f1_macro,
                "precision": precision,
                "recall": recall
            })
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                documents = batch["documents"]
                labels = batch.get("labels", None)
                
                if self.task_type == "classification":
                    logits = self.model(documents)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                else:
                    output = self.model(documents)
                    loss = torch.tensor(0.0)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = {"val_loss": avg_loss}
        
        if self.task_type == "classification" and all_predictions:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average="weighted")
            
            metrics.update({
                "val_accuracy": accuracy,
                "val_f1": f1
            })
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model."""
        logger.info("Starting training...")
        
        # Set up scheduler
        total_steps = len(train_loader) * self.config.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
            
            if self.config.use_wandb and wandb is not None:
                wandb.log(epoch_metrics)
            
            # Save best model
            if self.task_type == "classification":
                score = epoch_metrics.get("val_f1", 0.0)
            else:
                score = -epoch_metrics.get("val_loss", float('inf'))
            
            if score > self.best_score:
                self.best_score = score
                self.save_model(os.path.join(self.config.output_dir, "best_model.pt"))
            
            # Save checkpoint
            self.save_checkpoint(os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        logger.info(f"Training completed. Best score: {self.best_score:.4f}")
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_score": self.best_score,
            "config": self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]
        logger.info(f"Model loaded from {path}")
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_score": self.best_score,
            "training_history": self.training_history,
            "config": self.config
        }, path)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train HSGM model")
    parser.add_argument("--config", type=str, default="config.py", help="Config file path")
    parser.add_argument("--task", type=str, default="classification", 
                       choices=["classification", "generation", "semantic_parsing"],
                       help="Task type")
    parser.add_argument("--dataset", type=str, default="document_amr",
                       help="Dataset to use")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configurations
    config = HSGMConfig()
    dataset_config = DatasetConfig()
    experiment_config = ExperimentConfig()
    
    # Override task type
    config.task_type = args.task
    
    # Create trainer
    trainer = HSGMTrainer(config, dataset_config, experiment_config)
    
    # Set up wandb
    trainer.setup_wandb()
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        config=config,
        dataset_config=dataset_config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_model(args.resume)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    if config.use_wandb and wandb is not None:
        wandb.log({"test_metrics": test_metrics})
        wandb.finish()

if __name__ == "__main__":
    main()
