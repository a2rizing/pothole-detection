"""
Training pipeline for pothole detection models
Supports both classification and detection tasks with comprehensive logging and model optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import json
import time
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pothole_net import create_model, count_parameters
from utils.metrics import MetricsCalculator
from utils.dataloader import get_dataloaders

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially save weights
        
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning (classification + severity + optional depth)
    """
    
    def __init__(self, 
                 cls_weight=1.0, 
                 severity_weight=0.5, 
                 depth_weight=0.3,
                 use_focal=False):
        super(CombinedLoss, self).__init__()
        
        self.cls_weight = cls_weight
        self.severity_weight = severity_weight
        self.depth_weight = depth_weight
        
        # Classification loss
        if use_focal:
            self.cls_loss = FocalLoss(alpha=1, gamma=2)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        
        # Severity loss (MSE for regression)
        self.severity_loss = nn.MSELoss()
        
        # Depth loss (if applicable)
        self.depth_loss = nn.L1Loss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss
        
        Args:
            predictions (dict): Model predictions
            targets (dict): Ground truth targets
        
        Returns:
            dict: Loss components and total loss
        """
        losses = {}
        total_loss = 0
        
        # Classification loss
        if 'classification' in predictions and 'labels' in targets:
            cls_loss = self.cls_loss(predictions['classification'], targets['labels'])
            losses['classification'] = cls_loss
            total_loss += self.cls_weight * cls_loss
        
        # Severity loss
        if 'severity' in predictions and 'severity' in targets:
            sev_loss = self.severity_loss(
                predictions['severity'].squeeze(), 
                targets['severity'].float()
            )
            losses['severity'] = sev_loss
            total_loss += self.severity_weight * sev_loss
        
        # Depth loss (if available)
        if 'depth' in predictions and 'depth' in targets:
            depth_loss = self.depth_loss(predictions['depth'], targets['depth'])
            losses['depth'] = depth_loss
            total_loss += self.depth_weight * depth_loss
        
        losses['total'] = total_loss
        return losses


class Trainer:
    """
    Comprehensive trainer class for pothole detection models
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 test_loader=None,
                 config=None,
                 device='cuda'):
        """
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader (optional)
            config (dict): Training configuration
            device (str): Device to train on
        """
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Default configuration
        default_config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'loss_type': 'combined',
            'use_focal': False,
            'early_stopping_patience': 15,
            'save_best_only': True,
            'mixed_precision': True,
            'gradient_clip_val': 1.0,
            'log_interval': 10,
            'save_dir': './models',
            'experiment_name': f'pothole_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Setup metrics
        task = 'classification'  # Default, can be inferred from model
        self.metrics_calc = MetricsCalculator(task=task, device=device)
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config['mixed_precision'] else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Model info
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        
    def _setup_optimizer(self):
        """Setup optimizer based on configuration"""
        params = self.model.parameters()
        
        if self.config['optimizer'].lower() == 'adam':
            return optim.Adam(
                params, 
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'].lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'].lower() == 'sgd':
            return optim.SGD(
                params,
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config['scheduler'].lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['epochs']
            )
        elif self.config['scheduler'].lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config['scheduler'].lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return None
    
    def _setup_loss(self):
        """Setup loss function"""
        if self.config['loss_type'] == 'combined':
            return CombinedLoss(use_focal=self.config['use_focal'])
        elif self.config['loss_type'] == 'focal':
            return FocalLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _setup_logging(self):
        """Setup logging"""
        # Create save directory
        save_dir = Path(self.config['save_dir']) / self.config['experiment_name']
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        
        # Setup logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Save configuration
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        self.metrics_calc.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != 'image'}
            else:
                images, targets = batch
                images = images.to(self.device)
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in targets.items()}
                else:
                    targets = {'labels': targets.to(self.device)}
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['gradient_clip_val'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_val']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total']
                
                loss.backward()
                
                # Gradient clipping
                if self.config['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_val']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            if 'classification' in predictions and 'labels' in targets:
                self.metrics_calc.update_classification(
                    predictions['classification'], 
                    targets['labels'], 
                    loss.item()
                )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_calc.compute_metrics()
        
        return avg_loss, metrics
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        self.metrics_calc.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch.items() if k != 'image'}
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in targets.items()}
                    else:
                        targets = {'labels': targets.to(self.device)}
                
                # Forward pass
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total']
                
                # Update metrics
                total_loss += loss.item()
                
                if 'classification' in predictions and 'labels' in targets:
                    self.metrics_calc.update_classification(
                        predictions['classification'], 
                        targets['labels'], 
                        loss.item()
                    )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calc.compute_metrics()
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_metrics.get('accuracy', 0):.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth', epoch, val_loss, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self.save_model('final_model.pth', epoch, val_loss, val_metrics)
        
        # Plot training history
        self.plot_training_history()
        
        # Test evaluation if test loader provided
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            self.logger.info(f"Test metrics: {test_metrics}")
        
        return self.history
    
    def evaluate(self, data_loader):
        """Evaluate model on given data loader"""
        self.model.eval()
        self.metrics_calc.reset()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluation'):
                # Move data to device
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch.items() if k != 'image'}
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in targets.items()}
                    else:
                        targets = {'labels': targets.to(self.device)}
                
                # Forward pass
                predictions = self.model(images)
                
                if 'classification' in predictions and 'labels' in targets:
                    self.metrics_calc.update_classification(
                        predictions['classification'], 
                        targets['labels']
                    )
        
        return self.metrics_calc.compute_metrics()
    
    def save_model(self, filename, epoch, val_loss, val_metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.config,
            'history': self.history
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Model loaded from: {filepath}")
        return checkpoint
    
    def plot_training_history(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        train_acc = [m.get('accuracy', 0) for m in self.history['train_metrics']]
        val_acc = [m.get('accuracy', 0) for m in self.history['val_metrics']]
        
        axes[0, 1].plot(train_acc, label='Train Accuracy')
        axes[0, 1].plot(val_acc, label='Val Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score curves
        train_f1 = [m.get('f1_macro', 0) for m in self.history['train_metrics']]
        val_f1 = [m.get('f1_macro', 0) for m in self.history['val_metrics']]
        
        axes[1, 0].plot(train_f1, label='Train F1')
        axes[1, 0].plot(val_f1, label='Val F1')
        axes[1, 0].set_title('F1 Score Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = []
            # This would need to be tracked during training
            # For now, just show a placeholder
            axes[1, 1].plot([])
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


# Training script example
def main():
    """Main training function"""
    
    # Configuration
    config = {
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'image_size': (224, 224),
        'model_type': 'potholeNet',
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'use_focal': False,
        'mixed_precision': True,
        'early_stopping_patience': 10,
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        num_classes=2,
        with_depth=False
    )
    
    # Create data loaders (example - adjust paths)
    try:
        dataloaders = get_dataloaders(
            image_dir="./data/images",
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            task='classification'
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test'],
            config=config,
            device=device
        )
        
        # Start training
        history = trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure to prepare your dataset first!")


if __name__ == "__main__":
    main()