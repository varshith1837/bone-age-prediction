"""
Training functions for Bone Age Prediction models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from config import (
    DEVICE, NUM_EPOCHS_REGRESSION, NUM_EPOCHS_CLASSIFICATION,
    LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER,
    LR_SCHEDULER, LR_WARMUP_EPOCHS, MIN_LR,
    PATIENCE, MIN_DELTA, GRADIENT_CLIP_VAL,
    USE_AMP, LOG_INTERVAL, SAVE_INTERVAL,
    REGRESSION_LOSS, HUBER_DELTA,
    get_model_path, get_checkpoint_path
)
from utils import EarlyStopping, format_time, save_model


class Trainer:
    """
    Trainer class for bone age models
    """
    
    def __init__(self, model, train_loader, val_loader, task='regression', device=DEVICE):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            task: 'regression' or 'classification'
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.device = device
        
        # Setup loss function
        if task == 'regression':
            if REGRESSION_LOSS == 'huber':
                self.criterion = nn.HuberLoss(delta=HUBER_DELTA)
            elif REGRESSION_LOSS == 'mse':
                self.criterion = nn.MSELoss()
            else:  # mae
                self.criterion = nn.L1Loss()
        else:  # classification
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if OPTIMIZER.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
        else:  # sgd
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=LEARNING_RATE,
                momentum=0.9,
                weight_decay=WEIGHT_DECAY
            )
        
        # Setup learning rate scheduler
        num_epochs = NUM_EPOCHS_REGRESSION if task == 'regression' else NUM_EPOCHS_CLASSIFICATION
        
        if LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - LR_WARMUP_EPOCHS,
                eta_min=MIN_LR
            )
        elif LR_SCHEDULER == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_epochs // 3,
                gamma=0.1
            )
        else:  # reduce on plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=MIN_LR
            )
        
        # Disable AMP for MPS - not well supported in PyTorch 2.1.0
        self.use_amp = False
        self.scaler = None
        
        # Early stopping
        mode = 'min' if task == 'regression' else 'max'
        self.early_stopping = EarlyStopping(
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            mode=mode,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'lr': []
        }
        
        self.best_val_metric = float('inf') if task == 'regression' else 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_metric = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            sex = batch['sex'].to(self.device)
            
            if self.task == 'regression':
                targets = batch['bone_age'].to(self.device)
            else:
                targets = batch['age_category'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, sex)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if GRADIENT_CLIP_VAL > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_VAL)
            
            self.optimizer.step()
            
            # Calculate metric
            if self.task == 'regression':
                metric = torch.abs(outputs - targets).mean().item()  # MAE
            else:
                _, predicted = torch.max(outputs, 1)
                metric = (predicted == targets).float().mean().item()  # Accuracy
            
            # Update running stats
            running_loss += loss.item()
            running_metric += metric
            
            # Update progress bar
            if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == num_batches:
                avg_loss = running_loss / (batch_idx + 1)
                avg_metric = running_metric / (batch_idx + 1)
                metric_name = 'MAE' if self.task == 'regression' else 'Acc'
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    metric_name: f'{avg_metric:.4f}'
                })
        
        # Calculate epoch averages
        epoch_loss = running_loss / num_batches
        epoch_metric = running_metric / num_batches
        
        return epoch_loss, epoch_metric
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_metric = 0.0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]  ')
        
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                sex = batch['sex'].to(self.device)
                
                if self.task == 'regression':
                    targets = batch['bone_age'].to(self.device)
                else:
                    targets = batch['age_category'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, sex)
                loss = self.criterion(outputs, targets)
                
                # Calculate metric
                if self.task == 'regression':
                    metric = torch.abs(outputs - targets).mean().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    metric = (predicted == targets).float().mean().item()
                
                running_loss += loss.item()
                running_metric += metric
        
        # Calculate epoch averages
        epoch_loss = running_loss / num_batches
        epoch_metric = running_metric / num_batches
        
        return epoch_loss, epoch_metric
    
    def train(self, num_epochs, model_name='model'):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            model_name: Name for saving model
        """
        print("\n" + "="*70)
        print(f"TRAINING {self.task.upper()} MODEL")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Optimizer: {OPTIMIZER}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Loss function: {self.criterion.__class__.__name__}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metric = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metric = self.validate_epoch(epoch)
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif epoch >= LR_WARMUP_EPOCHS:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            metric_name = 'MAE' if self.task == 'regression' else 'Acc'
            
            print(f"\nEpoch {epoch+1}/{num_epochs} - {format_time(epoch_time)}")
            print(f"  Train Loss: {train_loss:.4f} | Train {metric_name}: {train_metric:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val {metric_name}:   {val_metric:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            if self.task == 'regression':
                is_best = val_metric < self.best_val_metric
            else:
                is_best = val_metric > self.best_val_metric
            
            if is_best:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                
                # Save best model
                save_model(
                    self.model,
                    get_model_path(f'best_{model_name}'),
                    metadata={
                        'epoch': epoch + 1,
                        'val_loss': val_loss,
                        f'val_{metric_name.lower()}': val_metric,
                        'task': self.task
                    }
                )
                print(f"  ✓ New best model saved! ({metric_name}: {val_metric:.4f})")
            
            # Save checkpoint periodically
            if (epoch + 1) % SAVE_INTERVAL == 0:
                save_model(
                    self.model,
                    get_checkpoint_path(model_name, epoch + 1),
                    metadata={
                        'epoch': epoch + 1,
                        'val_loss': val_loss,
                        f'val_{metric_name.lower()}': val_metric
                    }
                )
            
            # Early stopping
            if self.early_stopping(val_metric):
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best epoch: {self.best_epoch + 1}")
        print(f"Best val {metric_name}: {self.best_val_metric:.4f}")
        print("="*70 + "\n")
        
        return self.history


def train_regression_model(model, train_loader, val_loader, model_name='cnn_regressor'):
    """
    Train regression model
    
    Args:
        model: BoneAgeRegressor model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name for saving
        
    Returns:
        dict: Training history
    """
    trainer = Trainer(model, train_loader, val_loader, task='regression')
    history = trainer.train(NUM_EPOCHS_REGRESSION, model_name)
    return history


def train_classification_model(model, train_loader, val_loader, model_name='cnn_classifier'):
    """
    Train classification model
    
    Args:
        model: BoneAgeClassifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name for saving
        
    Returns:
        dict: Training history
    """
    trainer = Trainer(model, train_loader, val_loader, task='classification')
    history = trainer.train(NUM_EPOCHS_CLASSIFICATION, model_name)
    return history


if __name__ == "__main__":
    from data_preprocessing import load_splits
    from dataset import create_data_loaders
    from models import BoneAgeRegressor
    from config import BATCH_SIZE, NUM_WORKERS, set_seed
    
    # Set seed for reproducibility
    set_seed()
    
    print("Testing training pipeline...")
    
    # Load data
    train_df, val_df, _ = load_splits()
    train_loader, val_loader = create_data_loaders(train_df, val_df, BATCH_SIZE, NUM_WORKERS)
    
    # Create model
    model = BoneAgeRegressor()
    
    # Test training for 2 epochs
    print("\n" + "="*70)
    print("Running quick training test (2 epochs)...")
    print("="*70)
    
    trainer = Trainer(model, train_loader, val_loader, task='regression')
    history = trainer.train(num_epochs=2, model_name='test_model')
    
    print("\n✓ Training test complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")