"""
Utility functions for Bone Age Prediction Project
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Any


def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Path):
    """Save object to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Saved pickle to {filepath}")


def load_pickle(filepath: Path) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_model(model: torch.nn.Module, filepath: Path, metadata: Dict = None):
    """Save PyTorch model with optional metadata"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Saved model to {filepath}")


def load_model(model: torch.nn.Module, filepath: Path, device: str = 'cpu') -> torch.nn.Module:
    """Load PyTorch model"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from {filepath}")
    if 'metadata' in checkpoint:
        print(f"  Metadata: {checkpoint['metadata']}")
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def plot_training_history(history: Dict, save_path: Path = None):
    """Plot training and validation metrics"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric plot (MAE or Accuracy)
    # Check what metrics are available in history
    if 'train_metric' in history and 'val_metric' in history:
        # Use generic metric key
        axes[1].plot(history['train_metric'], label='Train Metric', linewidth=2)
        axes[1].plot(history['val_metric'], label='Val Metric', linewidth=2)
        metric_name = 'MAE/Accuracy'
    else:
        # Fall back to checking specific keys
        if 'val_mae' in history:
            metric_name = 'MAE'
            if 'train_mae' in history:
                axes[1].plot(history['train_mae'], label=f'Train {metric_name}', linewidth=2)
            axes[1].plot(history['val_mae'], label=f'Val {metric_name}', linewidth=2)
        elif 'val_accuracy' in history:
            metric_name = 'Accuracy'
            if 'train_accuracy' in history:
                axes[1].plot(history['train_accuracy'], label=f'Train {metric_name}', linewidth=2)
            axes[1].plot(history['val_accuracy'], label=f'Val {metric_name}', linewidth=2)
        else:
            # No metric found, skip this plot
            axes[1].text(0.5, 0.5, 'No metric data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            metric_name = 'Metric'
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel(metric_name, fontsize=12)
    axes[1].set_title(f'Training and Validation {metric_name}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history plot to {save_path}")
    
    plt.close()


def plot_predictions_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Predicted vs Actual Bone Age",
                            save_path: Path = None):
    """Plot scatter plot of predictions vs actual values"""
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Actual Bone Age (years)', fontsize=12)
    plt.ylabel('Predicted Bone Age (years)', fontsize=12)
    plt.title(f'{title}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved scatter plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str],
                         title: str = "Confusion Matrix",
                         save_path: Path = None):
    """Plot confusion matrix for classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                           title: str = "Error Distribution",
                           save_path: Path = None):
    """Plot distribution of prediction errors"""
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Error histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (years)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error vs actual age
    axes[1].scatter(y_true, errors, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Actual Bone Age (years)', fontsize=12)
    axes[1].set_ylabel('Prediction Error (years)', fontsize=12)
    axes[1].set_title('Error vs Actual Age', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved error distribution plot to {save_path}")
    
    plt.show()


def analyze_gender_bias(df: pd.DataFrame, y_true_col: str, y_pred_col: str, sex_col: str):
    """Analyze model performance by gender"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    results = {}
    
    for sex in df[sex_col].unique():
        subset = df[df[sex_col] == sex]
        y_true = subset[y_true_col].values
        y_pred = subset[y_pred_col].values
        
        results[sex] = {
            'count': len(subset),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(y_pred - y_true),
            'std_error': np.std(y_pred - y_true)
        }
    
    # Print results
    print("\n" + "="*70)
    print("GENDER-WISE PERFORMANCE ANALYSIS")
    print("="*70)
    
    for sex, metrics in results.items():
        print(f"\n{sex}:")
        print(f"  Sample count: {metrics['count']}")
        print(f"  MAE: {metrics['mae']:.4f} years")
        print(f"  RMSE: {metrics['rmse']:.4f} years")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Mean Error: {metrics['mean_error']:.4f} years")
        print(f"  Std Error: {metrics['std_error']:.4f} years")
    
    # Check for significant bias
    sexes = list(results.keys())
    if len(sexes) == 2:
        mae_diff = abs(results[sexes[0]]['mae'] - results[sexes[1]]['mae'])
        mean_error_diff = abs(results[sexes[0]]['mean_error'] - results[sexes[1]]['mean_error'])
        
        print(f"\n{'Bias Analysis':}")
        print(f"  MAE difference: {mae_diff:.4f} years")
        print(f"  Mean error difference: {mean_error_diff:.4f} years")
        
        if mae_diff > 0.5:
            print(f"  ⚠️  Significant MAE difference detected between genders")
        else:
            print(f"  ✓ No significant MAE bias between genders")
    
    print("="*70 + "\n")
    
    return results


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered!")
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            return score > (self.best_score + self.min_delta)


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """Pretty print metrics dictionary"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")
    print(f"{'='*50}\n")