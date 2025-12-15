"""
Configuration file for Bone Age Prediction Project
Optimized for M4 Mac with 16GB RAM
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training data paths
TRAIN_IMG_DIR = DATASET_ROOT / "boneage-training-dataset"
TRAIN_CSV = DATASET_ROOT / "boneage-training-dataset.csv"

# Test data paths
TEST_IMG_DIR = DATASET_ROOT / "boneage-test-dataset"
TEST_CSV = DATASET_ROOT / "boneage-test-dataset.csv"

# Output directories
SPLITS_DIR = DATA_DIR / "splits"
ANALYSIS_DIR = DATA_DIR / "analysis"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
PLOTS_DIR = RESULTS_DIR / "plots"
GRADCAM_DIR = PLOTS_DIR / "gradcam_samples"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for dir_path in [DATA_DIR, SPLITS_DIR, ANALYSIS_DIR, MODELS_DIR, 
                 CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR, 
                 GRADCAM_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
# Image settings (384x384 for better resolution on M4 Mac)
IMAGE_SIZE = 384
INPUT_CHANNELS = 3

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HOLDOUT_RATIO = 0.15
RANDOM_SEED = 42

# Age binning for classification (in months)
AGE_BINS = [0, 60, 120, 180, 228]  # 4 classes
AGE_LABELS = ['Infant/Toddler', 'Child', 'Pre-adolescent', 'Adolescent']
NUM_CLASSES = len(AGE_LABELS)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Backbone
BACKBONE = 'efficientnet_b0'  # Light enough for M4 Air
PRETRAINED = True

# Feature dimensions
BACKBONE_FEATURES = 1280  # EfficientNet-B0 output
SEX_EMBEDDING_DIM = 32
TOTAL_FEATURES = BACKBONE_FEATURES + SEX_EMBEDDING_DIM

# Regression head
REGRESSION_HIDDEN_DIMS = [256, 128]
REGRESSION_DROPOUT = [0.3, 0.2]

# Classification head
CLASSIFICATION_HIDDEN_DIMS = [256, 128]
CLASSIFICATION_DROPOUT = [0.3, 0.2]

# ============================================================================
# TRAINING PARAMETERS - OPTIMIZED FOR M4 16GB RAM
# ============================================================================
# Device
DEVICE = 'mps'  # Metal Performance Shaders for M4 Mac

# Batch size (conservative for 16GB RAM with 384x384 images)
BATCH_SIZE = 16  # Reduced for larger images
NUM_WORKERS = 4  # M4 has good CPU cores

# Training epochs
NUM_EPOCHS_REGRESSION = 50
NUM_EPOCHS_CLASSIFICATION = 40

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = 'adamw'

# Learning rate scheduler
LR_SCHEDULER = 'cosine'
LR_WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001

# Loss functions
REGRESSION_LOSS = 'huber'  # More robust to outliers
HUBER_DELTA = 1.0
CLASSIFICATION_LOSS = 'cross_entropy'

# Gradient clipping
GRADIENT_CLIP_VAL = 1.0

# Mixed precision training (supported on M4)
USE_AMP = True

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
# Training augmentations (moderate for medical images)
TRAIN_AUGMENTATION = {
    'rotation_range': 15,
    'zoom_range': (0.9, 1.1),
    'brightness_range': (0.85, 1.15),
    'contrast_range': (0.85, 1.15),
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.0,  # Not anatomically valid
}

# Normalization (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# XGBOOST PARAMETERS
# ============================================================================
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'tree_method': 'hist',  # Faster for CPU
    'n_jobs': -1,
}

# Ensemble weights
ENSEMBLE_CNN_WEIGHT = 0.7
ENSEMBLE_XGB_WEIGHT = 0.3

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
# Metrics to track
REGRESSION_METRICS = ['mae', 'rmse', 'r2']
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'qwk']

# Visualization
NUM_GRADCAM_SAMPLES = 20
SCATTER_PLOT_ALPHA = 0.5
FIGURE_DPI = 150

# Error analysis
ERROR_PERCENTILES = [75, 90, 95]  # Top errors to analyze

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 50  # Log every N batches
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
VERBOSE = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_path(model_name):
    """Get full path for model file"""
    return MODELS_DIR / f"{model_name}.pth"

def get_checkpoint_path(model_name, epoch):
    """Get checkpoint path for specific epoch"""
    return CHECKPOINTS_DIR / f"{model_name}_epoch_{epoch}.pth"

def months_to_years(months):
    """Convert bone age from months to years"""
    return months / 12.0

def years_to_months(years):
    """Convert bone age from years to months"""
    return years * 12.0

def get_age_bin(age_months):
    """
    Get age bin index for classification
    Fixed to return proper indices (0-3)
    """
    import numpy as np
    # digitize returns 0 for values < first bin, len(bins)-1 for values >= last bin
    # We subtract 1 and clip to ensure we get indices 0-3
    bin_idx = np.digitize(age_months, AGE_BINS[1:])
    # Clip to valid range [0, NUM_CLASSES-1]
    return min(bin_idx, NUM_CLASSES - 1)

# Print configuration on import
if __name__ != "__main__":
    print(f"✓ Configuration loaded")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Device: {DEVICE}")
    print(f"  - Backbone: {BACKBONE}")