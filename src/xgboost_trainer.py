"""
XGBoost ensemble training for Bone Age Prediction
"""

import torch
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    DEVICE, XGBOOST_PARAMS, MODELS_DIR,
    ENSEMBLE_CNN_WEIGHT, ENSEMBLE_XGB_WEIGHT
)
from utils import save_pickle, load_pickle, print_metrics


def extract_features(model, data_loader, device=DEVICE):
    """
    Extract features from CNN model for XGBoost training
    
    Args:
        model: Trained BoneAgeRegressor model
        data_loader: Data loader
        device: Device to run on
        
    Returns:
        tuple: (features, targets, predictions)
    """
    model.eval()
    model = model.to(device)
    
    all_features = []
    all_targets = []
    all_predictions = []
    
    print("\nExtracting features from CNN...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting"):
            images = batch['image'].to(device)
            sex = batch['sex'].to(device)
            targets = batch['bone_age']
            
            # Extract features
            features = model.extract_features(images, sex)
            
            # Also get CNN predictions
            predictions = model(images, sex)
            
            all_features.append(features.cpu().numpy())
            all_targets.append(targets.numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all batches
    features = np.vstack(all_features)
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    return features, targets, predictions


def train_xgboost_model(train_features, train_targets, val_features, val_targets):
    """
    Train XGBoost regression model
    
    Args:
        train_features: Training features from CNN
        train_targets: Training bone age targets
        val_features: Validation features
        val_targets: Validation targets
        
    Returns:
        xgb.Booster: Trained XGBoost model
    """
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL")
    print("="*70)
    
    # Create DMatrix
    dtrain = xgb.DMatrix(train_features, label=train_targets)
    dval = xgb.DMatrix(val_features, label=val_targets)
    
    # Training parameters
    params = XGBOOST_PARAMS.copy()
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = ['mae', 'rmse']
    
    print(f"Training samples: {train_features.shape[0]}")
    print(f"Validation samples: {val_features.shape[0]}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"\nXGBoost parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Train model with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    print("\nTraining...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")
    
    return model


def evaluate_xgboost(model, features, targets):
    """
    Evaluate XGBoost model
    
    Args:
        model: Trained XGBoost model
        features: Features to predict on
        targets: True targets
        
    Returns:
        dict: Evaluation metrics
    """
    # Predict
    dtest = xgb.DMatrix(features)
    predictions = model.predict(dtest)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    metrics = {
        'mae_years': mae,
        'rmse_years': rmse,
        'r2_score': r2,
        'mae_months': mae * 12,
        'rmse_months': rmse * 12
    }
    
    return metrics, predictions


def create_ensemble_predictions(cnn_predictions, xgb_predictions, 
                                cnn_weight=ENSEMBLE_CNN_WEIGHT,
                                xgb_weight=ENSEMBLE_XGB_WEIGHT):
    """
    Create weighted ensemble predictions
    
    Args:
        cnn_predictions: Predictions from CNN model
        xgb_predictions: Predictions from XGBoost model
        cnn_weight: Weight for CNN predictions
        xgb_weight: Weight for XGBoost predictions
        
    Returns:
        np.ndarray: Ensemble predictions
    """
    ensemble_pred = cnn_weight * cnn_predictions + xgb_weight * xgb_predictions
    return ensemble_pred


def train_and_evaluate_xgboost_ensemble(cnn_model, train_loader, val_loader):
    """
    Complete pipeline to train XGBoost and create ensemble
    
    Args:
        cnn_model: Trained CNN regression model
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        tuple: (xgb_model, results_dict)
    """
    print("\n" + "="*70)
    print("XGBOOST ENSEMBLE TRAINING PIPELINE")
    print("="*70)
    
    # Extract features
    print("\n1. Extracting training features...")
    train_features, train_targets, train_cnn_pred = extract_features(cnn_model, train_loader)
    
    print("\n2. Extracting validation features...")
    val_features, val_targets, val_cnn_pred = extract_features(cnn_model, val_loader)
    
    # Train XGBoost
    print("\n3. Training XGBoost model...")
    xgb_model = train_xgboost_model(train_features, train_targets, val_features, val_targets)
    
    # Save XGBoost model
    xgb_model_path = MODELS_DIR / 'xgboost_model.json'
    xgb_model.save_model(str(xgb_model_path))
    print(f"✓ XGBoost model saved to {xgb_model_path}")
    
    # Evaluate individual models
    print("\n4. Evaluating models...")
    
    # CNN metrics
    cnn_mae = mean_absolute_error(val_targets, val_cnn_pred)
    cnn_rmse = np.sqrt(mean_squared_error(val_targets, val_cnn_pred))
    cnn_r2 = r2_score(val_targets, val_cnn_pred)
    
    print("\nCNN Model:")
    print_metrics({
        'mae_years': cnn_mae,
        'rmse_years': cnn_rmse,
        'r2_score': cnn_r2
    }, "CNN Metrics")
    
    # XGBoost metrics
    xgb_metrics, val_xgb_pred = evaluate_xgboost(xgb_model, val_features, val_targets)
    
    print("\nXGBoost Model:")
    print_metrics(xgb_metrics, "XGBoost Metrics")
    
    # Ensemble metrics
    ensemble_pred = create_ensemble_predictions(val_cnn_pred, val_xgb_pred)
    
    ensemble_mae = mean_absolute_error(val_targets, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(val_targets, ensemble_pred))
    ensemble_r2 = r2_score(val_targets, ensemble_pred)
    
    print("\nEnsemble Model:")
    print_metrics({
        'mae_years': ensemble_mae,
        'rmse_years': ensemble_rmse,
        'r2_score': ensemble_r2,
        'cnn_weight': ENSEMBLE_CNN_WEIGHT,
        'xgb_weight': ENSEMBLE_XGB_WEIGHT
    }, "Ensemble Metrics")
    
    # Compare improvements
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"CNN MAE:      {cnn_mae:.4f} years")
    print(f"XGBoost MAE:  {xgb_metrics['mae_years']:.4f} years")
    print(f"Ensemble MAE: {ensemble_mae:.4f} years")
    
    mae_improvement = ((cnn_mae - ensemble_mae) / cnn_mae) * 100
    r2_improvement = ((ensemble_r2 - cnn_r2) / cnn_r2) * 100
    
    print(f"\nImprovement over CNN:")
    print(f"  MAE: {mae_improvement:.2f}%")
    print(f"  R²:  {r2_improvement:.2f}%")
    print("="*70)
    
    results = {
        'cnn_metrics': {'mae': cnn_mae, 'rmse': cnn_rmse, 'r2': cnn_r2},
        'xgb_metrics': xgb_metrics,
        'ensemble_metrics': {'mae': ensemble_mae, 'rmse': ensemble_rmse, 'r2': ensemble_r2},
        'val_predictions': {
            'cnn': val_cnn_pred,
            'xgb': val_xgb_pred,
            'ensemble': ensemble_pred
        },
        'val_targets': val_targets
    }
    
    return xgb_model, results


if __name__ == "__main__":
    from data_preprocessing import load_splits
    from dataset import create_data_loaders
    from models import BoneAgeRegressor
    from config import BATCH_SIZE, NUM_WORKERS, set_seed, get_model_path
    from utils import load_model
    
    # Set seed
    set_seed()
    
    print("Testing XGBoost ensemble pipeline...")
    
    # Load data
    train_df, val_df, _ = load_splits()
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, BATCH_SIZE, NUM_WORKERS
    )
    
    # Create dummy CNN model (in practice, load trained model)
    print("\nCreating CNN model (using untrained for testing)...")
    cnn_model = BoneAgeRegressor()
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features, targets, predictions = extract_features(cnn_model, val_loader)
    
    print(f"\n✓ XGBoost pipeline test complete!")
    print(f"  Features extracted: {features.shape}")
    print(f"  Ready for XGBoost training")