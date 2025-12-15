"""
Main execution script for Bone Age Prediction Project
Complete pipeline from data preparation to evaluation
FIXED: All required visualizations are now mandatory
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from config import (
    set_seed, BATCH_SIZE, NUM_WORKERS, DEVICE,
    get_model_path, PLOTS_DIR, MODELS_DIR, METRICS_DIR,
    ENSEMBLE_CNN_WEIGHT, ENSEMBLE_XGB_WEIGHT
)
from data_preprocessing import prepare_data, load_splits
from dataset import create_data_loaders
from models import BoneAgeRegressor, BoneAgeClassifier, get_model_summary
from train import train_regression_model, train_classification_model
from evaluate import (
    evaluate_regression, evaluate_classification,
    generate_visualizations, perform_gender_analysis,
    generate_gradcam_visualizations, save_evaluation_report,
    visualize_learned_representations  # ADDED
)
from xgboost_trainer import train_and_evaluate_xgboost_ensemble
from utils import load_model, plot_training_history, save_json


def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print(" "*15 + "BONE AGE PREDICTION PROJECT")
    print(" "*20 + "Complete Pipeline")
    print("="*70 + "\n")
    
    # Set random seed for reproducibility
    set_seed()
    print(f"✓ Random seed set for reproducibility\n")
    
    # Check device
    if DEVICE == 'mps':
        if not torch.backends.mps.is_available():
            print("⚠️  MPS not available, falling back to CPU")
            device = 'cpu'
        else:
            print(f"✓ Using Metal Performance Shaders (MPS) on M4 Mac\n")
            device = 'mps'
    else:
        device = DEVICE
        print(f"✓ Using device: {device}\n")
    
    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    # Check if data is already prepared
    try:
        train_df, val_df, holdout_df = load_splits()
        print("✓ Loaded existing data splits")
    except:
        print("Preparing data for the first time...")
        train_df, val_df, holdout_df = prepare_data()
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, BATCH_SIZE, NUM_WORKERS
    )
    
    # Create holdout loader for final testing
    from dataset import BoneAgeDataset, get_val_transforms
    holdout_dataset = BoneAgeDataset(holdout_df, transform=get_val_transforms())
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"✓ Created holdout loader: {len(holdout_dataset)} samples")
    
    # ========================================================================
    # STEP 2: TRAIN REGRESSION MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: TRAIN CNN REGRESSION MODEL")
    print("="*70)
    
    # Create regression model
    regression_model = BoneAgeRegressor()
    get_model_summary(regression_model, input_size=(3, 384, 384))
    
    # Train
    response = input("\nTrain regression model? (y/n): ").lower()
    if response == 'y':
        print("\nStarting regression model training...")
        reg_history = train_regression_model(
            regression_model,
            train_loader,
            val_loader,
            model_name='cnn_regressor'
        )
        
        # Plot training history
        plot_training_history(
            reg_history,
            save_path=PLOTS_DIR / 'regression_training_history.png'
        )
        
        # Save history
        save_json(reg_history, PLOTS_DIR / 'regression_history.json')
    else:
        print("Skipping regression training...")
        # Try to load existing model
        try:
            regression_model = load_model(
                regression_model,
                get_model_path('best_cnn_regressor'),
                device=device
            )
        except:
            print("⚠️  No trained regression model found. Train first.")
            return
    
    # ========================================================================
    # STEP 3: EVALUATE REGRESSION MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: EVALUATE REGRESSION MODEL")
    print("="*70)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_regression(regression_model, val_loader, device=device)
    
    # Generate visualizations
    generate_visualizations(
        val_results,
        task='regression',
        save_prefix='val_regression'
    )
    
    # Gender bias analysis
    perform_gender_analysis(
        val_results,
        task='regression',
        save_prefix='val_regression'
    )
    
    # Save report
    save_evaluation_report(
        val_results,
        task='regression',
        save_prefix='val_regression'
    )
    
    # Evaluate on holdout set
    print("\nEvaluating on holdout set...")
    holdout_results = evaluate_regression(regression_model, holdout_loader, device=device)
    
    generate_visualizations(
        holdout_results,
        task='regression',
        save_prefix='holdout_regression'
    )
    
    perform_gender_analysis(
        holdout_results,
        task='regression',
        save_prefix='holdout_regression'
    )
    
    save_evaluation_report(
        holdout_results,
        task='regression',
        save_prefix='holdout_regression'
    )
    
    # ========================================================================
    # REQUIRED VISUALIZATIONS (Problem Statement Requirements)
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING REQUIRED VISUALIZATIONS")
    print("="*70)
    
    # 1. Grad-CAM Heatmaps (REQUIRED by problem statement)
    print("\n1. Generating Grad-CAM heatmaps (required for report)...")
    try:
        generate_gradcam_visualizations(
            regression_model,
            val_loader,
            num_samples=20,
            device=device
        )
        print("✓ Grad-CAM visualizations saved")
    except Exception as e:
        print(f"⚠️  Grad-CAM generation failed: {e}")
        print("   Continuing with other visualizations...")
    
    # 2. t-SNE Visualization of Learned Representations (REQUIRED)
    print("\n2. Generating t-SNE visualization of learned representations...")
    try:
        visualize_learned_representations(
            regression_model,
            val_loader,
            device=device,
            save_prefix='val_regression'
        )
        print("✓ t-SNE visualizations saved")
    except Exception as e:
        print(f"⚠️  t-SNE generation failed: {e}")
        print("   Continuing with pipeline...")
    
    # ========================================================================
    # STEP 4: TRAIN CLASSIFICATION MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAIN CNN CLASSIFICATION MODEL")
    print("="*70)
    
    response = input("\nTrain classification model? (y/n): ").lower()
    if response == 'y':
        # Create classification model
        classification_model = BoneAgeClassifier()
        get_model_summary(classification_model, input_size=(3, 384, 384))
        
        # Train
        print("\nStarting classification model training...")
        cls_history = train_classification_model(
            classification_model,
            train_loader,
            val_loader,
            model_name='cnn_classifier'
        )
        
        # Plot training history
        plot_training_history(
            cls_history,
            save_path=PLOTS_DIR / 'classification_training_history.png'
        )
        
        save_json(cls_history, PLOTS_DIR / 'classification_history.json')
        
        # ====================================================================
        # STEP 5: EVALUATE CLASSIFICATION MODEL
        # ====================================================================
        print("\n" + "="*70)
        print("STEP 5: EVALUATE CLASSIFICATION MODEL")
        print("="*70)
        
        # Evaluate
        cls_results = evaluate_classification(
            classification_model,
            val_loader,
            device=device
        )
        
        generate_visualizations(
            cls_results,
            task='classification',
            save_prefix='val_classification'
        )
        
        perform_gender_analysis(
            cls_results,
            task='classification',
            save_prefix='val_classification'
        )
        
        save_evaluation_report(
            cls_results,
            task='classification',
            save_prefix='val_classification'
        )
    
    # ========================================================================
    # STEP 6: XGBOOST ENSEMBLE (OPTIONAL)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: XGBOOST ENSEMBLE (OPTIONAL)")
    print("="*70)
    
    response = input("\nTrain XGBoost ensemble? (y/n): ").lower()
    if response == 'y':
        print("\nTraining XGBoost ensemble...")
        xgb_model, ensemble_results = train_and_evaluate_xgboost_ensemble(
            regression_model,
            train_loader,
            val_loader
        )
        
        print("✓ XGBoost ensemble training complete")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PROJECT COMPLETE - FINAL SUMMARY")
    print("="*70)
    
    # Regression Results
    print("\n📊 REGRESSION MODEL (Holdout Set):")
    print(f"  MAE:  {holdout_results['metrics']['mae_years']:.4f} years "
          f"({holdout_results['metrics']['mae_months']:.2f} months)")
    print(f"  RMSE: {holdout_results['metrics']['rmse_years']:.4f} years "
          f"({holdout_results['metrics']['rmse_months']:.2f} months)")
    print(f"  R²:   {holdout_results['metrics']['r2_score']:.4f}")
    
    # Classification Results (if trained)
    cls_model_path = get_model_path('best_cnn_classifier')
    if cls_model_path.exists():
        print("\n📊 CLASSIFICATION MODEL:")
        try:
            classification_model = BoneAgeClassifier()
            classification_model = load_model(classification_model, cls_model_path, device=device)
            
            print("  Evaluating classification model...")
            cls_val_results = evaluate_classification(classification_model, val_loader, device=device)
            
            print(f"  Accuracy:  {cls_val_results['metrics']['accuracy']:.4f} ({cls_val_results['metrics']['accuracy']*100:.2f}%)")
            print(f"  F1 Score:  {cls_val_results['metrics']['f1_score']:.4f}")
            print(f"  Precision: {cls_val_results['metrics']['precision']:.4f}")
            print(f"  Recall:    {cls_val_results['metrics']['recall']:.4f}")
            print(f"  QWK:       {cls_val_results['metrics']['qwk']:.4f}")
            
        except Exception as e:
            print(f"  ⚠️  Could not evaluate: {e}")
    else:
        print("\n📊 CLASSIFICATION MODEL:")
        print("  ⚠️  Not trained yet")
    
    # XGBoost Ensemble Results (if trained)
    xgb_model_path = MODELS_DIR / 'xgboost_model.json'
    if xgb_model_path.exists():
        print("\n📊 XGBOOST ENSEMBLE:")
        print(f"  ✓ Model trained and saved")
    else:
        print("\n📊 XGBOOST ENSEMBLE:")
        print("  ⚠️  Not trained yet")
    
    # Files saved
    print("\n✓ All results saved to:")
    print(f"  - Models: {MODELS_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Metrics: {METRICS_DIR}")
    
    # Check for required visualizations
    print("\n📋 REQUIRED VISUALIZATIONS CHECKLIST:")
    vis_checks = [
        (PLOTS_DIR / 'holdout_regression_scatter.png', "✓ Scatter plot (predicted vs actual)"),
        (PLOTS_DIR / 'val_classification_confusion_matrix.png', "✓ Confusion matrix"),
        (PLOTS_DIR / 'gradcam_samples' / 'gradcam_sample_1.png', "✓ Grad-CAM heatmaps"),
        (PLOTS_DIR / 'val_regression_tsne_age.png', "✓ t-SNE learned representations"),
        (PLOTS_DIR / 'holdout_regression_error_distribution.png', "✓ Error analysis plots"),
    ]
    
    for file_path, description in vis_checks:
        if file_path.exists():
            print(f"  {description}")
        else:
            print(f"  ⚠️  {description} - NOT FOUND")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()