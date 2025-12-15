"""
Generate predictions on the official test dataset
This creates a submission file with predictions for the test set
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    DEVICE, TEST_CSV, TEST_IMG_DIR, BATCH_SIZE, NUM_WORKERS,
    get_model_path, RESULTS_DIR, years_to_months
)
from models import BoneAgeRegressor, BoneAgeClassifier
from dataset import TestDataset, get_val_transforms
from utils import load_model


def load_test_data():
    """Load test dataset CSV"""
    print("\n" + "="*70)
    print("LOADING TEST DATASET")
    print("="*70)
    
    # Load test CSV
    test_df = pd.read_csv(TEST_CSV)
    print(f"✓ Loaded test CSV: {len(test_df)} samples")
    print(f"  Columns: {list(test_df.columns)}")
    
    # Standardize column names
    if 'Case ID' in test_df.columns:
        test_df = test_df.rename(columns={'Case ID': 'image_id'})
    elif 'id' in test_df.columns:
        test_df = test_df.rename(columns={'id': 'image_id'})
    
    # Standardize sex column
    if 'Sex' in test_df.columns:
        # Convert M/F to binary (1/0)
        test_df['sex_binary'] = test_df['Sex'].map({'M': 1, 'F': 0})
        test_df['sex'] = test_df['Sex']
    elif 'male' in test_df.columns:
        test_df['sex_binary'] = test_df['male'].astype(int)
        test_df['sex'] = test_df['male'].map({True: 'M', False: 'F'})
    
    print(f"\nTest set composition:")
    print(f"  Total samples: {len(test_df)}")
    if 'sex' in test_df.columns:
        print(f"  Gender distribution: {test_df['sex'].value_counts().to_dict()}")
    
    return test_df


def create_test_loader(test_df):
    """Create DataLoader for test dataset"""
    print("\nCreating test data loader...")
    
    # Create sex mapping dictionary
    sex_data = dict(zip(test_df['image_id'], test_df['sex'])) if 'sex' in test_df.columns else None
    
    # Create dataset
    test_dataset = TestDataset(
        image_dir=TEST_IMG_DIR,
        image_ids=test_df['image_id'].tolist(),
        sex_data=sex_data,
        transform=get_val_transforms()
    )
    
    # Create loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✓ Test loader created: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return test_loader


def generate_regression_predictions(model, test_loader, device=DEVICE):
    """Generate bone age predictions using regression model"""
    print("\n" + "="*70)
    print("GENERATING REGRESSION PREDICTIONS")
    print("="*70)
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_image_ids = []
    
    print("\nPredicting...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(device)
            sex = batch['sex'].to(device)
            image_ids = batch['image_id']
            
            # Predict bone age
            predictions = model(images, sex)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_image_ids.extend(image_ids)
    
    # Convert to numpy array
    predictions_years = np.array(all_predictions)
    predictions_months = predictions_years * 12  # Convert to months
    
    print(f"\n✓ Generated {len(predictions_years)} predictions")
    print(f"  Age range: {predictions_years.min():.2f} - {predictions_years.max():.2f} years")
    print(f"  Mean age: {predictions_years.mean():.2f} years")
    print(f"  Std dev: {predictions_years.std():.2f} years")
    
    return all_image_ids, predictions_years, predictions_months


def generate_classification_predictions(model, test_loader, device=DEVICE):
    """Generate age category predictions using classification model"""
    print("\n" + "="*70)
    print("GENERATING CLASSIFICATION PREDICTIONS")
    print("="*70)
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_probabilities = []
    all_image_ids = []
    
    print("\nPredicting...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(device)
            sex = batch['sex'].to(device)
            image_ids = batch['image_id']
            
            # Predict age category
            logits = model(images, sex)
            probabilities = torch.softmax(logits, dim=1)
            _, predictions = torch.max(logits, 1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_image_ids.extend(image_ids)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    
    from config import AGE_LABELS
    
    print(f"\n✓ Generated {len(predictions)} predictions")
    print(f"\nPredicted category distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for cat, count in zip(unique, counts):
        print(f"  {AGE_LABELS[cat]}: {count} ({count/len(predictions)*100:.1f}%)")
    
    return all_image_ids, predictions, probabilities


def save_predictions(test_df, regression_preds_years, regression_preds_months, 
                     classification_preds=None, save_dir=RESULTS_DIR):
    """Save predictions to CSV files"""
    print("\n" + "="*70)
    print("SAVING PREDICTIONS")
    print("="*70)
    
    # Create predictions directory
    pred_dir = save_dir / 'test_predictions'
    pred_dir.mkdir(exist_ok=True)
    
    # Create base dataframe
    results_df = test_df[['image_id']].copy()
    
    if 'sex' in test_df.columns:
        results_df['sex'] = test_df['sex']
    
    # Add regression predictions
    results_df['predicted_age_years'] = regression_preds_years
    results_df['predicted_age_months'] = regression_preds_months
    
    # Add classification predictions if available
    if classification_preds is not None:
        from config import AGE_LABELS
        results_df['predicted_category'] = classification_preds
        results_df['predicted_category_name'] = [AGE_LABELS[c] for c in classification_preds]
    
    # Save detailed predictions
    detailed_file = pred_dir / 'test_predictions_detailed.csv'
    results_df.to_csv(detailed_file, index=False)
    print(f"✓ Saved detailed predictions: {detailed_file}")
    
    # Save Kaggle-style submission (if needed)
    submission_df = pd.DataFrame({
        'Case ID': results_df['image_id'],
        'boneage': results_df['predicted_age_months']
    })
    submission_file = pred_dir / 'test_predictions_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"✓ Saved submission format: {submission_file}")
    
    # Save simple format for easy viewing
    simple_df = results_df[['image_id', 'predicted_age_years', 'predicted_age_months']].copy()
    simple_df = simple_df.round(2)
    simple_file = pred_dir / 'test_predictions_simple.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"✓ Saved simple format: {simple_file}")
    
    return results_df


def print_summary_statistics(results_df):
    """Print summary statistics of predictions"""
    print("\n" + "="*70)
    print("PREDICTION SUMMARY STATISTICS")
    print("="*70)
    
    print("\n📊 Overall Statistics:")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Age range: {results_df['predicted_age_years'].min():.2f} - "
          f"{results_df['predicted_age_years'].max():.2f} years")
    print(f"  Mean age: {results_df['predicted_age_years'].mean():.2f} ± "
          f"{results_df['predicted_age_years'].std():.2f} years")
    print(f"  Median age: {results_df['predicted_age_years'].median():.2f} years")
    
    # Gender-wise statistics
    if 'sex' in results_df.columns:
        print("\n📊 Gender-wise Statistics:")
        for sex in results_df['sex'].unique():
            subset = results_df[results_df['sex'] == sex]
            print(f"\n  {sex}:")
            print(f"    Count: {len(subset)}")
            print(f"    Mean age: {subset['predicted_age_years'].mean():.2f} years")
            print(f"    Std dev: {subset['predicted_age_years'].std():.2f} years")
            print(f"    Range: {subset['predicted_age_years'].min():.2f} - "
                  f"{subset['predicted_age_years'].max():.2f} years")
    
    # Classification statistics
    if 'predicted_category_name' in results_df.columns:
        print("\n📊 Age Category Distribution:")
        category_counts = results_df['predicted_category_name'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} ({count/len(results_df)*100:.1f}%)")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print(" "*15 + "TEST SET PREDICTIONS")
    print(" "*10 + "Bone Age Prediction Project")
    print("="*70)
    
    # Load test data
    test_df = load_test_data()
    
    # Create test loader
    test_loader = create_test_loader(test_df)
    
    # ========================================================================
    # REGRESSION PREDICTIONS
    # ========================================================================
    
    # Load regression model
    print("\n" + "="*70)
    print("LOADING REGRESSION MODEL")
    print("="*70)
    
    regression_model = BoneAgeRegressor()
    regression_model_path = get_model_path('best_cnn_regressor')
    
    if not regression_model_path.exists():
        print("❌ ERROR: Regression model not found!")
        print(f"   Expected at: {regression_model_path}")
        print("   Train the model first using: python main.py")
        return
    
    regression_model = load_model(regression_model, regression_model_path, device=DEVICE)
    
    # Generate predictions
    image_ids, pred_years, pred_months = generate_regression_predictions(
        regression_model, test_loader, device=DEVICE
    )
    
    # ========================================================================
    # CLASSIFICATION PREDICTIONS (Optional)
    # ========================================================================
    
    classification_preds = None
    classification_model_path = get_model_path('best_cnn_classifier')
    
    if classification_model_path.exists():
        print("\n" + "="*70)
        print("LOADING CLASSIFICATION MODEL")
        print("="*70)
        
        classification_model = BoneAgeClassifier()
        classification_model = load_model(
            classification_model, classification_model_path, device=DEVICE
        )
        
        # Generate predictions
        _, classification_preds, _ = generate_classification_predictions(
            classification_model, test_loader, device=DEVICE
        )
    else:
        print("\n⚠️  Classification model not found - skipping classification predictions")
    
    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================
    
    results_df = save_predictions(
        test_df, pred_years, pred_months, classification_preds
    )
    
    # Print summary
    print_summary_statistics(results_df)
    
    # Final message
    print("\n" + "="*70)
    print("✅ TEST PREDICTIONS COMPLETE")
    print("="*70)
    print(f"\n📁 Predictions saved to: {RESULTS_DIR / 'test_predictions'}/")
    print("\nFiles created:")
    print("  1. test_predictions_detailed.csv  - All predictions with metadata")
    print("  2. test_predictions_submission.csv - Kaggle submission format")
    print("  3. test_predictions_simple.csv    - Simple format (ID, age)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()