"""
Evaluation functions for Bone Age Prediction models
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from config import (
    DEVICE, AGE_LABELS, PLOTS_DIR, GRADCAM_DIR,
    METRICS_DIR, NUM_GRADCAM_SAMPLES, IMAGENET_MEAN, IMAGENET_STD
)
from utils import (
    plot_predictions_scatter, plot_confusion_matrix,
    plot_error_distribution, analyze_gender_bias, print_metrics
)


def evaluate_regression(model, data_loader, device=DEVICE):
    """
    Evaluate regression model
    
    Args:
        model: BoneAgeRegressor model
        data_loader: Data loader
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_image_ids = []
    all_sex = []
    
    print("\nEvaluating regression model...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            sex = batch['sex'].to(device)
            targets = batch['bone_age'].to(device)
            
            # Predict
            predictions = model(images, sex)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_image_ids.extend(batch['image_id'])
            all_sex.extend(sex.cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)
    sex_array = np.array(all_sex)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAE in months for comparison
    mae_months = mae * 12
    rmse_months = rmse * 12
    
    metrics = {
        'mae_years': mae,
        'rmse_years': rmse,
        'r2_score': r2,
        'mae_months': mae_months,
        'rmse_months': rmse_months,
        'n_samples': len(y_true)
    }
    
    results = {
        'metrics': metrics,
        'predictions': y_pred,
        'targets': y_true,
        'image_ids': all_image_ids,
        'sex': sex_array
    }
    
    print_metrics(metrics, "Regression Metrics")
    
    return results


def evaluate_classification(model, data_loader, device=DEVICE):
    """
    Evaluate classification model
    
    Args:
        model: BoneAgeClassifier model
        data_loader: Data loader
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_image_ids = []
    all_sex = []
    
    print("\nEvaluating classification model...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            sex = batch['sex'].to(device)
            targets = batch['age_category'].to(device)
            
            # Predict
            logits = model(images, sex)
            _, predictions = torch.max(logits, 1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_image_ids.extend(batch['image_id'])
            all_sex.extend(sex.cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)
    sex_array = np.array(all_sex)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'qwk': qwk,
        'n_samples': len(y_true)
    }
    
    results = {
        'metrics': metrics,
        'predictions': y_pred,
        'targets': y_true,
        'image_ids': all_image_ids,
        'sex': sex_array
    }
    
    print_metrics(metrics, "Classification Metrics")
    
    return results


def generate_visualizations(results, task='regression', save_prefix=''):
    """
    Generate and save visualizations
    
    Args:
        results: Results dictionary from evaluation
        task: 'regression' or 'classification'
        save_prefix: Prefix for saved files
    """
    print(f"\nGenerating {task} visualizations...")
    
    y_true = results['targets']
    y_pred = results['predictions']
    
    if task == 'regression':
        # Scatter plot
        plot_predictions_scatter(
            y_true, y_pred,
            title=f"{save_prefix} Predicted vs Actual Bone Age",
            save_path=PLOTS_DIR / f'{save_prefix}_scatter.png'
        )
        
        # Error distribution
        plot_error_distribution(
            y_true, y_pred,
            title=f"{save_prefix} Error Analysis",
            save_path=PLOTS_DIR / f'{save_prefix}_error_distribution.png'
        )
        
    else:  # classification
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=AGE_LABELS,
            title=f"{save_prefix} Confusion Matrix",
            save_path=PLOTS_DIR / f'{save_prefix}_confusion_matrix.png'
        )
    
    print("✓ Visualizations saved")


def perform_gender_analysis(results, task='regression', save_prefix=''):
    """
    Perform gender-wise bias analysis
    
    Args:
        results: Results dictionary from evaluation
        task: 'regression' or 'classification'
        save_prefix: Prefix for saved files
    """
    print(f"\nPerforming gender-wise analysis...")
    
    # Create dataframe
    df = pd.DataFrame({
        'y_true': results['targets'],
        'y_pred': results['predictions'],
        'sex': ['M' if s == 1 else 'F' for s in results['sex']]
    })
    
    if task == 'regression':
        bias_results = analyze_gender_bias(df, 'y_true', 'y_pred', 'sex')
        
        # Save bias analysis
        bias_file = METRICS_DIR / f'{save_prefix}_gender_bias.txt'
        with open(bias_file, 'w') as f:
            f.write("GENDER-WISE BIAS ANALYSIS\n")
            f.write("="*70 + "\n\n")
            for sex, metrics in bias_results.items():
                f.write(f"{sex}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"✓ Bias analysis saved to {bias_file}")
        
    else:  # classification
        from sklearn.metrics import classification_report
        
        for sex in ['M', 'F']:
            subset = df[df['sex'] == sex]
            print(f"\n{sex} Performance:")
            print(classification_report(
                subset['y_true'],
                subset['y_pred'],
                target_names=AGE_LABELS,
                zero_division=0
            ))


def generate_gradcam_visualizations(model, data_loader, num_samples=NUM_GRADCAM_SAMPLES, device=DEVICE):
    """
    Generate Grad-CAM visualizations (Corrected for Regression dimensions)
    """
    print(f"\nGenerating Grad-CAM visualizations for {num_samples} samples...")
    
    model.eval()
    model = model.to(device)
    
    # Target the last convolutional layer in the backbone
    target_layers = [model.backbone.conv_head]
    
    # Denormalization function
    def denormalize(tensor):
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return tensor * std + mean
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= num_samples:
            break
        
        images = batch['image'].to(device)
        sex = batch['sex'].to(device)
        
        batch_size = images.size(0)
        
        for i in range(batch_size):
            if sample_count >= num_samples:
                break
            
            # Get single image
            img_tensor = images[i:i+1]
            sex_val = sex[i:i+1]
            
            # --- FIX: Wrapper now unsqueezes output to (Batch, 1) ---
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model, sex):
                    super().__init__()
                    self.model = model
                    self.sex = sex
                
                def forward(self, x):
                    # Original model returns shape (Batch,)
                    # We unsqueeze to (Batch, 1) so GradCAM can index it
                    return self.model(x, self.sex).unsqueeze(1)
            
            wrapped_model = ModelWrapper(model, sex_val)
            wrapped_cam = GradCAM(model=wrapped_model, target_layers=target_layers)
            
            # Target index 0 of the now 2D output
            targets = [ClassifierOutputTarget(0)]
            
            # Generate CAM
            grayscale_cam = wrapped_cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Denormalize image for visualization
            img_denorm = denormalize(img_tensor[0].cpu())
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_np = img_denorm.permute(1, 2, 0).numpy()
            
            # Create visualization
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # Save
            save_path = GRADCAM_DIR / f'gradcam_sample_{sample_count+1}.png'
            
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original (Sex: {"M" if sex_val.item() else "F"})')
            axes[0].axis('off')
            
            axes[1].imshow(visualization)
            axes[1].set_title('Grad-CAM Attention')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"✓ Saved {sample_count} Grad-CAM visualizations to {GRADCAM_DIR}")

def save_evaluation_report(results, task='regression', save_prefix=''):
    """
    Save comprehensive evaluation report
    
    Args:
        results: Results dictionary
        task: 'regression' or 'classification'
        save_prefix: Prefix for saved file
    """
    report_file = METRICS_DIR / f'{save_prefix}_evaluation_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{save_prefix.upper()} EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Task: {task}\n")
        f.write(f"Number of samples: {results['metrics']['n_samples']}\n\n")
        
        f.write("METRICS:\n")
        f.write("-"*70 + "\n")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Evaluation report saved to {report_file}")

def visualize_learned_representations(model, data_loader, device=DEVICE, save_prefix=''):
    """
    Visualize learned feature embeddings using t-SNE
    Satisfies 'Visualization of learned representations' requirement
    """
    print("\nGenerating t-SNE visualization of learned representations...")
    from sklearn.manifold import TSNE
    
    model.eval()
    model = model.to(device)
    
    all_features = []
    all_ages = []
    all_sex = []
    
    # 1. Extract features (limit to 1000 samples to keep it fast)
    max_samples = 1000
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            if count >= max_samples:
                break
                
            images = batch['image'].to(device)
            sex = batch['sex'].to(device)
            bone_age = batch['bone_age'].numpy()
            
            # Use the extract_features method you already have in models.py
            features = model.extract_features(images, sex)
            
            all_features.append(features.cpu().numpy())
            all_ages.extend(bone_age)
            all_sex.extend(sex.cpu().numpy())
            
            count += images.size(0)
            
    # Concatenate
    X = np.vstack(all_features)
    y_age = np.array(all_ages)
    y_sex = np.array(['M' if s==1 else 'F' for s in all_sex])
    
    # 2. Run t-SNE
    print(f"  Running t-SNE on {len(X)} samples (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 3. Plot 1: Colored by Bone Age
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                          c=y_age, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Bone Age (Years)')
    plt.title(f'Learned Representations (t-SNE) by Age\n{save_prefix}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / f'{save_prefix}_tsne_age.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Plot 2: Colored by Sex
    plt.figure(figsize=(10, 8))
    for s in ['M', 'F']:
        mask = y_sex == s
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                    label=s, alpha=0.6)
    plt.legend()
    plt.title(f'Learned Representations (t-SNE) by Sex\n{save_prefix}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / f'{save_prefix}_tsne_sex.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE visualizations saved to {PLOTS_DIR}")


if __name__ == "__main__":
    from data_preprocessing import load_splits
    from dataset import create_data_loaders
    from models import BoneAgeRegressor, BoneAgeClassifier
    from config import BATCH_SIZE, NUM_WORKERS, set_seed, get_model_path
    from utils import load_model
    
    # Set seed
    set_seed()
    
    print("Testing evaluation pipeline...")
    
    # Load data
    _, val_df, _ = load_splits()
    _, val_loader = create_data_loaders(val_df, val_df, BATCH_SIZE, NUM_WORKERS)
    
    # Test with untrained model (just for testing pipeline)
    print("\n" + "="*70)
    print("Testing Regression Evaluation")
    print("="*70)
    
    model = BoneAgeRegressor()
    results = evaluate_regression(model, val_loader)
    generate_visualizations(results, task='regression', save_prefix='test_regression')
    perform_gender_analysis(results, task='regression', save_prefix='test_regression')
    
    print("\n✓ Evaluation test complete!")