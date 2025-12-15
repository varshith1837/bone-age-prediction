"""
Model architectures for Bone Age Prediction
"""

import torch
import torch.nn as nn
import timm

from config import (
    BACKBONE, PRETRAINED, BACKBONE_FEATURES, SEX_EMBEDDING_DIM,
    REGRESSION_HIDDEN_DIMS, REGRESSION_DROPOUT,
    CLASSIFICATION_HIDDEN_DIMS, CLASSIFICATION_DROPOUT,
    NUM_CLASSES
)


class BoneAgeRegressor(nn.Module):
    """
    CNN model for bone age regression
    Uses EfficientNet backbone + sex embedding + regression head
    """
    
    def __init__(self, backbone_name=BACKBONE, pretrained=PRETRAINED):
        super(BoneAgeRegressor, self).__init__()
        
        # Load backbone (EfficientNet-B0)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Sex embedding (2 sexes -> embedding_dim)
        self.sex_embedding = nn.Embedding(2, SEX_EMBEDDING_DIM)
        
        # Regression head
        layers = []
        input_dim = BACKBONE_FEATURES + SEX_EMBEDDING_DIM
        
        for hidden_dim, dropout in zip(REGRESSION_HIDDEN_DIMS, REGRESSION_DROPOUT):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.regressor = nn.Sequential(*layers)
        
    def forward(self, image, sex):
        """
        Forward pass
        
        Args:
            image: (batch_size, 3, H, W) - RGB images
            sex: (batch_size,) - sex labels (0=F, 1=M)
            
        Returns:
            (batch_size, 1) - predicted bone age in years
        """
        # Extract image features
        img_features = self.backbone(image)  # (batch_size, 1280)
        
        # Get sex embedding
        sex_embed = self.sex_embedding(sex)  # (batch_size, 32)
        
        # Concatenate features
        features = torch.cat([img_features, sex_embed], dim=1)  # (batch_size, 1312)
        
        # Predict bone age
        age = self.regressor(features)  # (batch_size, 1)
        
        return age.squeeze(1)  # (batch_size,)
    
    def extract_features(self, image, sex):
        """
        Extract features for XGBoost ensemble
        
        Returns:
            (batch_size, 1312) - concatenated features
        """
        with torch.no_grad():
            img_features = self.backbone(image)
            sex_embed = self.sex_embedding(sex)
            features = torch.cat([img_features, sex_embed], dim=1)
        return features


class BoneAgeClassifier(nn.Module):
    """
    CNN model for bone age classification
    Uses same backbone as regressor but with classification head
    """
    
    def __init__(self, backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_CLASSES):
        super(BoneAgeClassifier, self).__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # Sex embedding
        self.sex_embedding = nn.Embedding(2, SEX_EMBEDDING_DIM)
        
        # Classification head
        layers = []
        input_dim = BACKBONE_FEATURES + SEX_EMBEDDING_DIM
        
        for hidden_dim, dropout in zip(CLASSIFICATION_HIDDEN_DIMS, CLASSIFICATION_DROPOUT):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, image, sex):
        """
        Forward pass
        
        Args:
            image: (batch_size, 3, H, W)
            sex: (batch_size,)
            
        Returns:
            (batch_size, num_classes) - class logits
        """
        # Extract features
        img_features = self.backbone(image)
        sex_embed = self.sex_embedding(sex)
        features = torch.cat([img_features, sex_embed], dim=1)
        
        # Classify
        logits = self.classifier(features)
        
        return logits


class EnsembleModel:
    """
    Ensemble of CNN and XGBoost models
    """
    
    def __init__(self, cnn_model, xgb_model, cnn_weight=0.7, xgb_weight=0.3):
        self.cnn_model = cnn_model
        self.xgb_model = xgb_model
        self.cnn_weight = cnn_weight
        self.xgb_weight = xgb_weight
        
    def predict(self, image, sex, device='cpu'):
        """
        Make ensemble predictions
        
        Args:
            image: torch tensor (batch_size, 3, H, W)
            sex: torch tensor (batch_size,)
            device: device to run on
            
        Returns:
            np.ndarray: ensemble predictions
        """
        import numpy as np
        
        # CNN prediction
        self.cnn_model.eval()
        with torch.no_grad():
            image = image.to(device)
            sex = sex.to(device)
            
            cnn_pred = self.cnn_model(image, sex).cpu().numpy()
            
            # Extract features for XGBoost
            features = self.cnn_model.extract_features(image, sex).cpu().numpy()
        
        # XGBoost prediction
        xgb_pred = self.xgb_model.predict(features)
        
        # Ensemble
        ensemble_pred = (self.cnn_weight * cnn_pred + 
                        self.xgb_weight * xgb_pred)
        
        return ensemble_pred


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(3, 384, 384)):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input image size (C, H, W)
    """
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Backbone: {BACKBONE}")
    print(f"Input size: {input_size}")
    
    total_params = count_parameters(model)
    print(f"\nTrainable parameters: {total_params:,}")
    
    # Calculate model size in MB
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Model size: ~{param_size:.2f} MB")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_image = torch.randn(1, *input_size).to(device)
    dummy_sex = torch.tensor([0]).to(device)
    
    with torch.no_grad():
        output = model(dummy_image, dummy_sex)
    
    print(f"Output shape: {output.shape}")
    print("="*70 + "\n")


# Test models
if __name__ == "__main__":
    print("Testing BoneAgeRegressor...")
    regressor = BoneAgeRegressor()
    get_model_summary(regressor)
    
    print("\nTesting BoneAgeClassifier...")
    classifier = BoneAgeClassifier()
    get_model_summary(classifier)
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 384, 384)
    dummy_sex = torch.randint(0, 2, (batch_size,))
    
    print("\nTesting forward passes...")
    with torch.no_grad():
        reg_output = regressor(dummy_images, dummy_sex)
        cls_output = classifier(dummy_images, dummy_sex)
    
    print(f"Regressor output shape: {reg_output.shape} (expected: {batch_size})")
    print(f"Classifier output shape: {cls_output.shape} (expected: {batch_size}, {NUM_CLASSES})")
    
    # Test feature extraction
    features = regressor.extract_features(dummy_images, dummy_sex)
    print(f"Extracted features shape: {features.shape}")
    
    print("\n✓ Model tests complete!")