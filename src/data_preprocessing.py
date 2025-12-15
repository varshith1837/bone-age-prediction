"""
Data preprocessing and splitting for Bone Age Prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TRAIN_CSV, TRAIN_IMG_DIR, SPLITS_DIR, ANALYSIS_DIR,
    TRAIN_RATIO, VAL_RATIO, HOLDOUT_RATIO, RANDOM_SEED,
    AGE_BINS, AGE_LABELS, months_to_years, get_age_bin
)


def load_and_standardize_data():
    """
    Load training CSV and standardize column names and formats
    
    Returns:
        pd.DataFrame: Standardized dataframe
    """
    print("\n" + "="*70)
    print("LOADING AND STANDARDIZING DATA")
    print("="*70)
    
    # Load CSV
    df = pd.read_csv(TRAIN_CSV)
    print(f"✓ Loaded {len(df)} records from {TRAIN_CSV.name}")
    print(f"  Original columns: {list(df.columns)}")
    
    # Standardize column names
    df = df.rename(columns={
        'id': 'image_id',
        'boneage': 'bone_age_months',
        'male': 'is_male'
    })
    
    # Convert boolean to binary and create sex label
    df['sex_binary'] = df['is_male'].astype(int)
    df['sex'] = df['is_male'].map({True: 'M', False: 'F'})
    
    # Add bone age in years
    df['bone_age_years'] = df['bone_age_months'].apply(months_to_years)
    
    # Add age category for classification
    df['age_category'] = df['bone_age_months'].apply(get_age_bin)
    df['age_category_name'] = df['age_category'].map(lambda x: AGE_LABELS[x])
    
    # Verify all images exist
    df['image_path'] = df['image_id'].apply(lambda x: TRAIN_IMG_DIR / f"{x}.png")
    df['exists'] = df['image_path'].apply(lambda x: x.exists())
    
    missing_count = (~df['exists']).sum()
    if missing_count > 0:
        print(f"⚠️  Warning: {missing_count} images not found")
        df = df[df['exists']].copy()
        print(f"  Removed missing images. Remaining: {len(df)} records")
    else:
        print(f"✓ All {len(df)} images found")
    
    df = df.drop(columns=['exists'])
    
    print(f"\n✓ Standardized columns: {list(df.columns)}")
    print(f"  - bone_age_years: {df['bone_age_years'].min():.2f} to {df['bone_age_years'].max():.2f} years")
    print(f"  - sex distribution: {df['sex'].value_counts().to_dict()}")
    print(f"  - age categories: {df['age_category_name'].value_counts().to_dict()}")
    
    return df


def perform_eda(df):
    """
    Perform exploratory data analysis and save visualizations
    
    Args:
        df: Standardized dataframe
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Basic statistics
    stats = {
        'total_samples': len(df),
        'age_mean': df['bone_age_years'].mean(),
        'age_std': df['bone_age_years'].std(),
        'age_min': df['bone_age_years'].min(),
        'age_max': df['bone_age_years'].max(),
        'male_count': (df['sex'] == 'M').sum(),
        'female_count': (df['sex'] == 'F').sum(),
        'male_pct': (df['sex'] == 'M').mean() * 100,
    }
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Age range: {stats['age_min']:.2f} - {stats['age_max']:.2f} years")
    print(f"  Age mean ± std: {stats['age_mean']:.2f} ± {stats['age_std']:.2f} years")
    print(f"  Male: {stats['male_count']} ({stats['male_pct']:.1f}%)")
    print(f"  Female: {stats['female_count']} ({100-stats['male_pct']:.1f}%)")
    
    # Save statistics
    stats_file = ANALYSIS_DIR / 'statistics.txt'
    with open(stats_file, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Saved statistics to {stats_file}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Age distribution
    axes[0, 0].hist(df['bone_age_years'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['bone_age_years'].mean(), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["bone_age_years"].mean():.2f}')
    axes[0, 0].set_xlabel('Bone Age (years)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Bone Age Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gender distribution
    sex_counts = df['sex'].value_counts()
    axes[0, 1].bar(sex_counts.index, sex_counts.values, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Sex', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Gender Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Age by gender
    df.boxplot(column='bone_age_years', by='sex', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Sex', fontsize=11)
    axes[1, 0].set_ylabel('Bone Age (years)', fontsize=11)
    axes[1, 0].set_title('Bone Age by Gender', fontsize=13, fontweight='bold')
    plt.sca(axes[1, 0])
    plt.xticks([1, 2], ['F', 'M'])
    
    # 4. Age category distribution
    category_counts = df['age_category_name'].value_counts().reindex(AGE_LABELS)
    axes[1, 1].bar(range(len(category_counts)), category_counts.values, 
                   tick_label=AGE_LABELS, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Age Category', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Age Category Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    eda_plot_path = ANALYSIS_DIR / 'eda_analysis.png'
    plt.savefig(eda_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved EDA visualizations to {eda_plot_path}")
    plt.close()
    
    # Age correlation with sex
    print(f"\nAge statistics by gender:")
    for sex in ['M', 'F']:
        subset = df[df['sex'] == sex]
        print(f"  {sex}: mean={subset['bone_age_years'].mean():.2f}, "
              f"std={subset['bone_age_years'].std():.2f}")


def create_data_splits(df):
    """
    Split data into train, validation, and holdout sets with stratification
    
    Args:
        df: Standardized dataframe
        
    Returns:
        tuple: (train_df, val_df, holdout_df)
    """
    print("\n" + "="*70)
    print("CREATING DATA SPLITS")
    print("="*70)
    
    # Create stratification column (age_category + sex)
    df['stratify_col'] = df['age_category'].astype(str) + '_' + df['sex']
    
    # First split: separate holdout set (15%)
    train_val_df, holdout_df = train_test_split(
        df,
        test_size=HOLDOUT_RATIO,
        stratify=df['stratify_col'],
        random_state=RANDOM_SEED
    )
    
    # Second split: split remaining into train (70%) and val (15%)
    # Calculate val size relative to train_val
    val_size_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        stratify=train_val_df['stratify_col'],
        random_state=RANDOM_SEED
    )
    
    # Remove stratification column
    train_df = train_df.drop(columns=['stratify_col'])
    val_df = val_df.drop(columns=['stratify_col'])
    holdout_df = holdout_df.drop(columns=['stratify_col'])
    
    print(f"✓ Created splits:")
    print(f"  Train:    {len(train_df):5d} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:      {len(val_df):5d} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Holdout:  {len(holdout_df):5d} samples ({len(holdout_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print(f"\nVerifying stratification:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Holdout', holdout_df)]:
        print(f"\n  {split_name}:")
        print(f"    Age categories: {split_df['age_category_name'].value_counts().to_dict()}")
        print(f"    Sex: {split_df['sex'].value_counts().to_dict()}")
    
    # Save splits
    train_df.to_csv(SPLITS_DIR / 'train_split.csv', index=False)
    val_df.to_csv(SPLITS_DIR / 'val_split.csv', index=False)
    holdout_df.to_csv(SPLITS_DIR / 'holdout_split.csv', index=False)
    
    print(f"\n✓ Saved splits to {SPLITS_DIR}")
    
    return train_df, val_df, holdout_df


def load_splits():
    """
    Load previously created data splits
    
    Returns:
        tuple: (train_df, val_df, holdout_df)
    """
    train_df = pd.read_csv(SPLITS_DIR / 'train_split.csv')
    val_df = pd.read_csv(SPLITS_DIR / 'val_split.csv')
    holdout_df = pd.read_csv(SPLITS_DIR / 'holdout_split.csv')
    
    print(f"✓ Loaded splits:")
    print(f"  Train:   {len(train_df)} samples")
    print(f"  Val:     {len(val_df)} samples")
    print(f"  Holdout: {len(holdout_df)} samples")
    
    return train_df, val_df, holdout_df


def prepare_data():
    """
    Main function to prepare all data
    Loads, standardizes, analyzes, and splits data
    
    Returns:
        tuple: (train_df, val_df, holdout_df)
    """
    # Check if splits already exist
    if (SPLITS_DIR / 'train_split.csv').exists():
        print("Found existing data splits.")
        response = input("Load existing splits? (y/n): ").lower()
        if response == 'y':
            return load_splits()
    
    # Full pipeline
    df = load_and_standardize_data()
    perform_eda(df)
    train_df, val_df, holdout_df = create_data_splits(df)
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    
    return train_df, val_df, holdout_df


if __name__ == "__main__":
    # Run data preparation
    train_df, val_df, holdout_df = prepare_data()