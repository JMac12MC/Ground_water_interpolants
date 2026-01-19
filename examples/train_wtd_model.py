#!/usr/bin/env python3
"""
Train a Random Forest model for Water Table Depth (WTD) prediction.

This script follows a similar approach to HydroFrame-ML/high-res-WTD-static
for training a machine learning model to predict groundwater depth from
environmental covariates.

Usage:
    python train_wtd_model.py \
        --input wells_with_covariates.csv \
        --covariates elevation slope aspect twi \
        --target ground_water_level \
        --output wtd_rf_model.pkl \
        --test-split 0.2

Author: Groundwater Interpolation System
Date: 2024
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


def load_and_validate_data(filepath, covariate_cols, target_col):
    """Load well data and validate required columns exist."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Check required columns
    required_cols = covariate_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} wells")
    print(f"Columns: {list(df.columns)}")
    
    return df


def prepare_training_data(df, covariate_cols, target_col):
    """Prepare feature matrix X and target vector y, handling missing data."""
    print("\nPreparing training data...")
    
    # Extract features and target
    X = df[covariate_cols].values
    y = df[target_col].values
    
    # Check for missing data
    n_total = len(y)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    n_valid = mask.sum()
    n_dropped = n_total - n_valid
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Total samples: {n_total}")
    print(f"Valid samples: {n_valid}")
    print(f"Dropped (missing data): {n_dropped}")
    
    # Data statistics
    print(f"\nTarget variable ({target_col}) statistics:")
    print(f"  Mean: {y_clean.mean():.2f} m")
    print(f"  Std: {y_clean.std():.2f} m")
    print(f"  Min: {y_clean.min():.2f} m")
    print(f"  Max: {y_clean.max():.2f} m")
    
    return X_clean, y_clean


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, 
                        max_features='sqrt', random_state=42):
    """
    Train Random Forest model for WTD prediction.
    
    Parameters follow HydroFrame-ML approach:
    - n_estimators: Number of trees (default 100)
    - max_depth: Maximum tree depth (None = unlimited)
    - max_features: Features per split ('sqrt' ≈ 1/3 for many features)
    - random_state: For reproducibility
    """
    print("\nTraining Random Forest model...")
    print(f"Parameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  max_features: {max_features}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=1,
        bootstrap=True,
        max_samples=1.0,
        n_jobs=-1,  # Use all available cores
        random_state=random_state,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("Training complete!")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, covariate_names):
    """Evaluate model performance on train and test sets."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Training set performance
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("\nTraining Set Performance:")
    print(f"  RMSE: {train_rmse:.3f} m")
    print(f"  MAE:  {train_mae:.3f} m")
    print(f"  R²:   {train_r2:.4f}")
    
    # Test set performance
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.3f} m")
    print(f"  MAE:  {test_mae:.3f} m")
    print(f"  R²:   {test_r2:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': covariate_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (top 10):")
    print(importance_df.head(10).to_string(index=False))
    
    return {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'feature_importance': importance_df
    }


def plot_results(y_test, y_test_pred, metrics, output_dir):
    """Generate diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Observed vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, y_test_pred, alpha=0.5, s=10)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='1:1 line')
    axes[0].set_xlabel('Observed WTD (m)')
    axes[0].set_ylabel('Predicted WTD (m)')
    axes[0].set_title(f"Test Set: R² = {metrics['test_r2']:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_test_pred
    axes[1].scatter(y_test_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted WTD (m)')
    axes[1].set_ylabel('Residual (m)')
    axes[1].set_title(f"Residuals: RMSE = {metrics['test_rmse']:.3f} m")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_diagnostics.png', dpi=150)
    print(f"\nDiagnostic plot saved to {output_dir / 'model_diagnostics.png'}")
    
    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df = metrics['feature_importance'].head(15)
    ax.barh(range(len(importance_df)), importance_df['importance'])
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Most Important Features')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    print(f"Feature importance plot saved to {output_dir / 'feature_importance.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Random Forest model for WTD prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', required=True,
                        help='Input CSV file with wells and covariates')
    parser.add_argument('--covariates', nargs='+', required=True,
                        help='Names of covariate columns to use as features')
    parser.add_argument('--target', default='ground_water_level',
                        help='Name of target column (default: ground_water_level)')
    parser.add_argument('--output', required=True,
                        help='Output path for trained model (.pkl)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees (default: 100)')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum tree depth (default: None)')
    parser.add_argument('--max-features', default='sqrt',
                        help='Max features per split (default: sqrt)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--plots-dir', default='model_outputs',
                        help='Directory for output plots (default: model_outputs)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("WATER TABLE DEPTH MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_and_validate_data(args.input, args.covariates, args.target)
    
    # Prepare training data
    X, y = prepare_training_data(df, args.covariates, args.target)
    
    # Split into train/test
    print(f"\nSplitting data (test_size={args.test_split})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.random_seed
    )
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    
    # Train model
    model = train_random_forest(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        random_state=args.random_seed
    )
    
    # Evaluate
    metrics = evaluate_model(
        model, X_train, y_train, X_test, y_test, args.covariates
    )
    
    # Save model
    print(f"\nSaving model to {args.output}...")
    joblib.dump(model, args.output)
    
    # Save metrics
    metrics_file = Path(args.output).with_suffix('.metrics.json')
    import json
    with open(metrics_file, 'w') as f:
        json.dump({
            'train_rmse': float(metrics['train_rmse']),
            'train_r2': float(metrics['train_r2']),
            'test_rmse': float(metrics['test_rmse']),
            'test_r2': float(metrics['test_r2']),
            'n_features': len(args.covariates),
            'n_train': len(y_train),
            'n_test': len(y_test)
        }, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    # Generate plots
    y_test_pred = model.predict(X_test)
    plot_results(y_test, y_test_pred, metrics, args.plots_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {args.output}")
    print(f"Test RMSE: {metrics['test_rmse']:.3f} m")
    print(f"Test R²: {metrics['test_r2']:.4f}")


if __name__ == '__main__':
    main()
