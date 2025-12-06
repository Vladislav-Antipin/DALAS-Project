"""
Model Training Module for DALAS Drug Repurposing Pipeline.

This module handles:
1. Data preparation (train/test split)
2. Model training (RandomForest, LogisticRegression)
3. Evaluation metrics
4. Feature importance analysis
5. Model persistence

Design Decisions (documented):
------------------------------

TRAIN/TEST SPLIT:
- 80/20 split with stratification on label
- Random state fixed for reproducibility
- Stratification ensures balanced class distribution in both sets

PREPROCESSING:
- Numeric features: Mean imputation + StandardScaler
- This matches the notebook approach
- Missing values are imputed before scaling to avoid NaN propagation

MODELS:
- RandomForest: Good for feature importance, handles non-linearity
  - n_estimators=300 (matching notebook)
  - class_weight='balanced' to handle imbalanced classes
- LogisticRegression: Interpretable baseline
  - class_weight='balanced'
  - max_iter=500 for convergence

EVALUATION:
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrix for detailed error analysis
- Focus on AUC as primary metric (handles class imbalance well)

FEATURE IMPORTANCE:
- RF: Built-in feature_importances_
- LogReg: Absolute coefficient values
- Both normalized and sorted for comparison
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from .config import PROCESSED_DATA_DIR, RESULTS_DIR


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(
    merged_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        merged_df: ML-ready dataset with 'label' column
        test_size: Fraction of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
        
    Design choices:
        - Stratified split ensures class balance in both sets
        - Label column is separated from features
    """
    # Separate features and labels
    if "label" not in merged_df.columns:
        raise ValueError("Dataset must have 'label' column")
    
    X = merged_df.drop(columns=["label"])
    y = merged_df["label"]
    
    # Convert boolean to int for sklearn
    y = y.astype(int)
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print("Data split:")
    print(f"  Training: {len(X_train)} samples ({y_train.mean():.1%} positive)")
    print(f"  Testing:  {len(X_test)} samples ({y_test.mean():.1%} positive)")
    
    return X_train, X_test, y_train, y_test


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline for features.
    
    Args:
        X: Feature DataFrame to determine column types
        
    Returns:
        ColumnTransformer for preprocessing
        
    Design choices:
        - Numeric: Mean imputation + StandardScaler
        - This handles missing values and normalizes features
    """
    # Identify numeric columns
    num_cols = X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
        ],
        remainder="drop"  # Drop any remaining non-numeric columns
    )
    
    return preprocessor


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a RandomForest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Preprocessing pipeline
        n_estimators: Number of trees (default: 300, matching notebook)
        random_state: Random seed
        
    Returns:
        Trained sklearn Pipeline
        
    Design choices:
        - class_weight='balanced' handles class imbalance
        - n_estimators=300 for good performance without overfitting
        - n_jobs=-1 for parallel training
    """
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ))
    ])
    
    print("Training RandomForest...")
    model.fit(X_train, y_train)
    
    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a Logistic Regression classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Preprocessing pipeline
        max_iter: Maximum iterations for convergence
        random_state: Random seed
        
    Returns:
        Trained sklearn Pipeline
        
    Design choices:
        - class_weight='balanced' handles class imbalance
        - max_iter=500 ensures convergence
        - L2 regularization (default) prevents overfitting
    """
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        ))
    ])
    
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn Pipeline
        X_test: Test features
        y_test: Test labels
        model_name: Name for display
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    return metrics


def get_feature_importance(
    model: Pipeline,
    feature_names: list,
    model_type: str = "rf",
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained sklearn Pipeline
        feature_names: List of feature names
        model_type: "rf" for RandomForest, "logreg" for LogisticRegression
        
    Returns:
        DataFrame with feature importances sorted by importance
    """
    classifier = model.named_steps["classifier"]
    
    if model_type == "rf":
        importances = classifier.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
    elif model_type == "logreg":
        coef = classifier.coef_[0]
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef,
            "importance": np.abs(coef),
        })
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_model_training(
    force: bool = False,
    cross_validate: bool = True,
) -> Dict[str, Any]:
    """
    Run complete model training pipeline.
    
    Args:
        force: If True, retrain even if models exist
        cross_validate: If True, run cross-validation
        
    Returns:
        Dictionary with models, metrics, and feature importances
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Check for existing models
    rf_path = RESULTS_DIR / "random_forest_model.pkl"
    lr_path = RESULTS_DIR / "logistic_regression_model.pkl"
    
    if rf_path.exists() and lr_path.exists() and not force:
        print("\nLoading existing models...")
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
        with open(lr_path, "rb") as f:
            lr_model = pickle.load(f)
        print("  Models loaded from cache. Use --force to retrain.")
        
        # Load metrics if available
        metrics_path = RESULTS_DIR / "model_metrics.pkl"
        if metrics_path.exists():
            with open(metrics_path, "rb") as f:
                results = pickle.load(f)
            return results
    
    # Load data
    print("\n--- Loading Data ---")
    merged_path = PROCESSED_DATA_DIR / "merged_df.pkl"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {merged_path}. "
            "Run the merge step first."
        )
    
    with open(merged_path, "rb") as f:
        merged_df = pickle.load(f)
    
    print(f"  Loaded {len(merged_df)} samples with {merged_df.shape[1]-1} features")
    
    # Prepare data
    print("\n--- Preparing Data ---")
    X_train, X_test, y_train, y_test = prepare_data(merged_df)
    
    # Get feature names (after preprocessing will use these)
    feature_names = X_train.select_dtypes(
        include=["float64", "int64", "float32", "int32"]
    ).columns.tolist()
    
    # Note: preprocessor is created fresh for each model to avoid state issues
    
    # Cross-validation (optional)
    if cross_validate:
        print("\n--- Cross-Validation ---")
        # Quick CV on RF
        rf_cv = Pipeline(steps=[
            ("preprocess", create_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,  # Fewer trees for CV speed
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ])
        cv_scores = cross_val_score(rf_cv, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train models
    print("\n--- Training Models ---")
    rf_model = train_random_forest(X_train, y_train, create_preprocessor(X_train))
    lr_model = train_logistic_regression(X_train, y_train, create_preprocessor(X_train))
    
    # Evaluate models
    print("\n--- Evaluating Models ---")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")
    
    # Feature importance
    print("\n--- Feature Importance ---")
    rf_importance = get_feature_importance(rf_model, feature_names, "rf")
    lr_importance = get_feature_importance(lr_model, feature_names, "logreg")
    
    print("Top 10 Features (RandomForest):")
    for i, row in rf_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    print("\nTop 10 Features (LogisticRegression):")
    for i, row in lr_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:30s} {row['coefficient']:.4f}")
    
    # Save models and results
    print("\n--- Saving Results ---")
    
    # Save models
    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"  Saved RandomForest to {rf_path}")
    
    with open(lr_path, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"  Saved LogisticRegression to {lr_path}")
    
    # Save feature importances
    rf_importance.to_csv(RESULTS_DIR / "rf_feature_importance.csv", index=False)
    lr_importance.to_csv(RESULTS_DIR / "lr_feature_importance.csv", index=False)
    print(f"  Saved feature importances to {RESULTS_DIR}")
    
    # Compile results
    results = {
        "models": {
            "random_forest": rf_model,
            "logistic_regression": lr_model,
        },
        "metrics": {
            "random_forest": rf_metrics,
            "logistic_regression": lr_metrics,
        },
        "feature_importance": {
            "random_forest": rf_importance,
            "logistic_regression": lr_importance,
        },
        "feature_names": feature_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save full results
    metrics_path = RESULTS_DIR / "model_metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved all metrics to {metrics_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Model Comparison:")
    print(f"  {'Model':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    print(f"  {'RandomForest':<20} {rf_metrics['accuracy']:>10.4f} {rf_metrics['auc_roc']:>10.4f} {rf_metrics['f1']:>10.4f}")
    print(f"  {'LogisticRegression':<20} {lr_metrics['accuracy']:>10.4f} {lr_metrics['auc_roc']:>10.4f} {lr_metrics['f1']:>10.4f}")
    
    return results


def predict_new_pairs(
    drug_disease_df: pd.DataFrame,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Predict repurposing probability for new drug-disease pairs.
    
    Args:
        drug_disease_df: DataFrame with same features as training data
        model_path: Path to saved model (default: results/random_forest_model.pkl)
        
    Returns:
        DataFrame with predictions and probabilities
    """
    if model_path is None:
        model_path = RESULTS_DIR / "random_forest_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Predict
    predictions = model.predict(drug_disease_df)
    probabilities = model.predict_proba(drug_disease_df)[:, 1]
    
    # Add to DataFrame
    result = drug_disease_df.copy()
    result["predicted_success"] = predictions
    result["success_probability"] = probabilities
    
    return result
