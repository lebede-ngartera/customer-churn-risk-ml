"""
Main training script for customer churn prediction models.

This script:
1. Loads and preprocesses data
2. Trains multiple models
3. Evaluates on validation set
4. Saves the best model

Usage:
    python src/models/train.py
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import (
    prepare_data,
    create_splits,
    get_split_summary,
    print_split_summary,
    validate_no_leakage
)
from src.features import (
    create_preprocessor,
    get_feature_names,
    FEATURE_COLS,
    TARGET_COL
)
from src.models import (
    create_logistic_regression,
    create_random_forest,
    create_gradient_boosting,
    train_model,
    evaluate_model,
    compare_models,
    print_evaluation_report,
    save_model
)


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 70)
    
    # =========================================================================
    # 1. LOAD AND PREPARE DATA
    # =========================================================================
    print("\n[1/5] Loading and preparing data...")
    
    data_path = project_root / "data" / "raw" / "telco_customer_churn.csv"
    df = prepare_data(str(data_path))
    print(f"  Loaded {len(df):,} customers")
    
    # =========================================================================
    # 2. CREATE TRAIN/VALIDATION/TEST SPLITS
    # =========================================================================
    print("\n[2/5] Creating data splits...")
    
    train_df, val_df, test_df = create_splits(df, random_state=42)
    
    # Validate no data leakage
    validate_no_leakage(
        train_df['customerID'],
        val_df['customerID'],
        test_df['customerID']
    )
    print("  ✓ No customer overlap between splits")
    
    # Print summary
    summary = get_split_summary(train_df, val_df, test_df)
    print_split_summary(summary)
    
    # =========================================================================
    # 3. PREPROCESS FEATURES
    # =========================================================================
    print("\n[3/5] Preprocessing features...")
    
    # Separate features and target
    X_train = train_df[FEATURE_COLS]
    X_val = val_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    
    # Create and fit preprocessor on training data ONLY
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    # X_test_processed saved for later (Phase 5: Final Evaluation)
    
    feature_names = get_feature_names(preprocessor)
    print(f"  Features: {len(FEATURE_COLS)} raw → {X_train_processed.shape[1]} processed")
    
    # =========================================================================
    # 4. TRAIN MODELS
    # =========================================================================
    print("\n[4/5] Training models...")
    
    models = {
        'Logistic Regression': create_logistic_regression(),
        'Random Forest': create_random_forest(),
        'Gradient Boosting': create_gradient_boosting()
    }
    
    trained_models = {}
    training_times = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()
        
        trained_model = train_model(model, X_train_processed, y_train)
        
        elapsed = time.time() - start_time
        training_times[name] = elapsed
        trained_models[name] = trained_model
        
        print(f"    ✓ Completed in {elapsed:.2f} seconds")
    
    # =========================================================================
    # 5. EVALUATE ON VALIDATION SET
    # =========================================================================
    print("\n[5/5] Evaluating models on validation set...")
    
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_val_processed, y_val)
    
    # Print individual reports
    for name, result in results.items():
        print()
        print_evaluation_report(name, result)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 70)
    
    comparison_df = compare_models(results)
    print("\n" + comparison_df.to_string())
    
    # Add training times
    print("\nTraining Times:")
    for name, time_sec in training_times.items():
        print(f"  {name}: {time_sec:.2f}s")
    
    # =========================================================================
    # DETERMINE BEST MODEL
    # =========================================================================
    best_model_name = comparison_df.index[0]  # Already sorted by ROC-AUC
    best_roc_auc = comparison_df.loc[best_model_name, 'roc_auc']
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\nBest model by ROC-AUC: {best_model_name} ({best_roc_auc:.4f})")
    
    # =========================================================================
    # SAVE BEST MODEL
    # =========================================================================
    best_model = trained_models[best_model_name]
    model_path = project_root / "models" / "best_model.joblib"
    save_model(best_model, str(model_path), preprocessor)
    print(f"\nModel saved to: {model_path}")
    
    # Also save logistic regression for interpretability analysis
    lr_path = project_root / "models" / "logistic_regression.joblib"
    save_model(trained_models['Logistic Regression'], str(lr_path), preprocessor)
    print(f"Logistic Regression saved to: {lr_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # =========================================================================
    # BIAS-VARIANCE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("=" * 70)
    
    # Calculate train vs validation performance gap
    print("\nTrain vs Validation Performance (ROC-AUC):")
    print("-" * 50)
    
    for name, model in trained_models.items():
        train_result = evaluate_model(model, X_train_processed, y_train)
        val_result = results[name]
        
        train_auc = train_result['metrics']['roc_auc']
        val_auc = val_result['metrics']['roc_auc']
        gap = train_auc - val_auc
        
        print(f"\n{name}:")
        print(f"  Train ROC-AUC:      {train_auc:.4f}")
        print(f"  Validation ROC-AUC: {val_auc:.4f}")
        print(f"  Gap:                {gap:.4f}", end="")
        
        if gap > 0.05:
            print(" ⚠️  (potential overfitting)")
        elif gap < 0.01:
            print(" ✓ (good generalization)")
        else:
            print(" (acceptable)")
    
    print("\n" + "-" * 50)
    print("""
INTERPRETATION:
- Large gap (>5%) suggests overfitting (high variance)
- Small gap (<1%) with low performance suggests underfitting (high bias)
- Logistic Regression: Simple model, prone to underfitting but stable
- Random Forest: Bagging reduces variance, good balance
- Gradient Boosting: Complex model, prone to overfitting if not regularized
""")
    
    return trained_models, results, preprocessor


if __name__ == "__main__":
    main()
