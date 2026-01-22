"""Model training and evaluation module."""

from .training import (
    create_logistic_regression,
    create_random_forest,
    create_gradient_boosting,
    get_model_catalog,
    train_model,
    save_model,
    load_model
)

from .evaluation import (
    calculate_metrics,
    calculate_confusion_matrix,
    evaluate_model,
    compare_models,
    print_evaluation_report,
    get_roc_curve_data,
    get_precision_recall_curve_data,
    find_optimal_threshold,
    explain_metrics_business_context
)

from .explainability import (
    get_logistic_regression_coefficients,
    get_tree_feature_importance,
    get_feature_importance,
    explain_prediction,
    generate_plain_english_explanation,
    print_global_feature_importance
)

__all__ = [
    # Training
    'create_logistic_regression',
    'create_random_forest',
    'create_gradient_boosting',
    'get_model_catalog',
    'train_model',
    'save_model',
    'load_model',
    # Evaluation
    'calculate_metrics',
    'calculate_confusion_matrix',
    'evaluate_model',
    'compare_models',
    'print_evaluation_report',
    'get_roc_curve_data',
    'get_precision_recall_curve_data',
    'find_optimal_threshold',
    'explain_metrics_business_context',
    # Explainability
    'get_logistic_regression_coefficients',
    'get_tree_feature_importance',
    'get_feature_importance',
    'explain_prediction',
    'generate_plain_english_explanation',
    'print_global_feature_importance'
]
