"""
Model explainability utilities.

This module provides functions for understanding model predictions
and generating human-readable explanations.

Explainability is critical for:
1. Building trust with stakeholders
2. Debugging model behavior
3. Regulatory compliance
4. Identifying potential biases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_logistic_regression_coefficients(
    model: LogisticRegression,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract and format logistic regression coefficients.
    
    Coefficients in logistic regression represent log-odds:
    - Positive coefficient: Feature increases churn probability
    - Negative coefficient: Feature decreases churn probability
    - Magnitude indicates strength of effect
    
    Args:
        model: Fitted LogisticRegression model.
        feature_names: List of feature names matching model input.
        
    Returns:
        DataFrame with features sorted by absolute coefficient value.
    """
    coefficients = model.coef_[0]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'direction': ['Increases Churn' if c > 0 else 'Decreases Churn' 
                      for c in coefficients]
    })
    
    df = df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    
    return df


def get_tree_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    For Random Forest and Gradient Boosting, importance is based on
    how much each feature reduces impurity (Gini or entropy) across all trees.
    
    Args:
        model: Fitted tree-based model with feature_importances_ attribute.
        feature_names: List of feature names.
        
    Returns:
        DataFrame with features sorted by importance.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['cumulative_importance'] = df['importance'].cumsum()
    
    return df


def get_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from any supported model type.
    
    Args:
        model: Fitted model.
        feature_names: List of feature names.
        
    Returns:
        DataFrame with feature importance.
    """
    if isinstance(model, LogisticRegression):
        df = get_logistic_regression_coefficients(model, feature_names)
        # Normalize coefficients to importance scale
        df['importance'] = df['abs_coefficient'] / df['abs_coefficient'].sum()
        return df[['feature', 'importance', 'coefficient', 'direction']]
    
    elif hasattr(model, 'feature_importances_'):
        return get_tree_feature_importance(model, feature_names)
    
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def explain_prediction(
    model,
    preprocessor,
    sample: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Generate an explanation for a single prediction.
    
    Args:
        model: Fitted model.
        preprocessor: Fitted preprocessor.
        sample: Single row DataFrame with raw features.
        feature_names: List of processed feature names.
        top_n: Number of top features to include in explanation.
        
    Returns:
        Dictionary containing prediction details and explanation.
    """
    # Preprocess the sample
    X = preprocessor.transform(sample)
    
    # Get prediction
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)
    
    # Get feature importance
    importance_df = get_feature_importance(model, feature_names)
    top_features = importance_df.head(top_n)
    
    # For logistic regression, we can show actual contribution
    if isinstance(model, LogisticRegression):
        contributions = X[0] * model.coef_[0]
        contrib_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False)
        top_contributors = contrib_df.head(top_n)
    else:
        top_contributors = None
    
    return {
        'churn_probability': float(prob),
        'prediction': 'Will Churn' if pred == 1 else 'Will Stay',
        'confidence': float(max(prob, 1 - prob)),
        'risk_level': _get_risk_level(prob),
        'top_features': top_features.to_dict('records'),
        'top_contributors': top_contributors.to_dict('records') if top_contributors is not None else None
    }


def _get_risk_level(probability: float) -> str:
    """Categorize churn probability into risk levels."""
    if probability >= 0.7:
        return 'High Risk'
    elif probability >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'


def generate_plain_english_explanation(
    explanation: Dict[str, Any],
    customer_data: Optional[Dict] = None
) -> str:
    """
    Generate a plain English explanation of a prediction.
    
    Args:
        explanation: Output from explain_prediction().
        customer_data: Optional raw customer data for context.
        
    Returns:
        Human-readable explanation string.
    """
    prob = explanation['churn_probability']
    risk = explanation['risk_level']
    
    text = f"""
CHURN PREDICTION EXPLANATION
============================

Prediction: {explanation['prediction']}
Churn Probability: {prob:.1%}
Risk Level: {risk}
Confidence: {explanation['confidence']:.1%}

"""
    
    if explanation.get('top_contributors'):
        text += "KEY FACTORS INFLUENCING THIS PREDICTION:\n"
        text += "-" * 40 + "\n"
        
        for i, contrib in enumerate(explanation['top_contributors'][:5], 1):
            feature = contrib['feature']
            value = contrib['contribution']
            direction = "increases" if value > 0 else "decreases"
            
            # Clean up feature name for readability
            clean_name = feature.replace('numeric__', '').replace('binary__', '').replace('categorical__', '')
            
            text += f"{i}. {clean_name}: {direction} churn risk\n"
    
    text += f"""
INTERPRETATION:
- A {prob:.0%} probability means that among 100 similar customers,
  approximately {int(prob*100)} would be expected to churn.
- This customer is classified as '{risk}'.
"""
    
    if risk == 'High Risk':
        text += "\nRECOMMENDED ACTION: Prioritize for retention campaign.\n"
    elif risk == 'Medium Risk':
        text += "\nRECOMMENDED ACTION: Monitor closely and consider outreach.\n"
    else:
        text += "\nRECOMMENDED ACTION: Standard service, no immediate action needed.\n"
    
    return text


def print_global_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15
) -> None:
    """
    Print a formatted global feature importance report.
    
    Args:
        importance_df: DataFrame from get_feature_importance().
        top_n: Number of features to display.
    """
    print("=" * 60)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 60)
    print(f"\nTop {top_n} most important features:\n")
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12}")
    print("-" * 60)
    
    for i, row in importance_df.head(top_n).iterrows():
        # Clean feature name
        name = row['feature']
        name = name.replace('numeric__', '[NUM] ')
        name = name.replace('binary__', '[BIN] ')
        name = name.replace('categorical__', '[CAT] ')
        
        # Truncate if too long
        if len(name) > 38:
            name = name[:35] + "..."
        
        importance = row['importance']
        bar = "█" * int(importance * 50)
        
        print(f"{i+1:<6} {name:<40} {importance:>6.1%} {bar}")
    
    # Print cumulative importance
    if 'cumulative_importance' in importance_df.columns:
        top_n_cumulative = importance_df.head(top_n)['importance'].sum()
        print(f"\nTop {top_n} features explain {top_n_cumulative:.1%} of model decisions.")
