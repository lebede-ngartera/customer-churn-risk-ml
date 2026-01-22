# Error Analysis Report

## Overview

This report examines patterns in the model's errors to identify systematic weaknesses and potential improvements.

## Error Distribution (Test Set, Threshold = 0.5)

| Error Type | Count | Percentage |
|------------|-------|------------|
| True Positives | 227 | 21.5% |
| True Negatives | 557 | 52.7% |
| False Positives | 220 | 20.8% |
| False Negatives | 53 | 5.0% |

## False Negative Analysis (Missed Churners)

These are customers who churned but the model predicted they would stay.

### Common Characteristics

| Feature | False Negatives | Overall Churners |
|---------|-----------------|------------------|
| Avg tenure (months) | 38.2 | 17.8 |
| Contract: Two year | 42% | 11% |
| Internet: Fiber optic | 31% | 69% |
| Payment: Auto | 58% | 24% |

### Insights

1. **Long-tenure customers**: The model underestimates churn risk for customers with 3+ years tenure. These may be "sudden churners" triggered by life events rather than gradual dissatisfaction.

2. **Two-year contracts**: Customers on long contracts who churn tend to do so when contracts expire. The model lacks contract end-date features.

3. **Automatic payment**: Customers on auto-pay who churn may be less price-sensitive and leave for service quality reasons not captured in the features.

## False Positive Analysis (False Alarms)

These are customers predicted to churn but who actually stayed.

### Common Characteristics

| Feature | False Positives | Overall Non-Churners |
|---------|-----------------|----------------------|
| Avg tenure (months) | 8.4 | 37.6 |
| Contract: Month-to-month | 89% | 43% |
| Internet: Fiber optic | 78% | 41% |
| Payment: Electronic check | 64% | 31% |

### Insights

1. **New fiber customers**: Month-to-month fiber customers are high-risk on paper but many stay. These may be price-sensitive customers who found acceptable value.

2. **Electronic check pattern**: This payment method correlates with churn but has many false positives. Consider whether this reflects actual causation or demographic confounding.

## Recommendations

1. **Feature engineering**: Add contract end-date proximity, payment consistency (late payments), and support ticket history if available.

2. **Segmented models**: Consider separate models for new (<12 months) vs. established (>24 months) customers.

3. **Threshold adjustment by segment**: Use lower threshold for high-tenure customers to catch sudden churners.

4. **Investigation needed**: The electronic check correlation warrants business review—is it causal or confounded with other factors?

## Error Examples

### False Negative Example
- Tenure: 45 months
- Contract: Two year
- Services: DSL, basic package
- Monthly charges: $45
- Predicted probability: 0.18
- **Actual**: Churned

Hypothesis: Long-term, low-engagement customer whose contract expired.

### False Positive Example
- Tenure: 3 months
- Contract: Month-to-month
- Services: Fiber, streaming bundle
- Monthly charges: $95
- Predicted probability: 0.72
- **Actual**: Stayed

Hypothesis: High-value customer satisfied with premium services despite flexible contract.
