# Calibration Report

## Overview

This report assesses whether the model's predicted probabilities match observed churn rates. A well-calibrated model should predict 30% churn probability for a group where 30% actually churn.

## Brier Score

| Dataset | Brier Score |
|---------|-------------|
| Validation | 0.156 |
| Test | 0.152 |

Lower is better. A Brier score of 0.25 corresponds to random guessing for balanced classes.

## Calibration by Probability Bucket

| Predicted Probability | Actual Churn Rate | Count | Gap |
|-----------------------|-------------------|-------|-----|
| 0.0 - 0.1 | 0.05 | 142 | -0.02 |
| 0.1 - 0.2 | 0.14 | 198 | -0.01 |
| 0.2 - 0.3 | 0.22 | 187 | -0.03 |
| 0.3 - 0.4 | 0.31 | 156 | -0.04 |
| 0.4 - 0.5 | 0.42 | 124 | -0.03 |
| 0.5 - 0.6 | 0.54 | 98 | -0.01 |
| 0.6 - 0.7 | 0.68 | 72 | +0.03 |
| 0.7 - 0.8 | 0.76 | 48 | +0.01 |
| 0.8 - 0.9 | 0.85 | 24 | +0.00 |
| 0.9 - 1.0 | 0.94 | 8 | +0.02 |

## Observations

1. **Slight underconfidence in low-risk predictions**: Customers with predicted probability 0.2-0.4 churn slightly less than expected.

2. **Good calibration at extremes**: Very low (<0.1) and very high (>0.8) predictions are well-calibrated.

3. **Overall calibration is acceptable**: Maximum gap is 4 percentage points, within acceptable range for business decisions.

## Recommendations

- The model is sufficiently calibrated for ranking customers by risk.
- If exact probability estimates are critical (e.g., expected value calculations), consider applying isotonic regression calibration.
- Monitor calibration drift in production as customer behavior changes.
