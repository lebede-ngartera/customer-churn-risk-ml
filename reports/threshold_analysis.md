# Threshold Analysis Report

## Overview

This report analyzes the precision-recall tradeoff at different classification thresholds for the customer churn model.

## Default Threshold (0.5)

| Metric | Value |
|--------|-------|
| Precision | 0.51 |
| Recall | 0.81 |
| F1 | 0.62 |

## Threshold Sensitivity

| Threshold | Precision | Recall | F1 | Churners Caught | False Alarms |
|-----------|-----------|--------|-----|-----------------|--------------|
| 0.3 | 0.38 | 0.92 | 0.54 | 258/280 | 422 |
| 0.4 | 0.45 | 0.87 | 0.59 | 244/280 | 298 |
| 0.5 | 0.51 | 0.81 | 0.62 | 227/280 | 220 |
| 0.6 | 0.58 | 0.71 | 0.64 | 199/280 | 144 |
| 0.7 | 0.66 | 0.58 | 0.62 | 162/280 | 84 |

## Recommendations

Given the 10:1 cost asymmetry (false negative = $500, false positive = $50):

- **Conservative threshold (0.3-0.4)**: Catches more churners but generates more false alarms. Suitable when retention capacity is high.
- **Balanced threshold (0.5)**: Default sklearn behavior. Reasonable starting point.
- **Aggressive threshold (0.6-0.7)**: Higher precision, fewer wasted offers. Use when retention resources are limited.

## Business Decision

The optimal threshold depends on:
1. Retention team capacity
2. Cost per retention offer
3. Customer lifetime value distribution

A threshold of **0.4** is recommended for initial deployment, with monitoring to adjust based on actual intervention outcomes.
