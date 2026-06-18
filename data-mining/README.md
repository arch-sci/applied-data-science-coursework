# Data Mining — Homework 1-3

Three assignments from the Data Mining course, each on a synthetic dataset graded against a hidden ground truth.

## HW1 — Fraudulent Transaction Detection

**Brief:** Detect fraudulent transactions in 10,000 records described by 4 features. Submit a `predictions.csv` of 0/1 labels, graded on Matthews Correlation Coefficient (MCC) against the hidden labels.

**Approach:** Log-transformed the two skewed features (transaction amount, items in cart), then flagged outliers using Mahalanobis distance with a chi-square threshold at the 95th percentile (4 degrees of freedom). Verified the manual distance calculation against a reference implementation before flagging.

## HW2 — Patient Clustering (Cancer Subtypes)

**Brief:** Identify cancer subtypes from 451 PCA-reduced gene expression features (no labels given), graded on Adjusted Rand Index (ARI) against hidden true labels.

**Approach:** No re-standardization (the PCs already encode variance ranking). Used K-means, selected K=4 via the elbow method on inertia, validated with silhouette score (outperforming a hierarchical clustering benchmark), and visualized clusters with t-SNE.

## HW3 — Insurance Premium Prediction

**Brief:** Predict `Annual_Premium` for health insurance customers from 6 demographic/health features, graded on RMSE.

**Approach:** One-hot encoded categoricals, then compared OLS, Ridge, and Lasso (with log-transformed target and interaction terms) via 10-fold cross-validation — Ridge outperformed, suggesting OLS overfit. Switched to tree-based models (Random Forest, Gradient Boosting); Gradient Boosting gave the lowest RMSE and was used for the final test predictions.

## Stack

Python · pandas · numpy · scikit-learn · matplotlib · seaborn · scipy
