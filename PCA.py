import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------
# 1. Load the Dataset
# ------------------------------
file_path = r"D:\Diploma\data_march\user_table_who_purchased.csv"  # Update the path if needed

dtype_mapping = {
    'Days since last purchase': 'int',
    'Orders2Visit ratio': 'float',
    'AVG Purchase value': 'float',
    'AVG Purchase Size': 'float',
    'User LT Month': 'int',
    'Purchase Freq.': 'float',
    'AVG Pages count per visit': 'float',
    'Total Time On site': 'float',
    'AVG Time On Page': 'float',
    'Pet': 'int',
    'Is interested in actions': 'int'
}

# Read and convert data types
df = pd.read_csv(file_path, sep=";", decimal=',')
df = df.astype(dtype_mapping)
df = df.dropna()  # Remove missing values

# ------------------------------
# 2. Data Preprocessing
# ------------------------------
# Since all columns are numeric, we'll use all variables
X = df.copy()

# Scale the features so that they are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 3. Run PCA
# ------------------------------
# Run PCA considering all features
n_features = X.shape[1]
pca = PCA(n_components=n_features)
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot explained variance and cumulative variance
plt.figure(figsize=(10, 6))
components = np.arange(1, n_features + 1)
plt.bar(components, explained_variance, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(components, cumulative_variance, where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by PCA Components')
plt.legend(loc='best')
plt.show()

# Print explained variance for each component
print("Explained Variance Ratio for each component:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance), start=1):
    print(f"PC{i}: {var:.3f} (Cumulative: {cum_var:.3f})")

# ------------------------------
# 4. Analyze PCA Loadings
# ------------------------------
# PCA loadings (components) tell us how much each original feature contributes to each principal component.
loadings = pca.components_.T  # shape: (n_features, n_components)
feature_names = X.columns

# Create a DataFrame for easier interpretation
loading_df = pd.DataFrame(loadings, index=feature_names, columns=[f'PC{i}' for i in range(1, n_features + 1)])
print("\nPCA Loadings (Feature Contributions):")
print(loading_df)

# Optionally, inspect the absolute loadings for the first two principal components,
# which usually capture most of the variance.
abs_loadings = loading_df[['PC1', 'PC2']].abs().sort_values(by='PC1', ascending=False)
print("\nFeatures sorted by absolute loading in PC1:")
print(abs_loadings)

# ------------------------------
# 5. Calculate PCA-Based Importance Scores
# ------------------------------
# Multiply absolute loadings by explained variance and sum across all components
abs_loadings_full = np.abs(loadings)
importance_scores = abs_loadings_full @ explained_variance  # Matrix multiplication

# Create a DataFrame for importance scores
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'PCA Importance Score': importance_scores
}).sort_values(by='PCA Importance Score', ascending=False)

# Display the results
print("\nPCA-Based Feature Importance Ranking:")
print(importance_df)

# Optional: Plot the importance scores
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='PCA Importance Score', y='Feature', palette='viridis')
plt.title("PCA-Based Feature Importance Scores")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
