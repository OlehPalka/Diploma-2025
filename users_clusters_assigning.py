import numpy as np
import pandas as pd
import joblib

# ================================
# Section 1: USERS WITHOUT PURCHASES
# ================================

CENTROIDS_CSV = r"D:\Diploma\data_march\projected_centroids.csv"
df_centroids = pd.read_csv(CENTROIDS_CSV, sep=";", decimal=",", index_col="cluster")
# Four common features (by their 0-based indices in the original 9-feature order):
# Index 5: 'AVG Pages count per visit'
# Index 6: 'AVG Time On Page'
# Index 7: 'Pet'
# Index 8: 'Is interested in actions'
common_feature_indices = [5, 6, 7, 8]

# Load the trained scaler and 9-feature K-means model

# The full list of features used in training (order matters)
all_features = [
    'Orders2Visit ratio', 
    'AVG Purchase value', 
    'AVG Purchase Size', 
    'User LT Month', 
    'Purchase Freq.', 
    'AVG Pages count per visit', 
    'AVG Time On Page', 
    'Cat',
    'Dog'
]

# Project the 9-dimensional centroids to the common 4D subspace.
print(df_centroids)
centroids_common = df_centroids[[
    'AVG Pages count per visit',
    'AVG Time On Page',
    'Cat',
    'Dog'
]].values   # shape: (n_clusters, 4)

# Load the full dataset for users without purchases (all columns preserved).
file_no_purchase = r"D:\Diploma\data_march\user_table_no_purchase.csv"  # Update path accordingly
df_no_purchase_full = pd.read_csv(file_no_purchase, sep=";", decimal=',', encoding="utf-8-sig")

# For cluster assignment, extract only the common features.
common_features = ['AVG Pages count per visit', 'AVG Time On Page', 'Pet']
df_no_purchase_common = df_no_purchase_full[common_features].copy()
df_no_purchase_common['Cat'] = ((df_no_purchase_common['Pet'] == 1) | (df_no_purchase_common['Pet'] == 3)).astype(int)
df_no_purchase_common['Dog'] = ((df_no_purchase_common['Pet'] == 2) | (df_no_purchase_common['Pet'] == 3)).astype(int)
df_no_purchase_common = df_no_purchase_common[['AVG Pages count per visit', 'AVG Time On Page', 'Cat', 'Dog']]
X_new_scaled = df_no_purchase_common.values

# Compute Euclidean distances (in 4D common space) from each sample to each projected centroid.
distances = np.linalg.norm(X_new_scaled[:, np.newaxis, :] - centroids_common[np.newaxis, :, :], axis=2)

# Assign each sample to the cluster with the minimum distance.
predicted_clusters = np.argmin(distances, axis=1)

# Add the predicted clusters back into the full dataset.
df_no_purchase_full['cluster'] = predicted_clusters

# Save the full dataset (with all original columns plus 'cluster') to CSV.
df_no_purchase_full.to_csv(r"D:\Diploma\data_march\user_table_no_purchase_with_clusters.csv", sep=";", index=False, decimal=',', encoding="utf-8-sig")
print("User assignment complete for no-purchase users. File saved as 'user_table_no_purchase_with_clusters.csv'.")


# ================================
# Section 2: USERS WITH PURCHASE (Purchased/Churned)
# ================================

# Define the full list of features used for clustering (9 features, order matters).
columns_to_include = [
    'Orders2Visit ratio', 
    'AVG Purchase value', 
    'AVG Purchase Size', 
    'User LT Month', 
    'Purchase Freq.', 
    'AVG Pages count per visit', 
    'AVG Time On Page'
]

# Load the full dataset for purchased/churned users (all columns preserved).
file_purchased_churned = r"D:\Diploma\data_march\user_table_purchased_churned.csv"  # Update with the actual path
df_purchased_full = pd.read_csv(file_purchased_churned, sep=";", decimal=',', encoding="utf-8-sig")

scaler = joblib.load(r"D:\Diploma\data_march\scaler.pkl")
kmeans_model = joblib.load(r"D:\Diploma\data_march\kmeans_model.pkl")

# For cluster assignment, extract only the clustering features.
df_purchased_cluster = df_purchased_full[columns_to_include].copy()

# Handle missing values by dropping rows with any missing clustering features.
df_purchased_cluster = df_purchased_cluster.dropna()

# Scale the clustering features using the saved scaler.
df_scaled = pd.DataFrame(
    scaler.transform(df_purchased_cluster),
    columns=columns_to_include,
    index=df_purchased_cluster.index
)
df_scaled['Cat'] = ((df_purchased_full.loc[df_scaled.index,'Pet'] == 1) |
                    (df_purchased_full.loc[df_scaled.index,'Pet'] == 3)).astype(int)
df_scaled['Dog'] = ((df_purchased_full.loc[df_scaled.index,'Pet'] == 2) |
                    (df_purchased_full.loc[df_scaled.index,'Pet'] == 3)).astype(int)

# now predict:
predicted_clusters_purchased = kmeans_model.predict(df_scaled.values)

# Add the predicted cluster to the clustering subset.
df_purchased_cluster = df_purchased_cluster.assign(cluster=predicted_clusters_purchased)

# Merge the cluster assignments back into the full purchased dataset.
# (Rows for which clustering features were incomplete are dropped from the merging.)
df_purchased_full = df_purchased_full.loc[df_purchased_cluster.index].copy()
df_purchased_full['cluster'] = predicted_clusters_purchased

# Save the full dataset (all original columns plus cluster) to CSV.
df_purchased_full.to_csv(r"D:\Diploma\data_march\user_table_purchased_churned_with_clusters.csv", sep=";", index=False, decimal=',', encoding="utf-8-sig")
print("User assignment complete for purchased/churned users. File saved as 'user_table_purchased_churned_with_clusters.csv'.")
