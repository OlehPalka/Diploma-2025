from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
file_path = r"D:\Diploma\data\user_table_who_purchased.xlsx"
data = pd.read_excel(file_path)

# Create IsChurned column: True if 'DaysSinceLastPurchase' is greater than 90
data["IsChurned"] = data["Days since last purchase"] > 90

# Selecting features for clustering
features = data[[
                      'Device', 
                      'Days since last purchase', 
                      'Orders2Visit ratio', 
                      'Total Spent', 
                      'AVG Purchase value', 
                      'AVG Purchase Size', 
                      'User LT Month', 
                      'Purchase Freq.', 
                      'Total Items Purchased', 
                      'AVG Pages count per visit', 
                      'Total Time On site', 
                      'AVG Time per visit', 
                      'AVG Time On Page', 
                      'Pet', 
                      'Is interested in actions']]

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Remove churned users only (No outlier filtering)
active_users = ~data["IsChurned"]

# Create a mask for non-NaN values in features_scaled
non_nan_mask = ~np.isnan(features_scaled).any(axis=1)

# Apply the mask to filter both active_users and features_scaled
features_scaled_cleaned = features_scaled[non_nan_mask]
active_users_cleaned = active_users[non_nan_mask]

# Check if there are any active users left after cleaning
if active_users_cleaned.sum() == 0:
    raise ValueError("All users are churned! Adjust threshold.")

# Fit the KMeans model on active users (after removing NaNs)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(features_scaled_cleaned)

# Assign cluster labels to active users
data.loc[non_nan_mask, "Cluster"] = kmeans.labels_.astype(int)

# Assign -1 to churned users
data.loc[~non_nan_mask, "Cluster"] = -1

# Calculate the centroids of the clusters
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

# Create a DataFrame for centroids
centroids_df = pd.DataFrame(centroids, columns=features.columns)

# Print centroids
print(centroids_df)

# Save the centroids to a CSV file
centroids_file_path = "data/cluster_centroids.csv"
centroids_df.to_csv(centroids_file_path, index=False)
print(f"Saved centroids to {centroids_file_path}")

# Save the results with cluster labels
new_file_path = "data/UserTable_club4paws_clustered_kmeans.csv"
data.to_csv(new_file_path, index=False)
print(f"Saved clustered data to {new_file_path}")

# Plot clusters (optional)
# plt.figure(figsize=(12, 6))
# for i in range(centroids.shape[1]):
#     plt.subplot(3, 3, i + 1)
#     plt.hist(features_scaled[:, i], bins=20, alpha=0.5, label="Data")
#     for j in range(centroids.shape[0]):
#         plt.axvline(x=centroids[j, i], color="r", linestyle="dashed", linewidth=2, label=f"Centroid {j}")
#     plt.title(features.columns[i])
#     plt.legend()

# plt.tight_layout()
# plt.savefig("segmentation/clusters_visualised_club4paws.png")
# plt.show()
