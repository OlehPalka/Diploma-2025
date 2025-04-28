# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import zscore
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples, 
    davies_bouldin_score, calinski_harabasz_score
)

# ------------------------------
# 1. Load and Explore the Dataset
# ------------------------------
file_path = r"D:\Diploma\data_march\user_table_who_purchased_full.csv"  # Ensure correct path

df = pd.read_csv(file_path, sep=";", decimal=',')

columns_to_include = [
                      'Orders2Visit ratio', 
                      'AVG Purchase value',
                      'AVG Purchase Size', 
                      'User LT Month', 
                      'Purchase Freq.', 
                      'AVG Pages count per visit', 
                      'AVG Time On Page', 
                      'Pet']

df = df[columns_to_include]

dtype_mapping = {
    'Orders2Visit ratio': 'float',
    'AVG Purchase value': 'float',
    'AVG Purchase Size': 'float',
    'User LT Month': 'int',
    'Purchase Freq.': 'float',
    'AVG Pages count per visit': 'float',
    'AVG Time On Page': 'float'
}

df = df.astype(dtype_mapping)


# Display first few rows, info, and summary statistics
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize correlations among numeric features
# Handle missing values (e.g., drop rows with missing values)
df = df.dropna()

numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# 2. Data Preprocessing
# ------------------------------
# In many cases, even non-numeric columns (if they are important) should be encoded.
# For simplicity, if non-numeric columns exist and are relevant, consider encoding them.
# Here, we use numeric columns only; adjust if you need to incorporate categorical data.
numerical_features = [
                      'Orders2Visit ratio', 
                      'AVG Purchase value', 
                      'AVG Purchase Size',
                      'User LT Month', 
                      'Purchase Freq.', 
                      'AVG Pages count per visit', 
                      'AVG Time On Page']
df_numeric = df[numerical_features].copy()

# Scale features so that distance-based algorithms work properly.
scaler = MinMaxScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(df_numeric),
    columns=numerical_features,
    index=df_numeric.index
)

pet_cat = ((df['Pet'] == 1) | (df['Pet'] == 3)).astype(int)
pet_dog = ((df['Pet'] == 2) | (df['Pet'] == 3)).astype(int)

pet_encoded_df = pd.DataFrame({
    'Cat': pet_cat,
    'Dog': pet_dog
}, index=df.index)

# device_phone = ((df['Device'] == 1) | (df['Device'] == 3)).astype(int)
# device_pc = ((df['Device'] == 2) | (df['Device'] == 3)).astype(int)

# device_encoded_df = pd.DataFrame({
#     'Phone': device_phone,
#     'PC': device_pc
# }, index=df.index)

# Append to your scaled DataFrame
X_scaled = pd.concat([X_scaled, pet_encoded_df], axis=1)
print(X_scaled)




# Preview the result

z_scores = np.abs(zscore(X_scaled))
filtered_idx = (z_scores < 2.5).all(axis=1)




# ------------------------------
# 3. Define Evaluation & Visualization Functions
# ------------------------------

# --- add at the top with your other imports ---
from sklearn.manifold import TSNE

# ------------------------------
# 2.5 Compute a shared t-SNE embedding (once)
# ------------------------------
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

def plot_tsne(labels, method_name):
    """
    Scatter-plot of the precomputed t-SNE embedding, colored by cluster labels.
    """
    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        # treat noise points (label == -1) specially
        label_name = 'Noise' if lab == -1 else f'Cluster {lab}'
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=20,
            label=label_name,
            alpha=0.7
        )
    plt.title(f"t-SNE Visualization for {method_name}")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def evaluate_clustering(X, labels, method_name):
    # Filter out noise (-1) for silhouette score if necessary
    valid_idx = labels != -1
    unique_valid_labels = np.unique(labels[valid_idx])
    if len(unique_valid_labels) > 1:
        sil = silhouette_score(X[valid_idx], labels[valid_idx])
    else:
        sil = np.nan
    if len(np.unique(labels)) > 1:
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        db = np.nan
        ch = np.nan
    print(f"{method_name} Evaluation:")
    print(f"  Silhouette Score: {sil:.3f}")
    print(f"  Davies-Bouldin Index: {db:.3f}")
    print(f"  Calinski-Harabasz Index: {ch:.3f}\n")
    return sil, db, ch

def plot_silhouette(X, labels, method_name):
    # Compute the silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, labels)
    unique_labels = np.unique(labels)
    
    y_lower = 10
    plt.figure(figsize=(10, 6))
    for cluster in unique_labels:
        if cluster == -1:
            continue  # Skip noise for silhouette plot
        # Aggregate silhouette scores for samples in the current cluster
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals)
        plt.text(-0.05, y_lower + 0.5 * size_cluster, f"Cluster {cluster}")
        y_lower = y_upper + 10
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster Label")
    plt.title(f"Silhouette Plot for {method_name}")
    # Filter X to match the filtered labels
    valid_idx = labels != -1
    if len(np.unique(labels[valid_idx])) > 1:
        overall_silhouette = silhouette_score(X[valid_idx], labels[valid_idx])
        plt.axvline(x=overall_silhouette, color="red", linestyle="--")
    plt.show()

def plot_clusters_pca(X, labels, method_name):
    # Reduce dimensions to 2D using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Cluster Plot for {method_name}")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def print_cluster_info(labels, method_name):
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        print(f"{method_name}: {len(unique_labels)-1} clusters (excluding noise); Noise points: {np.sum(labels == -1)}")
    else:
        print(f"{method_name}: {len(unique_labels)} clusters found.")

# ------------------------------
# 4. Clustering Approaches
# ------------------------------

# 4.1 K-means Clustering with Elbow Method
inertia = []
K_range = range(2, 11)
for k in K_range:
    km_temp = KMeans(n_clusters=k, random_state=42)
    km_temp.fit(X_scaled)
    inertia.append(km_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-means")
plt.show()

# Select optimal k based on the elbow plot (example: k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
evaluate_clustering(X_scaled, labels_kmeans, "K-means")
plot_silhouette(X_scaled, labels_kmeans, "K-means")

print_cluster_info(labels_kmeans, "K-means")
plot_tsne(labels_kmeans, "K-means")


joblib.dump(kmeans, r"D:\Diploma\data_march\kmeans_model.pkl")
# Save the scaler (this is necessary if new data needs to be scaled in the same way)
joblib.dump(scaler, r"D:\Diploma\data_march\scaler.pkl")


# --- 2. Extract the scaled centroids ---
scaled_centroids = kmeans.cluster_centers_   # shape (n_clusters, n_features_total)

# --- 3. Define your feature ordering ---
#    These must match exactly how you built X_scaled:
numeric_features = [
    'Orders2Visit ratio',
    'AVG Purchase value',
    'AVG Purchase Size',
    'User LT Month',
    'Purchase Freq.',
    'AVG Pages count per visit',
    'AVG Time On Page'
]
dummy_features = ['Cat', 'Dog']  # the two pet dummies appended after scaling

# --- 4. Split numeric vs. dummy parts ---
n_num = len(numeric_features)
scaled_num_part   = scaled_centroids[:, :n_num]   # (n_clusters x 7)
scaled_dummy_part = scaled_centroids[:, n_num:]   # (n_clusters x 2)

# --- 5. Inverse-transform only the numeric block ---
original_num_part = scaler.inverse_transform(scaled_num_part)

# --- 6. Re-assemble full-centroid array ---
original_centroids = np.hstack([
    original_num_part,
    scaled_dummy_part
])  # shape (n_clusters x 9)

# --- 7. Build DataFrame and save to CSV ---
columns = ['cluster'] + numeric_features + dummy_features
df_centroids = pd.DataFrame(
    np.column_stack([np.arange(original_centroids.shape[0]), original_centroids]),
    columns=columns
)
df_centroids.to_csv(r"D:\Diploma\data_march\projected_centroids.csv", sep=";", index=False, decimal=',', encoding="utf-8-sig")

print(f"✅ Saved {len(df_centroids)} centroids to:\n   {r"D:\Diploma\data_march\projected_centroids.csv"}")

# 4.2 Gaussian Mixture Models (GMM)
optimal_k = 4
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm.fit(X_scaled)
labels_gmm = gmm.predict(X_scaled)
evaluate_clustering(X_scaled, labels_gmm, "GMM")
plot_silhouette(X_scaled, labels_gmm, "GMM")
print_cluster_info(labels_gmm, "GMM")
plot_tsne(labels_gmm, "GMM")

# 4.3 OPTICS Clustering
# Parameters may require tuning; here we use a basic configuration.
min_samples = 10
xi = 0.03
min_cluster_size = 0.10
print(F"min_samples {min_samples}")
print(F"xi {xi}")
print(F"min_cluster_size {min_cluster_size}")
optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
labels_optics = optics.fit_predict(X_scaled)
evaluate_clustering(X_scaled, labels_optics, "OPTICS")
# Plot silhouette only if at least 2 clusters (excluding noise) exist.
if len(np.unique(labels_optics[labels_optics != -1])) > 1:
    plot_silhouette(X_scaled, labels_optics, "OPTICS")
print_cluster_info(labels_optics, "OPTICS")
plot_tsne(labels_optics, "OPTICS")


# 4.4 DBSCAN Clustering
# Parameters for DBSCAN may also require tuning based on data density.
eps = 0.6
min_samples = 15
print(F"Eps val {eps}")
print(F"Min sample {min_samples}")
dbscan = DBSCAN(eps, min_samples=min_samples)
labels_dbscan = dbscan.fit_predict(X_scaled)
evaluate_clustering(X_scaled, labels_dbscan, "DBSCAN")
if len(np.unique(labels_dbscan[labels_dbscan != -1])) > 1:
    plot_silhouette(X_scaled, labels_dbscan, "DBSCAN")
print_cluster_info(labels_dbscan, "DBSCAN")
plot_tsne(labels_dbscan, "DBSCAN")
