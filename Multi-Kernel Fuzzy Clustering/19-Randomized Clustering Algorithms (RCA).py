import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=20, n_clusters=5):
    X, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=0)
    return X, labels

# Randomized K-Means Initialization
def randomized_kmeans(X, n_clusters, n_init=10):
    best_labels = None
    best_score = -np.inf
    best_kmeans = None
    for i in range(n_init):
        # Randomly shuffle the data to get different initializations
        X_shuffled, _ = shuffle(X, random_state=i)
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=X_shuffled[:n_clusters], random_state=i)
        labels = kmeans.fit_predict(X)
        # Evaluate the clustering result
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -np.inf
        if score > best_score:
            best_score = score
            best_labels = labels
            best_kmeans = kmeans
    return best_labels, best_kmeans

# Compute metrics
def compute_metrics(true_labels, predicted_labels):
    silhouette = silhouette_score(X, predicted_labels) if len(set(predicted_labels)) > 1 else 'N/A'
    db_index = davies_bouldin_score(X, predicted_labels) if len(set(predicted_labels)) > 1 else 'N/A'
    rand_idx = rand_score(true_labels, predicted_labels) if true_labels is not None else 'N/A'
    ari = adjusted_rand_score(true_labels, predicted_labels) if true_labels is not None else 'N/A'
    nmi = normalized_mutual_info_score(true_labels, predicted_labels) if true_labels is not None else 'N/A'
    f1 = f1_score(true_labels, predicted_labels, average='weighted') if true_labels is not None else 'N/A'
    return silhouette, db_index, rand_idx, ari, nmi, f1

# Main function to run the experiment
def run_experiment():
    # Parameters
    n_samples = 1000
    n_features = 20
    n_clusters = 5
    n_init = 10  # Number of random initializations

    # Generate synthetic data
    X, true_labels = generate_synthetic_data(n_samples, n_features, n_clusters)
    X = StandardScaler().fit_transform(X)  # Feature scaling

    # Run clustering
    start_time = time.time()
    predicted_labels, kmeans = randomized_kmeans(X, n_clusters, n_init)
    execution_time = time.time() - start_time

    # Compute metrics
    silhouette, db_index, rand_idx, ari, nmi, f1 = compute_metrics(true_labels, predicted_labels)

    # Print metrics
    print(f"Silhouette Index: {silhouette}")
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Rand Index: {rand_idx}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"F1 Score: {f1}")
    print(f"Execution Time: {execution_time:.2f} seconds")

    # Plot synthetic data
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k', s=50)
    plt.title("Synthetic Data with Predicted Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

# Run the experiment
run_experiment()