import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
import time
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=20, n_clusters=5):
    np.random.seed(0)
    data = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_clusters, size=n_samples)
    return data, labels

# Define and train Autoencoder
def build_autoencoder(input_dim, encoding_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder

def train_autoencoder(autoencoder, data, epochs=50, batch_size=256):
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)

# Clustering with k-means
def cluster_features(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

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
    encoding_dim = 10

    # Generate synthetic data
    X, true_labels = generate_synthetic_data(n_samples, n_features, n_clusters)
    X = StandardScaler().fit_transform(X)  # Feature scaling

    # Define and train autoencoder
    autoencoder, encoder = build_autoencoder(n_features, encoding_dim)
    start_time = time.time()
    train_autoencoder(autoencoder, X, epochs=50, batch_size=256)
    features = encoder.predict(X)
    predicted_labels = cluster_features(features, n_clusters)
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
