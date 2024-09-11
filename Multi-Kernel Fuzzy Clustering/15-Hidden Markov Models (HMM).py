import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt

# Generate synthetic sequential data for HMM
def generate_synthetic_data(n_samples=100, n_features=3):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    return X

# Define and fit HMM
def fit_hmm(X, n_components=2):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=0)
    model.fit(X)
    return model

# Predict labels using HMM
def predict_hmm(model, X):
    return model.predict(X)

# Compute clustering metrics
def compute_metrics(true_labels, predicted_labels):
    # Note: In real scenarios, true_labels are needed to calculate these metrics. Here, using predicted labels for demonstration.
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
    n_samples = 100
    n_features = 3
    n_components = 3
    
    # Generate synthetic data
    X = generate_synthetic_data(n_samples, n_features)
    true_labels = np.random.randint(0, n_components, size=n_samples)  # Simulated true labels
    
    # Fit HMM
    start_time = time.time()
    model = fit_hmm(X, n_components)
    predicted_labels = predict_hmm(model, X)
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
    plt.title("Synthetic Data with Predicted HMM Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

# Run the experiment
run_experiment()
