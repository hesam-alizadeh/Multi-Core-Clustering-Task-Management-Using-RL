import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
import pymc3 as pm
import arviz as az
import time

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=2, n_clusters=3):
    X, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    return X, labels

# Bayesian Mixture Models using PyMC3
def bayesian_mixture_models(X, n_clusters, n_iter=1000):
    with pm.Model() as model:
        # Priors for cluster means
        mu = pm.Normal('mu', mu=0, sigma=10, shape=(n_clusters, X.shape[1]))
        
        # Priors for cluster variances
        sigma = pm.InverseGamma('sigma', alpha=1, beta=1, shape=(n_clusters, X.shape[1]))
        
        # Priors for cluster weights
        w = pm.Dirichlet('w', a=np.ones(n_clusters))
        
        # Multinomial likelihood
        obs = pm.Mixture('obs', w=w, comp_dists=pm.MultivariateNormal.dist(mu=mu, cov=sigma), observed=X)
        
        # Sampling
        trace = pm.sample(n_iter, return_inferencedata=True, cores=1)
    
    return trace

# Evaluate clustering results
def evaluate_clustering(X, true_labels, trace):
    # Extracting predicted cluster assignments from the trace
    cluster_assignments = np.argmax(trace.posterior['obs'].mean(axis=0), axis=1)
    
    # Compute metrics
    silhouette = silhouette_score(X, cluster_assignments) if len(set(cluster_assignments)) > 1 else 'N/A'
    db_index = davies_bouldin_score(X, cluster_assignments) if len(set(cluster_assignments)) > 1 else 'N/A'
    rand_idx = rand_score(true_labels, cluster_assignments) if true_labels is not None else 'N/A'
    ari = adjusted_rand_score(true_labels, cluster_assignments) if true_labels is not None else 'N/A'
    nmi = normalized_mutual_info_score(true_labels, cluster_assignments) if true_labels is not None else 'N/A'
    f1 = f1_score(true_labels, cluster_assignments, average='weighted') if true_labels is not None else 'N/A'
    
    return silhouette, db_index, rand_idx, ari, nmi, f1

# Main function to run the experiment
def run_experiment():
    # Parameters
    n_samples = 1000
    n_features = 2
    n_clusters = 3
    n_iter = 1000  # Number of MCMC samples

    # Generate synthetic data
    X, true_labels = generate_synthetic_data(n_samples, n_features, n_clusters)
    X = StandardScaler().fit_transform(X)  # Feature scaling

    # Run Bayesian Mixture Models
    start_time = time.time()
    trace = bayesian_mixture_models(X, n_clusters, n_iter)
    execution_time = time.time() - start_time

    # Evaluate clustering results
    silhouette, db_index, rand_idx, ari, nmi, f1 = evaluate_clustering(X, true_labels, trace)

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
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(trace.posterior['obs'].mean(axis=0), axis=1), cmap='viridis', edgecolor='k', s=50)
    plt.title("Synthetic Data with Predicted Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

# Run the experiment
run_experiment()
