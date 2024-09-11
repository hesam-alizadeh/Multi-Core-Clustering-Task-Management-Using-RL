import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import time

# Import necessary kernel libraries
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, cosine_similarity

class FuzzyMultiKernelClustering:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-4, gamma=0.1, beta=0.1):
        self.n_clusters = n_clusters
        self.m = m  # Fuzzification parameter
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma  # Tuning parameter for different cluster penalties
        self.beta = beta  # Tuning parameter for similar cluster penalties

    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids = np.random.rand(self.n_clusters, n_features)
        self.membership = np.random.rand(n_samples, self.n_clusters)
        self.membership = self.membership / self.membership.sum(axis=1, keepdims=True)
        self.kernel_weights = np.random.rand(n_samples)

        for iteration in range(self.max_iter):
            prev_membership = np.copy(self.membership)
            
            # Update centroids
            for c in range(self.n_clusters):
                numerator = np.sum((self.membership[:, c] ** self.m)[:, np.newaxis] * X, axis=0)
                denominator = np.sum(self.membership[:, c] ** self.m)
                self.centroids[c, :] = numerator / denominator
            
            # Compute distance in the feature space using multi-kernel functions
            distances = np.zeros((n_samples, self.n_clusters))
            for i in range(n_samples):
                for c in range(self.n_clusters):
                    # Using multiple kernel functions: Gaussian, Polynomial, and Cosine Similarity
                    gaussian_dist = rbf_kernel(X[i].reshape(1, -1), self.centroids[c].reshape(1, -1))
                    polynomial_dist = polynomial_kernel(X[i].reshape(1, -1), self.centroids[c].reshape(1, -1))
                    cosine_dist = cosine_similarity(X[i].reshape(1, -1), self.centroids[c].reshape(1, -1))
                    
                    distances[i, c] = (self.kernel_weights[i] * (gaussian_dist + polynomial_dist + cosine_dist)).sum()

            # Update membership degrees
            for i in range(n_samples):
                for c in range(self.n_clusters):
                    self.membership[i, c] = 1.0 / np.sum([(distances[i, c] / distances[i, k]) ** (2 / (self.m - 1)) 
                                                          for k in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(self.membership - prev_membership) < self.tol:
                break

        self.final_membership = self.membership.argmax(axis=1)

    def predict(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for c in range(self.n_clusters):
                gaussian_dist = rbf_kernel(X[i].reshape(1, -1), self.centroids[c].reshape(1, -1))
                polynomial_dist = polynomial_kernel(X[i].reshape(1, -1), self.centroids[c].reshape(1, -1))
                cosine_dist = cosine_similarity(X[i].reshape(1, -1), self.centroids[c].reshape(1, -1))
                
                distances[i, c] = (self.kernel_weights[i] * (gaussian_dist + polynomial_dist + cosine_dist)).sum()

        return distances.argmin(axis=1)

# Dataset generation for demonstration purposes
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
X = StandardScaler().fit_transform(X)

# Initialize and fit the Fuzzy Multi-Kernel Clustering model
fmkc = FuzzyMultiKernelClustering(n_clusters=3, max_iter=100, tol=1e-4, gamma=0.1, beta=0.1)

start_time = time.time()
fmkc.fit(X)
end_time = time.time()

# Calculate performance metrics
y_pred = fmkc.final_membership
accuracy = accuracy_score(y_true, y_pred)
silhouette = silhouette_score(X, y_pred)
davies_bouldin = davies_bouldin_score(X, y_pred)
rand_index = rand_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
execution_time = end_time - start_time
iterations = fmkc.max_iter

# Print the evaluation results
print(f'Accuracy: {accuracy:.4f}')
print(f'Silhouette Index: {silhouette:.4f}')
print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')
print(f'Rand Index: {rand_index:.4f}')
print(f'Adjusted Rand Index (ARI): {ari:.4f}')
print(f'Normalized Mutual Information (NMI): {nmi:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Execution Time: {execution_time:.4f} seconds')
print(f'Iterations to Convergence: {iterations}')
