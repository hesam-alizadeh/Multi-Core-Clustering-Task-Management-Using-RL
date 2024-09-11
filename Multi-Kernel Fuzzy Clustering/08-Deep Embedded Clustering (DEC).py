import numpy as np
import scipy.spatial
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
import time

class KernelKMeans:
    def __init__(self, n_clusters, gamma=1.0, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples = X.shape[0]
        # Compute the kernel matrix
        K = rbf_kernel(X, X, gamma=self.gamma)
        
        # Initialize cluster centers randomly
        self.labels_ = np.random.randint(0, self.n_clusters, size=n_samples)
        self.cluster_centers_ = np.array([K[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        
        for i in range(self.max_iter):
            # Compute distances to cluster centers
            distances = np.array([np.linalg.norm(K - center, axis=1) for center in self.cluster_centers_]).T
            new_labels = np.argmin(distances, axis=1)
            
            # Check for convergence
            if np.all(new_labels == self.labels_):
                break
            
            self.labels_ = new_labels
            
            # Update cluster centers
            self.cluster_centers_ = np.array([K[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            
        self.n_iter_ = i + 1

    def predict(self, X):
        K = rbf_kernel(X, X, gamma=self.gamma)
        distances = np.array([np.linalg.norm(K - center, axis=1) for center in self.cluster_centers_]).T
        return np.argmin(distances, axis=1)

# Load and preprocess the dataset
digits = load_digits()
X = digits.data
y = digits.target
X = StandardScaler().fit_transform(X)
input_dim = X.shape[1]
n_clusters = 10

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit Kernel K-Means
start_time = time.time()
kernel_kmeans = KernelKMeans(n_clusters=n_clusters, gamma=1.0)
kernel_kmeans.fit(X_train)
y_pred = kernel_kmeans.predict(X_test)
execution_time = time.time() - start_time

# Evaluation metrics
def evaluate_clustering(y_true, y_pred, X):
    """Evaluate clustering performance."""
    accuracy = accuracy_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)
    rand_index = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Silhouette Index: {silhouette:.4f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')
    print(f'Rand Index: {rand_index:.4f}')
    print(f'Adjusted Rand Index: {ari:.4f}')
    print(f'Normalized Mutual Information: {nmi:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    return {
        'accuracy': accuracy,
        'silhouette_index': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'rand_index': rand_index,
        'adjusted_rand_index': ari,
        'normalized_mutual_info_score': nmi,
        'f1_score': f1
    }

# Evaluate on the test set
metrics = evaluate_clustering(y_test, y_pred, X_test)
metrics['execution_time'] = execution_time

print("Kernel K-Means complete.")
