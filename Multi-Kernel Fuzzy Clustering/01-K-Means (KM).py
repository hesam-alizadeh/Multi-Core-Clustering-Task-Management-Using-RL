import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Define the K-Means clustering algorithm
class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.random_state = random_state  # Seed for random number generator
    
    def fit(self, X):
        # Initialize KMeans with the provided parameters
        self.kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
        
        # Start timing
        start_time = time.time()
        
        # Fit the model
        self.kmeans.fit(X)
        
        # Record execution time
        self.execution_time = time.time() - start_time
        
        # Store cluster labels
        self.labels_ = self.kmeans.labels_
        return self

# Load and preprocess the dataset
digits = load_digits()
X = digits.data
y = digits.target
X = StandardScaler().fit_transform(X)
n_clusters = len(np.unique(y))  # Number of clusters based on true labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit K-Means model
kmeans = CustomKMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_test)

# Evaluate clustering results
def evaluate_clustering(y_true, y_pred, X):
    """Evaluate clustering performance."""
    # Compute evaluation metrics
    accuracy = None  # K-Means does not directly provide accuracy
    silhouette = silhouette_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)
    rand_index = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f'Accuracy: {accuracy if accuracy is not None else "N/A"}')
    print(f'Silhouette Index: {silhouette}')
    print(f'Davies-Bouldin Index: {davies_bouldin}')
    print(f'Rand Index: {rand_index}')
    print(f'Adjusted Rand Index: {ari}')
    print(f'Normalized Mutual Information: {nmi}')
    print(f'F1-Score: {f1}')
    print(f'Execution Time: {kmeans.execution_time:.4f} seconds')
    
    return {
        'accuracy': accuracy,
        'silhouette_index': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'rand_index': rand_index,
        'adjusted_rand_index': ari,
        'normalized_mutual_info_score': nmi,
        'f1_score': f1,
        'execution_time': kmeans.execution_time
    }

# Evaluate on the test set
metrics = evaluate_clustering(y_test, kmeans.labels_, X_test)
print("K-Means Clustering complete.")
