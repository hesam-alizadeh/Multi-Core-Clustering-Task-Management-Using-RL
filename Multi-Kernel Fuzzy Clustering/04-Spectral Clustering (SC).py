import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
import time

# Load and preprocess the dataset
digits = load_digits()
X = digits.data
y = digits.target
X = StandardScaler().fit_transform(X)
n_clusters = 10  # Number of clusters for Spectral Clustering

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit Spectral Clustering model
start_time = time.time()
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
y_pred = spectral_clustering.fit_predict(X_test)
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

print("Spectral Clustering complete.")
