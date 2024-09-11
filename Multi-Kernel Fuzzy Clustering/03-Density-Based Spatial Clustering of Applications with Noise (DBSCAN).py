import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import time

# Load and preprocess the dataset
digits = load_digits()
X = digits.data
y = digits.target
X = StandardScaler().fit_transform(X)
n_clusters = len(np.unique(y))  # Number of clusters based on true labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit DBSCAN model
start_time = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5)  # You may need to adjust `eps` and `min_samples`
y_pred = dbscan.fit_predict(X_test)
execution_time = time.time() - start_time

# Evaluate clustering results
def evaluate_clustering(y_true, y_pred, X):
    """Evaluate clustering performance."""
    # DBSCAN may produce -1 for outliers which makes some metrics not applicable
    mask = (y_pred != -1)
    if np.sum(mask) > 1:  # Ensure there are enough points for evaluation
        accuracy = None  # DBSCAN does not directly provide accuracy
        silhouette = silhouette_score(X[mask], y_pred[mask])
        davies_bouldin = davies_bouldin_score(X[mask], y_pred[mask])
        rand_index = rand_score(y_true[mask], y_pred[mask])
        ari = adjusted_rand_score(y_true[mask], y_pred[mask])
        nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average='weighted')
    else:
        accuracy = None
        silhouette = davies_bouldin = rand_index = ari = nmi = f1 = None
    
    print(f'Accuracy: {accuracy if accuracy is not None else "N/A"}')
    print(f'Silhouette Index: {silhouette if silhouette is not None else "N/A"}')
    print(f'Davies-Bouldin Index: {davies_bouldin if davies_bouldin is not None else "N/A"}')
    print(f'Rand Index: {rand_index if rand_index is not None else "N/A"}')
    print(f'Adjusted Rand Index: {ari if ari is not None else "N/A"}')
    print(f'Normalized Mutual Information: {nmi if nmi is not None else "N/A"}')
    print(f'F1-Score: {f1 if f1 is not None else "N/A"}')
    
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

print("DBSCAN Clustering complete.")
