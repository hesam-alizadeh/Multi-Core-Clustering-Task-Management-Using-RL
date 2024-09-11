import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
import time

class SOM:
    def __init__(self, m, n, dim, n_iterations=1000, learning_rate=0.5, sigma=None):
        """Initialize the Self-Organizing Map (SOM) model.
        
        Parameters:
        - m, n: Dimensions of the SOM grid.
        - dim: Dimensionality of the input data.
        - n_iterations: Number of training iterations.
        - learning_rate: Initial learning rate.
        - sigma: Initial sigma for the neighborhood function.
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma else max(m, n) / 2
        self.weights = np.random.random((m, n, dim))
        self.time_constant = n_iterations / np.log(self.sigma)

    def _neighborhood_function(self, distance, iteration):
        """Compute the neighborhood function based on the distance."""
        return np.exp(-distance ** 2 / (2 * (self.sigma * np.exp(-iteration / self.time_constant)) ** 2))

    def _update_weights(self, input_vec, bmu_idx, iteration):
        """Update the weights of the SOM grid."""
        bmu_distance_sqr = (np.sum((np.array(np.unravel_index(np.arange(self.m * self.n), (self.m, self.n))) - np.array(bmu_idx).reshape(2, 1)) ** 2, axis=0))
        neighborhood_factor = self._neighborhood_function(bmu_distance_sqr, iteration).reshape(self.m, self.n)
        learning_rate = self.learning_rate * np.exp(-iteration / self.n_iterations)
        self.weights += learning_rate * neighborhood_factor[:, :, np.newaxis] * (input_vec - self.weights)

    def _find_bmu(self, input_vec):
        """Find the Best Matching Unit (BMU) for a given input vector."""
        distances = np.linalg.norm(np.subtract(self.weights, input_vec), axis=2)
        return np.unravel_index(np.argmin(distances), (self.m, self.n))

    def train(self, x):
        """Train the SOM on the dataset."""
        for iteration in range(self.n_iterations):
            for input_vec in x:
                bmu_idx = self._find_bmu(input_vec)
                self._update_weights(input_vec, bmu_idx, iteration)

    def predict(self, x):
        """Predict cluster labels for the dataset."""
        labels = np.zeros((x.shape[0],), dtype=int)
        for i, input_vec in enumerate(x):
            bmu_idx = self._find_bmu(input_vec)
            labels[i] = np.ravel_multi_index(bmu_idx, (self.m, self.n))
        return labels

    def evaluate(self, x, y_true):
        """Evaluate the SOM model using various metrics."""
        start_time = time.time()
        y_pred = self.predict(x)
        execution_time = time.time() - start_time

        acc = accuracy_score(y_true, y_pred)
        sil_score = silhouette_score(x, y_pred)
        db_score = davies_bouldin_score(x, y_pred)
        ri_score = rand_score(y_true, y_pred)
        ari_score = adjusted_rand_score(y_true, y_pred)
        nmi_score = normalized_mutual_info_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f'Accuracy: {acc:.4f}')
        print(f'Silhouette Score: {sil_score:.4f}')
        print(f'Davies-Bouldin Index: {db_score:.4f}')
        print(f'Rand Index: {ri_score:.4f}')
        print(f'Adjusted Rand Index: {ari_score:.4f}')
        print(f'Normalized Mutual Information: {nmi_score:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Execution Time: {execution_time:.4f} seconds')

        return {
            'accuracy': acc,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score,
            'rand_index': ri_score,
            'adjusted_rand_index': ari_score,
            'normalized_mutual_info_score': nmi_score,
            'f1_score': f1,
            'execution_time': execution_time
        }

# Example usage:
if __name__ == "__main__":
    # Load the dataset
    digits = load_digits()
    x = digits.data
    x = StandardScaler().fit_transform(x)
    y = digits.target

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the SOM model
    som = SOM(m=10, n=10, dim=x_train.shape[1], n_iterations=1000, learning_rate=0.5)
    som.train(x_train)

    # Evaluate the model on the test set
    metrics = som.evaluate(x_test, y_test)

    print("Clustering complete.")
