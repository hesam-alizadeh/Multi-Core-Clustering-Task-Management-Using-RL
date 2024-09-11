import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from deepbelief import DBN

# Load and preprocess the dataset
digits = load_digits()
X = digits.data
y = digits.target
X = StandardScaler().fit_transform(X)
n_classes = len(np.unique(y))  # Number of classes based on true labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the DBN model
class CustomDBN:
    def __init__(self, n_hidden_layers=[64, 32], n_epochs=10, batch_size=10, learning_rate=0.1, random_state=None):
        self.n_hidden_layers = n_hidden_layers  # List specifying number of neurons in each hidden layer
        self.n_epochs = n_epochs  # Number of training epochs
        self.batch_size = batch_size  # Size of each training batch
        self.learning_rate = learning_rate  # Learning rate for training
        self.random_state = random_state  # Seed for random number generator

    def fit(self, X_train, y_train):
        # Initialize DBN with specified parameters
        self.dbn = DBN(
            hidden_layers_sizes=self.n_hidden_layers,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        
        # Start timing
        start_time = time.time()
        
        # Fit the model
        self.dbn.fit(X_train, y_train)
        
        # Record execution time
        self.execution_time = time.time() - start_time

    def predict(self, X):
        return self.dbn.predict(X)

# Initialize and fit the DBN model
dbn = CustomDBN(n_hidden_layers=[64, 32], n_epochs=10, batch_size=10, learning_rate=0.1, random_state=42)
dbn.fit(X_train, y_train)

# Predict and evaluate
y_pred = dbn.predict(X_test)

# Evaluate the performance
def evaluate_classification(y_true, y_pred, X):
    """Evaluate classification performance."""
    accuracy = accuracy_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred) if len(set(y_pred)) > 1 else 'N/A'
    davies_bouldin = davies_bouldin_score(X, y_pred) if len(set(y_pred)) > 1 else 'N/A'
    rand_index = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Silhouette Index: {silhouette}')
    print(f'Davies-Bouldin Index: {davies_bouldin}')
    print(f'Rand Index: {rand_index}')
    print(f'Adjusted Rand Index: {ari}')
    print(f'Normalized Mutual Information: {nmi}')
    print(f'F1-Score: {f1}')
    print(f'Execution Time: {dbn.execution_time:.4f} seconds')
    
    return {
        'accuracy': accuracy,
        'silhouette_index': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'rand_index': rand_index,
        'adjusted_rand_index': ari,
        'normalized_mutual_info_score': nmi,
        'f1_score': f1,
        'execution_time': dbn.execution_time
    }

# Evaluate on the test set
metrics = evaluate_classification(y_test, y_pred, X_test)
print("Deep Belief Network (DBN) training complete.")
