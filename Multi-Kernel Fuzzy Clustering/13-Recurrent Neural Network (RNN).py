import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic time-series data for demonstration
def generate_data(n_samples=1000, n_timesteps=10, n_features=5, n_classes=3):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_clusters_per_class=1)
    X = X.reshape((n_samples, n_timesteps, n_features))
    return X, y

# Prepare data
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define the RNN model with LSTM cells
def build_rnn_model(input_shape, n_classes):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the RNN model
def train_rnn_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    execution_time = time.time() - start_time
    print(f"RNN training complete. Execution Time: {execution_time:.2f} seconds")
    return model, history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # For clustering metrics (not directly applicable for classification)
    silhouette = silhouette_score(X_test.reshape(X_test.shape[0], -1), y_pred, metric='euclidean') if len(np.unique(y_pred)) > 1 else 'N/A'
    db_index = davies_bouldin_score(X_test.reshape(X_test.shape[0], -1), y_pred) if len(np.unique(y_pred)) > 1 else 'N/A'
    rand_idx = rand_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Silhouette Index: {silhouette}")
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Rand Index: {rand_idx:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# Build and train the model
input_shape = (X_train.shape[1], X_train.shape[2])
n_classes = len(np.unique(y_train))
model = build_rnn_model(input_shape, n_classes)
model, history = train_rnn_model(model, X_train, y_train, X_test, y_test)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
