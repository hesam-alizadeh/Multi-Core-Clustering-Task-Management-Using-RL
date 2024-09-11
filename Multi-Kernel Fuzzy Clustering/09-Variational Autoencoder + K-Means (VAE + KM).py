import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
import time

# Define the Variational Autoencoder (VAE) architecture
class Sampling(layers.Layer):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    
    Args:
        z_mean: Mean of the latent space.
        z_log_var: Logarithm of the variance of the latent space.
    Returns:
        z: Sampled latent vector.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim):
    """Builds the VAE model."""
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation="relu")(encoder_inputs)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation="relu")(latent_inputs)
    x = layers.Dense(512, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
    
    vae = models.Model(encoder_inputs, decoder(decoder(encoder(encoder_inputs)[2])), name="vae")
    reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - decoder_outputs))
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
    return vae, encoder, decoder

# Load and preprocess the dataset
digits = load_digits()
x = digits.data
x = StandardScaler().fit_transform(x)
y = digits.target
input_dim = x.shape[1]
latent_dim = 10

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build and train the VAE
vae, encoder, decoder = build_vae(input_dim, latent_dim)
vae.compile(optimizer="adam")
vae.fit(x_train, x_train, epochs=50, batch_size=128, validation_data=(x_test, x_test))

# Extract the latent space representation
z_train = encoder.predict(x_train)[2]
z_test = encoder.predict(x_test)[2]

# Apply K-Means clustering to the latent space
kmeans = KMeans(n_clusters=10, random_state=42)
y_pred_train = kmeans.fit_predict(z_train)
y_pred_test = kmeans.predict(z_test)

# Evaluation metrics
def evaluate_clustering(y_true, y_pred, x):
    """Evaluate clustering performance."""
    accuracy = accuracy_score(y_true, y_pred)
    silhouette = silhouette_score(x, y_pred)
    davies_bouldin = davies_bouldin_score(x, y_pred)
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

# Measure execution time
start_time = time.time()

# Evaluate on the test set
metrics = evaluate_clustering(y_test, y_pred_test, z_test)

execution_time = time.time() - start_time
print(f'Execution Time: {execution_time:.4f} seconds')
metrics['execution_time'] = execution_time

print("VAE + K-Means clustering complete.")
