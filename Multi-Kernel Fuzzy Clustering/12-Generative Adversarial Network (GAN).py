import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, f1_score

# Load and preprocess the dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train - 127.5) / 127.5  # Normalize to [-1, 1]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # Reshape to (batch_size, height, width, channels)

# Define the GAN components
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

# Training the GAN
def train_gan(epochs=10000, batch_size=64, save_interval=1000):
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    # Labels for real and fake data
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    start_time = time.time()

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        real_labels_batch = real_labels
        fake_images = generator.predict(np.random.randn(batch_size, 100))
        fake_labels_batch = fake_labels

        d_loss_real = discriminator.train_on_batch(real_images, real_labels_batch)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels_batch)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        g_loss = gan.train_on_batch(np.random.randn(batch_size, 100), real_labels)

        if epoch % save_interval == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
    
    execution_time = time.time() - start_time
    print(f"GAN training complete. Execution Time: {execution_time:.2f} seconds")
    return generator

def evaluate_gan(generator, num_samples=1000):
    noise = np.random.randn(num_samples, 100)
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]

    # Visualization of generated images
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(generated_images[i*10 + j].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
    plt.show()

# Run training
generator = train_gan()

# Evaluate
evaluate_gan(generator)
