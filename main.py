import os

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import RMSprop

from losses import wasserstein_loss
from models.wgan_gp import WGAN

if __name__ == "__main__":
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    batch_size = 256
    # Size of the noise vector
    latent_dim = 128

    # Load the MNIST-fashion dataset
    img_shape = (28, 28, 1)
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(f"Number of examples: {len(train_images)}")
    print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

    # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
    train_images = train_images.reshape(train_images.shape[0], *img_shape).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    # Create and train the network
    epochs = 1
    optimizer = RMSprop(0.0005)
    wgan = WGAN(img_shape=img_shape, latent_dim=latent_dim, c_optimizer=optimizer, g_optimizer=optimizer)
    wgan.train(train_images, epochs, batch_size)
    print("FINISHED!")
