import os

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam

from models.wgan_gp import WGAN
from utils import plot_losses

if __name__ == "__main__":
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    PATH_IMG_LOSSES = os.path.join(ROOT_PATH, "imgs", "train_losses.png")

    batch_size =64
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
    generator_optimizer = Adam(
        learning_rate=0.0001, beta_1=0.5, beta_2=0.9
    )
    critic_optimizer = Adam(
        learning_rate=0.0001, beta_1=0.5, beta_2=0.9
    )
    wgan = WGAN(img_shape=img_shape, latent_dim=latent_dim, c_optimizer=critic_optimizer, g_optimizer=generator_optimizer)
    critic_losses, gen_losses = wgan.train(train_images, epochs, batch_size)
    plot_losses(PATH_IMG_LOSSES, critic_losses, gen_losses)
    print("FINISHED!")
