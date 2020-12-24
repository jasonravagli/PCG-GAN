import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img


class GANMonitor:
    def __init__(self, path_imgs_dir, generator, num_img=6, latent_dim=128):
        self.generator = generator
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.path_imgs_dir = path_imgs_dir

        if not os.path.isdir(self.path_imgs_dir):
            os.mkdir(self.path_imgs_dir)

    def save_imgs_on_epoch_end(self, epoch):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        fig, axs = plt.subplots(1, self.num_img, figsize=(12, 9))

        for i in range(self.num_img):
            # Training is done in eager mode: tensors must be evaluated
            img = generated_images[i].numpy()
            img = array_to_img(img)
            axs[i].imshow(img, cmap="gray")

        fig.savefig(os.path.join(self.path_imgs_dir, f"generated_img_{epoch}.png"))


def plot_losses(path_file, critic_loss, gen_loss):
    iterations = range(len(critic_loss))

    fig = plt.figure()
    plt.plot(iterations, critic_loss, color="r", label="Critic Loss")
    plt.plot(iterations, gen_loss, color="b", label="Gen. Loss")
    plt.legend()
    fig.savefig(path_file)
