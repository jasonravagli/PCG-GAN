import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img


class GANMonitor(Callback):
    def __init__(self, path_imgs_dir, num_img=6, latent_dim=128):
        super(GANMonitor).__init__()

        self.num_img = num_img
        self.latent_dim = latent_dim
        self.path_imgs_dir = path_imgs_dir

        if not os.path.isdir(self.path_imgs_dir):
            os.mkdir(self.path_imgs_dir)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = array_to_img(img)
            img.save(os.path.join(self.path_imgs_dir, f"generated_img_{i}_{epoch}.png"))
