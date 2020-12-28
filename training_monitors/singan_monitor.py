import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt


class SinGANMonitor:
    def __init__(self, path_imgs_dir, generator, img_shape, num_img=6):
        self.generator = generator
        self.num_img = num_img
        self.img_shape = img_shape
        self.path_imgs_dir = path_imgs_dir

        if not os.path.isdir(self.path_imgs_dir):
            os.mkdir(self.path_imgs_dir)

    def save_imgs_on_epoch_end(self, index_scale, epoch):
        # TODO adapt image creation (from one-hot to image)
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        # generated_images = (generated_images * 127.5) + 127.5

        fig, axs = plt.subplots(1, self.num_img, figsize=(12, 9))

        for i in range(self.num_img):
            # Training is done in eager mode: tensors must be evaluated
            img = generated_images[i].numpy()
            img = array_to_img(img)
            axs[i].imshow(img, cmap="gray")

        path_folder = os.path.join(self.path_imgs_dir, index_scale)
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)

        fig.savefig(os.path.join(path_folder, f"generated_img_{epoch}.png"))
