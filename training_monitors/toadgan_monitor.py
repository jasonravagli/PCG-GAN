import os
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

from levels.utils.conversion import one_hot_to_ascii, ascii_to_rgb


class TOADGANMonitor:
    def __init__(self, path_imgs_dir, singan, list_tokens_in_level, epochs_interval=20, num_img=6):
        self.singan = singan
        self.list_tokens_in_level = list_tokens_in_level
        self.num_img = num_img
        self.epochs_interval = epochs_interval
        self.path_imgs_dir = path_imgs_dir

        if not os.path.isdir(self.path_imgs_dir):
            os.mkdir(self.path_imgs_dir)

    def save_imgs_on_epoch_end(self, index_scale, epoch):
        """
        Generates and save some images using the generator
        :param index_scale:
        :param epoch:
        :return:
        """
        # Generate and save the images only at the chosen epochs checkpoints
        if epoch % self.epochs_interval == 0:
            fig, axs = plt.subplots(1, self.num_img, figsize=(12, 9))

            for i in range(self.num_img):
                image_one_hot = self.singan.generate_image()
                image_tokenized = one_hot_to_ascii(image_one_hot, self.list_tokens_in_level)
                image_rgb = ascii_to_rgb(image_tokenized)

                img = array_to_img(image_rgb)
                axs[i].imshow(img)

            path_folder = os.path.join(self.path_imgs_dir, index_scale)
            if not os.path.isdir(path_folder):
                os.mkdir(path_folder)

            fig.savefig(os.path.join(path_folder, f"generated_img_{epoch}.png"))
