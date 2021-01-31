import os
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from levels.utils.conversion import one_hot_to_ascii_level, ascii_to_rgb


class TOADGANMonitor:
    def __init__(self, path_imgs_dir, singan, list_tokens_in_level, epochs_interval=100, num_img=3):
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
            path_folder = os.path.join(self.path_imgs_dir, str(index_scale))
            if not os.path.isdir(path_folder):
                os.mkdir(path_folder)

            for i in range(self.num_img):
                image_one_hot = self.singan.generate_img_at_scale(index_scale)
                image_tokenized = one_hot_to_ascii_level(image_one_hot, self.list_tokens_in_level)
                image_rgb = ascii_to_rgb(image_tokenized)
                image_rgb.save(os.path.join(path_folder, f"epoch-{epoch}-{i}.png"), format="png")
