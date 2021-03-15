import os
import matplotlib

from levels.level import Level
from utils.converters import one_hot_to_ascii_level

matplotlib.use('Agg')


class TOADGANMonitor:
    def __init__(self, path_imgs_dir, toadgan, template_level: Level, epochs_interval=500, num_img=3):
        self.toadgan = toadgan
        self.template_level = template_level  # level object to use to render levels
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
                self.template_level.level_oh = self.toadgan.generate_img_at_scale(index_scale)
                self.template_level.level_size = self.template_level.level_oh.shape[:2]
                self.template_level.level_ascii = one_hot_to_ascii_level(self.template_level.level_oh,
                                                                         self.template_level.unique_tokens)
                image = self.template_level.render()
                image.save(os.path.join(path_folder, f"epoch-{epoch}-{i}.png"), format="png")

    def save_reconstructed_images(self):
        path_folder = os.path.join(self.path_imgs_dir, "reconstructed")
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)

        for index_scale in range(len(self.toadgan.list_gans)):
            self.template_level.level_oh = self.toadgan.get_reconstructed_image_at_scale(index_scale)[0]
            self.template_level.level_size = self.template_level.level_oh.shape[:2]
            self.template_level.level_ascii = one_hot_to_ascii_level(self.template_level.level_oh,
                                                                     self.template_level.unique_tokens)
            image = self.template_level.render()
            image.save(os.path.join(path_folder, f"rec-scale-{index_scale}.png"), format="png")
