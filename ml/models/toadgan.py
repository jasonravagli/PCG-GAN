import os
import time
import datetime

import tensorflow as tf

from config import cfg
from levels.level import Level
from ml.models.toadgan_single_scale import TOADGANSingleScale
from ml.training_info import TrainingInfo
from ml.training_monitors.toadgan_monitor import TOADGANMonitor
from utils.downsampling import downsample_image
from utils.utils import plot_losses, plot_lr, plot_noise_amplitude
from utils.converters import one_hot_to_ascii_level


class TOADGAN:
    def __init__(self):
        self.conv_receptive_field = 0
        self.list_gans = None
        self.n_scales = 0
        self.scaled_images = None
        self.scale_factor = None
        self.tokens_in_lvl = None
        self.token_hierarchy = None
        self.list_reconstruction_noise = None
        self.list_training_info = None

    def setup_network(self, level: Level, conv_receptive_field, scale_factor):
        self.tokens_in_lvl = level.unique_tokens
        self.token_hierarchy = level.tokenset.token_hierarchy

        # Set fields to determine the scaling hierarchy
        self.conv_receptive_field = conv_receptive_field
        self.scale_factor = scale_factor
        self.scaled_images = self._get_scaled_training_image(level.level_oh)
        self.n_scales = len(self.scaled_images)
        # Reverse the scaled images order: index 0 is for the lowest scale
        self.scaled_images.reverse()

    def train(self, level: Level, epochs: int):
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # # Disable first GPU
        # tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')

        self.setup_network(level, cfg.CONV_RECEPTIVE_FIELD, cfg.SCALE_FACTOR)

        # Create a template level to use to render images
        template_level = level.copy()
        # Render and save scaled levels
        for i in range(self.n_scales):
            template_level.level_oh = self.scaled_images[i]
            template_level.level_size = template_level.level_oh.shape[:2]
            template_level.level_ascii = one_hot_to_ascii_level(template_level.level_oh, template_level.unique_tokens)
            img = template_level.render()
            img.save(os.path.join(cfg.PATH.TRAIN.SCALED_IMGS, f"scale-{i}.png"), format="png")

        print(f"Calculated {self.n_scales} scales for the specified training image")

        # Create the training monitor to save images
        training_monitor = TOADGANMonitor(path_imgs_dir=cfg.PATH.TRAIN.MONITOR_IMGS, toadgan=self,
                                          template_level=template_level)

        # list_noise_amp = []
        # Create and train the GAN hierarchy
        self.list_gans = []
        self.list_training_info = []
        self.list_reconstruction_noise = []
        print(f"--------- START TRAINING TOAD-GAN ---------\n")
        for index_scale in range(self.n_scales):
            # The reconstruction noise is 0 for all scales except the lowest, which is fixed a priori
            current_scale_reconstruction_noise = self._get_reconstruction_noise_tensor(index_scale)
            self.list_reconstruction_noise.append(current_scale_reconstruction_noise.numpy())

            current_scale_gan = TOADGANSingleScale(img_shape=self.scaled_images[index_scale].shape,
                                                   index_scale=index_scale,
                                                   get_generated_img_at_scale=self.generate_img_at_scale,
                                                   get_reconstructed_img_at_scale=self.get_reconstructed_image_at_scale,
                                                   reconstruction_noise=current_scale_reconstruction_noise)

            # The generated level seem to be cleaner with this initialization
            if index_scale > 0:
                current_scale_gan.init_from_previous_scale(prev_scale_gan)
            self.list_gans.append(current_scale_gan)

            # print(f"--------- GENERATOR SCALE {index_scale} ---------")
            # current_scale_gan.generator.summary()
            #
            # print(f"--------- CRITIC SCALE {index_scale} ---------")
            # current_scale_gan.critic.summary()

            print(f"--------- START TRAINING GAN AT SCALE {index_scale} ---------")
            start = time.time()
            c_wass_loss, c_gp_loss, g_adv_loss, g_rec_loss, lr = current_scale_gan.train(self.scaled_images[index_scale],
                                                                                         epochs, training_monitor)
            end = time.time()
            # list_noise_amp.append(noise_amp)
            # plot_losses(os.path.join(cfg.PATH.TRAIN.LOSSES, f"{index_scale}.png"), c_wass_loss, c_gp_loss, g_adv_loss,
            #             g_rec_loss)
            # plot_lr(os.path.join(cfg.PATH.TRAIN.DIR, f"lr-{index_scale}.png"), lr)
            print(f"--------- TRAINING ENDED {index_scale} - Took {datetime.timedelta(seconds=int(end-start))} ---------")

            # Save scale training info
            training_info = TrainingInfo()
            training_info.lr = lr
            training_info.loss_critic_wass = c_wass_loss
            training_info.loss_critic_gp = c_gp_loss
            training_info.loss_gen_adversarial = g_adv_loss
            training_info.loss_gen_reconstruction = g_rec_loss
            self.list_training_info.append(training_info)

            prev_scale_gan = current_scale_gan

        # plot_noise_amplitude(os.path.join(cfg.PATH.TRAIN.DIR, f"noise-amp.png"), list_noise_amp)

        # Save the reconstructed images
        # training_monitor.save_reconstructed_images()
        print(f"\n--------- END TRAINING TOAD-GAN ---------")

    def generate_image(self):
        """
        Use the gan hierarchy to generate an image
        :return:
        """
        return self.generate_img_at_scale(self.n_scales - 1)

    def generate_img_at_scale(self, index_scale):
        """
        Use the gan hierarchy to generate an image at the specified scale
        :param index_scale:
        :return:
        """
        index_curr_scale = 0
        # Generate an image at the lowest scale
        generated_img = self.list_gans[index_curr_scale].generate_image()

        index_curr_scale += 1
        while index_curr_scale <= index_scale:
            upsampled = tf.image.resize(generated_img, (self.scaled_images[index_curr_scale].shape[0], self.scaled_images[index_curr_scale].shape[1]),
                                        method=tf.image.ResizeMethod.BILINEAR)
            generated_img = self.list_gans[index_curr_scale].generate_image(upsampled)
            index_curr_scale += 1

        return generated_img[0]

    def get_reconstructed_image_at_scale(self, index_scale):
        """
        Use the gan hierarchy to reconstruct the original training image at the specified scale
        :param index_scale:
        :return:
        """
        index_curr_scale = 0
        # Reconstruct the original image at the lowest scale
        reconstructed_img = self.list_gans[index_curr_scale].reconstruct_image()

        index_curr_scale += 1
        while index_curr_scale <= index_scale:
            upsampled = tf.image.resize(reconstructed_img, (self.scaled_images[index_curr_scale].shape[0], self.scaled_images[index_curr_scale].shape[1]),
                                        method=tf.image.ResizeMethod.BILINEAR)
            reconstructed_img = self.list_gans[index_curr_scale].reconstruct_image(upsampled)
            index_curr_scale += 1

        return reconstructed_img

    def _get_reconstruction_noise_tensor(self, index_scale):
        """
        Return the appropriate reconstruction noise tensor for the specified scale. It is zero for all scales except the
        lowest. This is required for the reconstruction term of the training loss (for further details see the original SinGAN paper)
        :param index_scale:
        :return:
        """
        if index_scale == 0:
            return tf.random.normal(
                shape=(1, *self.scaled_images[0].shape)
            )

        return tf.zeros((1, *self.scaled_images[index_scale].shape))

    def _get_scaled_training_image(self, training_image):
        """
        Calculates the number of scales required (i.e. the number of gans in the SinGAN hierarchy) from the training image shape.
        The lowest scale gan filters must cover about half of the image, and their global receptive field is 11x11 in the final conv block.
        It returns the training_image scaled at the calculated scales.
        :param training_image:
        :return:
        """
        aspect_ratio = training_image.shape[0]/float(training_image.shape[1])
        scaled_images = [training_image]
        current_scale_height = training_image.shape[0]
        while current_scale_height > self.conv_receptive_field:
            current_scale_height = round(current_scale_height*self.scale_factor)
            current_scale_width = round(current_scale_height/aspect_ratio)

            scaled_image = downsample_image(training_image, (current_scale_height, current_scale_width),
                                            self.tokens_in_lvl, self.token_hierarchy)

            scaled_images.append(scaled_image)

        return scaled_images
