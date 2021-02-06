import os
import time
import datetime

import tensorflow as tf

from config import cfg
from levels.utils.conversion import one_hot_to_ascii_level, ascii_to_rgb
from levels.utils.downsampling import downsample_image
from models.toadgan_single_scale import TOADGANSingleScale
from training_monitors.toadgan_monitor import TOADGANMonitor
from utils import plot_losses, generate_noise, plot_lr, plot_noise_amplitude


class TOADGAN:
    def __init__(self):
        self.n_scales = 0
        self.scaled_images = None
        self.list_gans = None
        self.tokens_in_lvl = None
        self.token_hierarchy = None

    def train(self, training_image, epochs, tokens_in_lvl, token_hierarchy):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # Disable first GPU
        tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')

        self.tokens_in_lvl = tokens_in_lvl
        self.token_hierarchy = token_hierarchy
        self.scaled_images = self._get_scaled_training_image(training_image)
        self.n_scales = len(self.scaled_images)
        # Reverse the scaled images order: index 0 is for the lowest scale
        self.scaled_images.reverse()

        # Save the scaled images
        for i in range(self.n_scales):
            img = ascii_to_rgb(one_hot_to_ascii_level(self.scaled_images[i], tokens_in_lvl))
            img.save(os.path.join(cfg.PATH.TRAIN.SCALED_IMGS, f"scale-{i}.png"), format="png")

        print(f"Calculated {self.n_scales} for the specified training image")

        # Create the training monitor to save images
        training_monitor = TOADGANMonitor(path_imgs_dir=cfg.PATH.TRAIN.MONITOR_IMGS, toadgan=self, list_tokens_in_level=tokens_in_lvl)

        list_noise_amp = []
        # Create and train the GAN hierarchy
        self.list_gans = []
        print(f"--------- START TRAINING TOAD-GAN ---------\n")
        for index_scale in range(self.n_scales):
            # The reconstruction noise is 0 for all scales except the lowest, which is fixed a priori
            current_scale_reconstruction_noise = self._get_reconstruction_noise_tensor(index_scale)
            current_scale_gan = TOADGANSingleScale(img_shape=self.scaled_images[index_scale].shape,
                                                   index_scale=index_scale,
                                                   get_generated_img_at_scale=self.generate_img_at_scale,
                                                   get_reconstructed_img_at_scale=self.get_reconstructed_image_at_scale,
                                                   reconstruction_noise=current_scale_reconstruction_noise)
            # if index_scale > 0:
            #     current_scale_gan.init_from_previous_scale(prev_scale_gan)
            self.list_gans.append(current_scale_gan)

            print(f"--------- GENERATOR SCALE {index_scale} ---------")
            current_scale_gan.generator.summary()

            print(f"--------- CRITIC SCALE {index_scale} ---------")
            current_scale_gan.critic.summary()

            print(f"--------- START TRAINING GAN AT SCALE {index_scale} ---------")
            start = time.time()
            # c_loss, g_loss = current_scale_gan.train(self.scaled_images[index_scale], epochs, training_monitor)
            c_wass_loss, c_gp_loss, g_adv_loss, g_rec_loss, lr, noise_amp = current_scale_gan.train(self.scaled_images[index_scale],
                                                                                                    epochs, training_monitor)
            end = time.time()
            list_noise_amp.append(noise_amp)
            # Save training curves fot the current scale
            # plot_losses(os.path.join(cfg.PATH.TRAIN.LOSSES, f"{index_scale}.png"), c_loss, g_loss)
            plot_losses(os.path.join(cfg.PATH.TRAIN.LOSSES, f"{index_scale}.png"), c_wass_loss, c_gp_loss, g_adv_loss,
                        g_rec_loss)
            plot_lr(os.path.join(cfg.PATH.TRAIN.DIR, f"lr-{index_scale}.png"), lr)
            print(f"--------- TRAINING ENDED {index_scale} - Took {datetime.timedelta(seconds=int(end-start))} ---------")

            prev_scale_gan = current_scale_gan

        plot_noise_amplitude(os.path.join(cfg.PATH.TRAIN.DIR, f"noise-amp.png"), list_noise_amp)
        # Save the reconstructed images
        training_monitor.save_reconstructed_images()
        print(f"\n--------- END TRAINING TOAD-GAN ---------")

    def generate_image(self):
        """
        Use the gan hierarchy to generate an image
        :return:
        """
        self.generate_img_at_scale(self.n_scales - 1)

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
        while current_scale_height > cfg.CONV_RECEPTIVE_FIELD:
            current_scale_height = int(current_scale_height*cfg.SCALE_FACTOR)
            current_scale_width = int(current_scale_height/aspect_ratio)

            scaled_image = downsample_image(training_image, (current_scale_height, current_scale_width),
                                            self.tokens_in_lvl, self.token_hierarchy)

            scaled_images.append(scaled_image)

        return scaled_images
