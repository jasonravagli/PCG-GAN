import time
import datetime

import tensorflow as tf

from config import cfg
from levels.utils.downsampling import downsample_image
from models.toadgan_single_scale import TOADGANSingleScale
from training_monitors.toadgan_monitor import TOADGANMonitor


class TOADGAN:
    def __init__(self):
        self.n_scales = 0
        self.scaled_images = None
        self.list_gans = None
        self.dict_tokens = None
        self.token_list = None
        self.token_hierarchy = None

    def train(self, training_image, epochs, dict_tokens, token_list, token_hierarchy):
        self.dict_tokens = dict_tokens
        self.token_list = token_list
        self.token_hierarchy = token_hierarchy
        self.scaled_images = self._get_scaled_training_image(training_image)
        self.n_scales = len(self.scaled_images)
        # Reverse the scaled images order: index 0 is for the lowest scale
        self.scaled_images.reverse()

        print(f"Calculated {self.n_scales} for the specified training image")

        # Create the training monitor to save images
        training_monitor = TOADGANMonitor()

        # Create and train the GAN hierarchy
        self.list_gans = []
        print(f"--------- START TRAINING TOAD-GAN ---------\n")
        for index_scale in range(self.n_scales):
            # The reconstruction noise is 0 for all scales except the lowest, which is fixed a priori
            current_scale_reconstruction_noise = self._get_reconstruction_noise_tensor(index_scale)
            current_scale_gan = TOADGANSingleScale(img_shape=self.scaled_images[index_scale].shape,
                                                   index_scale=index_scale,
                                                   get_generated_img_at_scale=self._get_generated_img_at_scale,
                                                   get_reconstructed_img_at_scale=self._get_reconstructed_image_at_scale,
                                                   reconstruction_noise=current_scale_reconstruction_noise)
            print(f"--------- START TRAINING GAN AT SCALE {index_scale} ---------")
            start = time.time()
            current_scale_gan.train(self.scaled_images[index_scale], epochs, training_monitor)
            end = time.time()
            print(f"--------- TRAINING ENDED {index_scale} - Took {datetime.timedelta(seconds=int(end-start))} ---------")

            self.list_gans.append(current_scale_gan)
        print(f"\n--------- END TRAINING TOAD-GAN ---------")

    def generate_image(self):
        """
        Use the gan hierarchy to generate an image
        :return:
        """
        index_curr_scale = 0
        generated_img = self.list_gans[index_curr_scale].generate_image(self._get_random_noise_tensor(index_curr_scale))

        index_curr_scale += 1
        while index_curr_scale < self.n_scales:
            generated_img = self.list_gans[index_curr_scale].generate_image(self._get_random_noise_tensor(index_curr_scale),
                                                                            generated_img)
            index_curr_scale += 1

        return generated_img

    def _get_generated_img_at_scale(self, index_scale):
        """
        Use the gan hierarchy to generate an image at the specified scale
        :param index_scale:
        :return:
        """
        index_curr_scale = 0
        # Generate an image at the lowest scale
        generated_img = self.list_gans[index_curr_scale].generate_image(self._get_random_noise_tensor(index_curr_scale))

        index_curr_scale += 1
        while index_curr_scale <= index_scale:
            generated_img = self.list_gans[index_curr_scale].generate_image(self._get_random_noise_tensor(index_curr_scale),
                                                                                            generated_img)
            index_curr_scale += 1

        return generated_img

    def _get_random_noise_tensor(self, index_scale):
        """
        Return a random (gaussian) noise tensor for the specified scale
        :return:
        """
        # Get a single noise tensor
        # TODO Read the article about the noise generation
        noise = tf.random.normal(
            shape=(1, *self.scaled_images[index_scale].shape)
        )

        return noise

    def _get_reconstructed_image_at_scale(self, index_scale):
        """
        Use the gan hierarchy to reconstruct the original training image at the specified scale
        :param index_scale:
        :return:
        """
        index_curr_scale = 0
        # Reconstruct the original image at the lowest scale
        reconstructed_img = self.list_gans[index_curr_scale].generate_image(self._get_reconstruction_noise_tensor(index_curr_scale))

        index_curr_scale += 1
        while index_curr_scale <= index_scale:
            reconstructed_img = self.list_gans[index_curr_scale].generate_image(self._get_reconstruction_noise_tensor(index_curr_scale),
                                                                                reconstructed_img)
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
            return None

        return tf.zeros(self.scaled_images[index_scale].shape)

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
        while current_scale_height > cfg.CONV_RECEPTIVE_FIELD*2:
            current_scale_height *= cfg.SCALE_FACTOR
            current_scale_width = int(current_scale_height/aspect_ratio)

            scaled_image = downsample_image(training_image, (current_scale_height, current_scale_width), self.token_list,
                                            self.token_hierarchy)

            scaled_images.append(scaled_image)

        return scaled_images