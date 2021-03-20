import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Softmax
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.layers import ZeroPadding2D
from tensorflow_core.python.keras.models import load_model
from tqdm import trange

from config import cfg
from losses import generator_wass_loss, critic_wass_loss, gradient_penalty_loss, reconstruction_loss
from utils.utils import generate_noise


class TOADGANSingleScale:
    """
    GAN used to generate a level at a certain scale.
    The implementation is based on the WGAN one inside wgan_gp.py
    """

    def __init__(
            self,
            img_shape,
            index_scale,
            get_generated_img_at_scale,
            get_reconstructed_img_at_scale,
            reconstruction_noise,
            critic_steps=3,
            gen_steps=3,
            gp_weight=0.1,
            rec_loss_weight=100,
            n_conv_blocks=3,
    ):
        """
        :param img_shape: Shape of the image to be generated from the GAN
        :param index_scale: Index of the GAN in the TOAD-GAN hierarchy (0 is the lowest scale)
        :param get_img_from_scale: Function to generate an image at a certain scale in the TOAD-GAN hierarchy (it is used
        to generate an image at the previous scale. This is image will be the input to the GAN together with the noise)
        :param get_reconstructed_img_from_scale: Function to get the reconstructed original image (generated from noise)
        at a certain scale (it is used to calculate the reconstruction loss. For further details look at the SinGAN paper)
        :param reconstruction_noise: Noise assigned to this GAN to reconstruct the real image (z* fixed by the TOAD-GAN
        for the lowest scale GAN, 0 for the other GANs)
        :param critic_steps: The training steps for the critic at each epoch (3 according to the original TOAD-GAN code)
        :param gen_steps: The training steps for the generator at each epoch (3 according to the original TOAD-GAN code)
        :param gp_weight: Weight of the gradient penalty term inside the critic loss (typically 10)
        :param rec_loss_weight: Weight of the reconstruction term inside the generator loss
        :param n_conv_blocks: Number of convolutional blocks inside both the generator and the critic (5, as recommended by the SinGAN paper)
        """
        self.img_shape = img_shape
        # Index of the GAN inside the SinGAN cascade (0 is the lowest GAN)
        self.index_scale = index_scale
        self.get_img_from_scale = get_generated_img_at_scale
        self.get_reconstructed_img_from_scale = get_reconstructed_img_at_scale
        self.reconstruction_noise = reconstruction_noise
        self.noise_amplitude = 1
        self.critic_steps = critic_steps
        self.gen_steps = gen_steps
        self.gp_weight = gp_weight
        self.rec_loss_weight = rec_loss_weight
        self.c_optimizer = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.999)
        self.n_conv_blocks = n_conv_blocks

        # Number of filters in the convolutional layers. 32 in the lowest scale, then it doubles every 4 scales
        self.n_conv_filters = 64#((self.index_scale//4) + 1) * 32

        # Calculate the shape of the image in input to the models including the padding
        self.pad_size = 1 * self.n_conv_blocks  # As kernel size is always 3 currently, padsize goes up by one per conv layer
        # self.input_shape_with_pad = (self.img_shape[0] + 2 * self.pad_size, self.img_shape[1] + 2 * self.pad_size, self.img_shape[2])

        self.critic = self._get_critic_model()
        self.generator = self._get_generator_model()

    def init_from_previous_scale(self, prev_gan):
        self.critic.set_weights(prev_gan.critic.get_weights())
        self.generator.set_weights(prev_gan.generator.get_weights())

    def init_generator_from_trained_model(self, path_model):
        scale_generator = load_model(path_model)
        self.generator.set_weights(scale_generator.get_weights())

    # ----------------------------------------
    #          PUBLIC UTILITY FUNCTIONS
    # ----------------------------------------

    def generate_image(self, img_from_prev_scale=None):
        """
        Use the GAN generator to generate an image
        :param noise: Noise to use as input to the GAN
        :param img_from_prev_scale: Input image upsampled from the previous scale. None if the GAN is at the lowest scale in the hierarchy
        :return:
        """
        noise = self.noise_amplitude * generate_noise((1, *self.img_shape))

        if self.index_scale == 0:
            return self.generator(noise)
        else:
            return self.generator([noise, img_from_prev_scale])

    def reconstruct_image(self, img_from_prev_scale=None):
        if self.index_scale == 0:
            return self.generator(self.reconstruction_noise)
        else:
            return self.generator([self.reconstruction_noise, img_from_prev_scale])

    # ----------------------------------------
    #                TRAINING
    # ----------------------------------------

    def train(self, real_image, epochs, singan_monitor):
        # list_c_loss = []
        # list_g_loss = []
        list_c_wass_loss = []
        list_c_gp_loss = []
        list_g_adv_loss = []
        list_g_rec_loss = []
        list_lr = []

        if self.index_scale == 0:
            rec_from_prev_scale = None
        else:
            rec_from_prev_scale = self._get_upsampled_reconstructed_img_from_prev_scale()

        self._calculate_noise_amplitude(real_image, rec_from_prev_scale)
        print(f"Noise amplitude at scale {self.index_scale}: {self.noise_amplitude}")

        # Setup optimizers with learning rate decay
        self.c_optimizer = Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [1600, 2500], [5e-4, 5e-5, 5e-6]), beta_1=0.5, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [1600, 2500], [5e-4, 5e-5, 5e-6]), beta_1=0.5, beta_2=0.999)
        t = trange(epochs, desc="Epoch ")
        for i in t:
            # Get a single noise tensor to use for all the training steps of the current epoch
            noise = self.noise_amplitude * generate_noise((1, *self.img_shape))

            # ----- Train the critic -----
            for _ in range(self.critic_steps):
                # Generate a single fake image to be evaluated by the critic using the fixed noise
                # fake_image = self._generate_image_for_training(noise)
                img_from_prev_scale = self._get_upsampled_img_from_prev_scale()
                img_from_prev_scale = img_from_prev_scale[np.newaxis, :, :, :]
                c_wass_loss, c_gp_loss = self._train_critic_step(real_image, noise, img_from_prev_scale)

                list_c_wass_loss.append(c_wass_loss)
                list_c_gp_loss.append(c_gp_loss)

            # ----- Train the generator -----
            # img_from_prev_scale = self._get_upsampled_img_from_prev_scale()
            for _ in range(self.gen_steps):
                # fake_image = self._generate_image_for_training(noise, img_from_prev_scale)
                g_adv_loss, g_rec_loss = self._train_generator_step(real_image, noise, img_from_prev_scale,
                                                                    rec_from_prev_scale)

                list_g_adv_loss.append(g_adv_loss)
                list_g_rec_loss.append(g_rec_loss)

            # list_c_loss.append(c_loss)
            # list_g_loss.append(g_loss)

            list_lr.append(self.g_optimizer.lr(i).numpy())

            # Print the losses
            t.set_postfix_str(f"Gen. Loss: {list_g_adv_loss[-1]} - Critic Loss: {list_c_wass_loss[-1]}")
            t.refresh()

            singan_monitor.save_imgs_on_epoch_end(index_scale=self.index_scale, epoch=i)

        # Plot images on training end
        singan_monitor.save_imgs_on_epoch_end(index_scale=self.index_scale, epoch=epochs)

        return list_c_wass_loss, list_c_gp_loss, list_g_adv_loss, list_g_rec_loss, list_lr

    @tf.function
    def _train_critic_step(self, real_img, noise, img_from_prev_scale):
        # Convert images to batch of images with one elements (to be fed to the models and the loss functions)
        batch_real_img = real_img[np.newaxis, :, :, :]

        with tf.GradientTape() as tape:
            fake_img = self._generate_image_for_training(noise, img_from_prev_scale)
            batch_fake_img = fake_img[np.newaxis, :, :, :]

            # Get the logits for the fake patches
            fake_logits = self.critic(batch_fake_img, training=True)
            # Get the logits for the real patches
            real_logits = self.critic(batch_real_img, training=True)

            # Calculate the critic loss using the fake and real image logits
            wass_loss = critic_wass_loss(real_score=real_logits, fake_score=fake_logits)
            # Calculate the gradient penalty
            gp = gradient_penalty_loss(batch_size=1, real_images=batch_real_img, fake_images=batch_fake_img,
                                       critic=self.critic)
            # Add the gradient penalty to the original discriminator loss
            loss = wass_loss + gp * self.gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(loss, self.critic.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.c_optimizer.apply_gradients(
            zip(d_gradient, self.critic.trainable_variables)
        )

        return wass_loss, gp * self.gp_weight

    @tf.function
    def _train_generator_step(self, real_img, noise, img_from_prev_scale, rec_from_prev_scale):
        # Convert images to batch of images with one elements (to be fed to the models and the loss functions)
        # batch_fake_img = fake_img[np.newaxis, :, :, :]

        with tf.GradientTape() as tape:
            fake_img = self._generate_image_for_training(noise, img_from_prev_scale)
            batch_fake_img = fake_img[np.newaxis, :, :, :]

            # Get the discriminator logits for fake patches
            fake_logits = self.critic(batch_fake_img, training=True)
            # Calculate the generator adversarial loss
            adv_loss = generator_wass_loss(fake_logits)
            # Calculate the reconstruction loss and update the noise sigma with it
            reconstructed = self._reconstruct_image_for_training(rec_from_prev_scale)
            rec_loss = reconstruction_loss(reconstructed, real_img)

            loss = adv_loss + self.rec_loss_weight * rec_loss
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return adv_loss, self.rec_loss_weight * rec_loss

    def _generate_image_for_training(self, noise, img_from_prev_scale=None):
        # Generate a single fake image from the noise and an upsampled generated image from previous scale
        # (if we are not at the lowest scale)
        if self.index_scale == 0:
            fake_image = self.generator(noise, training=True)[0]
        else:
            # if img_from_prev_scale is None:
            #     # Generate an image with the generator from the previous scale and upsample it
            #     img_from_prev_scale = self._get_upsampled_img_from_prev_scale()
            fake_image = self.generator([noise, img_from_prev_scale], training=True)[0]

        return fake_image

    def _get_upsampled_img_from_prev_scale(self):
        generated = self.get_img_from_scale(self.index_scale - 1)
        upsampled = tf.image.resize(generated, (self.img_shape[0], self.img_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
        return upsampled

    def _get_upsampled_reconstructed_img_from_prev_scale(self):
        reconstructed = self.get_reconstructed_img_from_scale(self.index_scale - 1)
        upsampled = tf.image.resize(reconstructed, (self.img_shape[0], self.img_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
        return upsampled

    def _reconstruct_image_for_training(self, rec_from_prev_scale):
        """
        Differently from _generate_image_for_training(), it uses the generator to create a reconstruction of the original image
        needed for the reconstruction loss calculation (see the original SinGAN paper for further details)
        :return:
        """
        # At the lowest scale the real image is reconstructed with the assigned reconstruction_noise only
        if self.index_scale == 0:
            reconstructed = self.generator(self.reconstruction_noise, training=True)
        else:
            reconstructed = self.generator([self.reconstruction_noise, rec_from_prev_scale], training=True)

        return reconstructed

    def _calculate_noise_amplitude(self, real_img, rec_from_prev_scale):
        if self.index_scale == 0:
            self.noise_amplitude = 1
        else:
            self.noise_amplitude = cfg.TRAIN.NOISE_UPDATE * tf.sqrt(tf.reduce_mean(
                tf.math.squared_difference(real_img, rec_from_prev_scale))).numpy()

    # ----------------------------------------
    #               GENERATOR
    # ----------------------------------------

    def _conv_block(self, x_in, n_conv_filters, activation):
        x_out = Conv2D(filters=n_conv_filters, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False,
                       activation=None, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02))(x_in)
        x_out = BatchNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(mean=1., stddev=0.02))(x_out)
        # x_out = InstanceNorm()(x_out)
        x_out = activation(x_out)

        return x_out

    def _get_generator_model(self):
        noise = Input(shape=self.img_shape)
        x_out = ZeroPadding2D(self.pad_size)(noise)

        # If we are not at the coarsest scale the generator takes as input the sum of the noise and the image generated
        # at the previous scale
        if self.index_scale != 0:
            prev_scale_img = Input(shape=self.img_shape)
            prev_scale_img_padded = ZeroPadding2D(self.pad_size)(prev_scale_img)
            x_out = Add()([x_out, prev_scale_img_padded])
        # else:
        #     x_out = noise

        for i in range(self.n_conv_blocks - 1):
            activation = LeakyReLU(0.2)
            x_out = self._conv_block(x_out, self.n_conv_filters, activation)

        # It is not clear which activation to use in the final convolutional block
        # activation = Activation(activations.linear)
        # The last convolutional block must have the same number of filters as the training image channels
        # x_out = self._conv_block(x_out, self.img_shape[-1], activation)
        # No batch normalization and activation on the last convolutional layer
        x_out = Conv2D(filters=self.img_shape[-1], kernel_size=(3, 3), strides=(1, 1), padding="valid", #activation="tanh",
                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02))(x_out)

        # If we are not at the coarsest scale the output have to be summed up with the image generated at the previous scale
        # Softmax is used as final layer (we are treating tensors of one-hot encoded vectors, not natural images)
        if self.index_scale != 0:
            x_out = Softmax()(x_out)
            x_out = Add()([x_out, prev_scale_img])
            g_model = Model([noise, prev_scale_img], x_out, name=f"generator_{self.index_scale}")
        else:
            x_out = Softmax()(x_out)
            g_model = Model(noise, x_out, name=f"generator_{self.index_scale}")
        return g_model

    # ----------------------------------------
    #                  CRITIC
    # ----------------------------------------

    def _get_critic_model(self):
        patch = Input(shape=self.img_shape)
        x_out = ZeroPadding2D(self.pad_size)(patch)
        for i in range(self.n_conv_blocks - 1):
            activation = LeakyReLU(0.2)
            x_out = self._conv_block(x_out, self.n_conv_filters, activation)

        # x_out = Flatten()(x_out)
        # Keep the dropout?
        # x_out = Dropout(0.2)(x_out)
        # x_out = Dense(1)(x_out)
        x_out = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x_out)

        c_model = Model(patch, x_out, name=f"critic_{self.index_scale}")
        return c_model
