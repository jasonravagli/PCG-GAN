import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import activations
from tensorflow.keras.layers import Activation, Add, Softmax
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from losses import generator_wass_loss, critic_wass_loss, gradient_penalty_loss, reconstruction_loss


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
            gp_weight=10.0,
            rec_loss_weight=100,
            n_conv_blocks=5,
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
        self.critic_steps = critic_steps
        self.gen_steps = gen_steps
        self.gp_weight = gp_weight
        self.rec_loss_weight = rec_loss_weight
        self.c_optimizer = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.999)
        self.n_conv_blocks = n_conv_blocks

        # Number of filters in the convolutional layers. 32 in the lowest scale, then it doubles every 4 scales
        self.n_conv_filters = ((self.index_scale % 4) + 1) * 32

        self.critic = self._get_critic_model()
        self.generator = self._get_generator_model()

    # ----------------------------------------
    #          PUBLIC UTILITY FUNCTIONS
    # ----------------------------------------

    def generate_image(self, noise, input_image=None):
        """
        Use the GAN generator to generate an image
        :param noise: Noise to use as input to the GAN
        :param input_image: Input image upsampled from the previous scale. None if the GAN is at the lowest scale in the hierarchy
        :return:
        """
        if self.index_scale == 0:
            return self.generator(noise)
        else:
            return self.generator([noise, input_image])

    # ----------------------------------------
    #                TRAINING
    # ----------------------------------------

    # def train(self, real_img, epochs, batch_size):
    #     # Get all the patches from the real image
    #     real_patches = get_all_patches_from_img(real_img, self.patch_shape[0], self.patch_shape[1])
    #
    #     n_batches = real_patches.shape[0]//batch_size
    #     steps_per_epoch = n_batches//self.c_steps
    #     list_c_loss = []
    #     list_g_loss = []
    #     gan_monitor = GANMonitor(path_imgs_dir="imgs", generator=self.generator, num_img=3, latent_dim=self.latent_dim)
    #
    #     for i in range(epochs):
    #         print(f"-------- Epoch {i} --------")
    #         np.random.shuffle(real_patches)
    #         # For each epoch, all batches of real images are shown to the critic once
    #         batch_index = 0
    #         for _ in tqdm(range(steps_per_epoch)):
    #             # Generate a single fake image and get all patches from it
    #             fake_image = self._generate_image()
    #             fake_patches = get_all_patches_from_img(fake_image, self.patch_shape[0], self.patch_shape[1])
    #
    #             for _ in range(self.c_steps):
    #                 batch_real_patches = real_patches[batch_index:batch_index + batch_size]
    #                 # TODO Fake patches can be sampled in a smarter way
    #                 batch_fake_patches = fake_patches[np.random.randint(0, fake_patches[0], batch_size)]
    #                 c_loss = self._train_critic_on_batch(batch_real_patches, batch_fake_patches, batch_size)
    #
    #                 batch_index = (batch_index + 1) % steps_per_epoch  # Use module operation to avoid index out of range errors
    #
    #             # Train the generator
    #             fake_image = self._generate_image()
    #             fake_patches = get_all_patches_from_img(fake_image, self.patch_shape[0], self.patch_shape[1])
    #             g_loss = self._train_generator_on_batch(fake_patches)
    #
    #             list_c_loss.append(c_loss)
    #             list_g_loss.append(g_loss)
    #
    #         print(f"Critic Loss: {list_c_loss[-1]} - Gen. Loss: {list_g_loss[-1]}")
    #
    #         gan_monitor.save_imgs_on_epoch_end(epoch=i)
    #
    #     return list_c_loss, list_g_loss

    # @tf.function
    # def _train_critic_on_batch(self, batch_real_patches, batch_fake_patches, batch_size):
    #     with tf.GradientTape() as tape:
    #         # Get the logits for the fake patches
    #         fake_logits = self.critic(batch_fake_patches, training=True)
    #         # Get the logits for the real patches
    #         real_logits = self.critic(batch_real_patches, training=True)
    #
    #         # Calculate the critic loss using the fake and real image logits
    #         wass_loss = critic_wass_loss(real_score=real_logits, fake_score=fake_logits)
    #         # Calculate the gradient penalty
    #         gp = gradient_penalty_loss(batch_size, batch_real_patches, batch_fake_patches, self.critic)
    #         # Add the gradient penalty to the original discriminator loss
    #         loss = wass_loss + gp * self.gp_weight
    #
    #     # Get the gradients w.r.t the discriminator loss
    #     d_gradient = tape.gradient(loss, self.critic.trainable_variables)
    #     # Update the weights of the discriminator using the discriminator optimizer
    #     self.c_optimizer.apply_gradients(
    #         zip(d_gradient, self.critic.trainable_variables)
    #     )
    #
    #     return loss

    def train(self, real_image, epochs, singan_monitor):
        list_c_loss = []
        list_g_loss = []

        t = trange(epochs, desc="Epoch ")
        for i in t:
            # Train the critic
            for _ in range(self.critic_steps):
                # Generate a single fake image to be evaluated by the critic
                fake_image = self._generate_image()
                c_loss = self._train_critic_step(real_image, fake_image)

            # Train the generator
            for _ in range(self.gen_steps):
                fake_image = self._generate_image()
                g_loss = self._train_generator_step(fake_image, real_image)

            list_c_loss.append(c_loss)
            list_g_loss.append(g_loss)

            # Print the losses
            t.set_postfix_str(f"Gen. Loss: {list_g_loss[-1]} - Critic Loss: {list_c_loss[-1]}")
            t.refresh()

            singan_monitor.save_imgs_on_epoch_end(index_scale=self.index_scale, epoch=i)

        # Plot images on training end
        singan_monitor.save_imgs_on_epoch_end(index_scale=self.index_scale, epoch=epochs)

        return list_c_loss, list_g_loss

    @tf.function
    def _train_critic_step(self, real_img, fake_img):
        # Convert images to batch of images with one elements (to be fed to the models and the loss functions)
        batch_real_img = real_img[np.newaxis, :, :, :]
        batch_fake_img = fake_img[np.newaxis, :, :, :]

        with tf.GradientTape() as tape:
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

        return loss

    @tf.function
    def _train_generator_step(self, fake_img, real_img):
        # Convert images to batch of images with one elements (to be fed to the models and the loss functions)
        batch_fake_img = fake_img[np.newaxis, :, :, :]

        with tf.GradientTape() as tape:
            # Get the discriminator logits for fake patches
            fake_logits = self.critic(batch_fake_img, training=True)
            # Calculate the generator adversarial loss
            adv_loss = generator_wass_loss(fake_logits)
            # Calculate the reconstruction loss and update the noise sigma with it
            reconstructed = self._reconstruct_image()
            rec_loss = reconstruction_loss(reconstructed, real_img)
            # TODO update the noise sigma

            loss = adv_loss + self.rec_loss_weight * rec_loss
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return loss

    def _generate_image(self):
        # Get a single noise tensor
        # TODO Read the article about the noise generation
        noise = tf.random.normal(
            shape=(1, *self.img_shape)
        )

        # Generate a single fake image from the noise and an upsampled generated image from previous scale (if we are not at the lowest scale)
        if self.index_scale == 0:
            fake_image = self.generator(noise, training=True)[0]
        else:
            # Generate an image with the generator from the previous scale and upsample it
            img_from_prev_scale = self._get_upsampled_img_from_prev_scale()
            fake_image = self.generator([noise, img_from_prev_scale], training=True)[0]

        return fake_image

    def _reconstruct_image(self):
        """
        Differently from _generate_image(), it uses the generator to create a reconstruction of the original image
        needed for the reconstruction loss calculation (see the original SinGAN paper for further details)
        :return:
        """
        # At the lowest scale the real image is reconstructed with the assigned reconstruction_noise only
        if self.index_scale == 0:
            reconstructed = self.generator(self.reconstruction_noise, training=True)
        else:
            prev_scale_reconstructed = self._get_upsampled_reconstructed_img_from_prev_scale()
            reconstructed = self.generator([self.reconstruction_noise, prev_scale_reconstructed], training=True)

        return reconstructed

    def _get_upsampled_img_from_prev_scale(self):
        generated = self.get_img_from_scale(self.index_scale - 1)
        upsampled = tf.image.resize(generated, (self.img_shape[0], self.img_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
        return upsampled

    def _get_upsampled_reconstructed_img_from_prev_scale(self):
        reconstructed = self.get_reconstructed_img_from_scale(self.index_scale - 1)
        upsampled = tf.image.resize(reconstructed, (self.img_shape[0], self.img_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
        return upsampled

    # ----------------------------------------
    #               GENERATOR
    # ----------------------------------------

    def _conv_block(self, x_in, n_conv_filters, activation):
        x_out = Conv2D(filters=n_conv_filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None)(x_in)
        x_out = BatchNormalization()(x_out)
        x_out = activation(x_out)

        return x_out

    def _get_generator_model(self):
        noise = Input(shape=self.img_shape)

        # If we are not at the coarsest scale the generator takes as input the sum of the noise and the image generated
        # at the previous scale
        if self.index_scale != 0:
            prev_scale_img = Input(shape=self.img_shape)
            x_out = Add()([noise, prev_scale_img])
        else:
            x_out = noise

        for i in range(self.n_conv_blocks - 1):
            activation = LeakyReLU(0.2)
            x_out = self._conv_block(x_out, self.n_conv_filters, activation)

        # It is not clear which activation to use in the final convolutional block
        activation = Activation(activations.linear)
        # The last convolutional block must have the same number of filters as the training image channels
        x_out = self._conv_block(x_out, self.img_shape[-1], activation)

        # If we are not at the coarsest scale the output have to be summed up with the image generated at the previous scale
        # Softmax is used as final layer (we are treating tensors of one-hot encoded vectors, not natural images)
        if self.index_scale != 0:
            x_out = Add()([x_out, prev_scale_img])
            x_out = Softmax()(x_out)
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
        x_out = patch
        for i in range(self.n_conv_blocks):
            activation = LeakyReLU(0.2)
            x_out = self._conv_block(x_out, self.n_conv_filters, activation)

        x_out = Flatten()(x_out)
        # Keep the dropout?
        # x_out = Dropout(0.2)(x_out)
        x_out = Dense(1)(x_out)

        c_model = Model(patch, x_out, name=f"critic_{self.index_scale}")
        return c_model
