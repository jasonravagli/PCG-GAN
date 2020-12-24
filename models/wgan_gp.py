import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D, Activation, Reshape, UpSampling2D, Cropping2D
from tqdm import tqdm

from losses import generator_wass_loss, critic_wass_loss
from utils import GANMonitor


class WGAN:
    """
    Original implementation:
    https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb
    """

    def __init__(
            self,
            img_shape,
            latent_dim,
            c_optimizer,
            g_optimizer,
            critic_extra_steps=3,
            gp_weight=10.0,
    ):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.c_steps = critic_extra_steps
        self.gp_weight = gp_weight
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

        self.critic = self._get_critic_model()
        self.generator = self._get_generator_model()

    # ----------------------------------------
    #                TRAINING
    # ----------------------------------------

    def train(self, real_imgs, epochs, batch_size):
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        n_batches = real_imgs.shape[0]//batch_size
        steps_per_epoch = n_batches//self.c_steps
        list_c_loss = []
        list_g_loss = []
        gan_monitor = GANMonitor(path_imgs_dir="imgs", generator=self.generator, num_img=3, latent_dim=self.latent_dim)

        for i in range(epochs):
            print(f"-------- Epoch {i} --------")
            np.random.shuffle(real_imgs)
            # For each epoch, all batches of real images are shown to the critic once
            batch_index = 0
            for _ in tqdm(range(steps_per_epoch)):

                # Train the discriminator first. The original paper recommends training
                # the discriminator for `x` more steps (typically 5) as compared to
                # one step of the generator. Here we will train it for 3 extra steps
                # as compared to 5 to reduce the training time.
                for _ in range(self.c_steps):
                    batch_real_imgs = real_imgs[batch_index:batch_index + batch_size]
                    c_loss = self._train_critic_on_batch(batch_real_imgs, batch_size)

                    batch_index += 1

                # Train the generator
                g_loss = self._train_generator_on_batch(batch_size)

                list_c_loss.append(c_loss)
                list_g_loss.append(g_loss)

            print(f"Critic Loss: {list_c_loss[-1]} - Gen. Loss: {list_g_loss[-1]}")

            gan_monitor.save_imgs_on_epoch_end(epoch=i)

        return list_c_loss, list_g_loss

    @tf.function
    def _train_critic_on_batch(self, batch_real_imgs, batch_size):
        # Get the latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator(random_latent_vectors, training=True)
            # Get the logits for the fake images
            fake_logits = self.critic(fake_images, training=True)
            # Get the logits for the real images
            real_logits = self.critic(batch_real_imgs, training=True)

            # Calculate the critic loss using the fake and real image logits
            wass_loss = critic_wass_loss(real_score=real_logits, fake_score=fake_logits)
            # Calculate the gradient penalty
            gp = self._gradient_penalty(batch_size, batch_real_imgs, fake_images)
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
    def _train_generator_on_batch(self, batch_size):
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.critic(generated_images, training=True)
            # Calculate the generator loss
            loss = generator_wass_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return loss

    def _gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = real_images - fake_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # ----------------------------------------
    #               GENERATOR
    # ----------------------------------------

    def _upsample_block(
            self,
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            up_size=(2, 2),
            padding="same",
            use_bn=False,
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
    ):
        x = UpSampling2D(up_size)(x)
        x = Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)

        if use_bn:
            x = BatchNormalization()(x)

        if activation:
            x = activation(x)
        if use_dropout:
            x = Dropout(drop_value)(x)
        return x

    def _get_generator_model(self):
        noise = Input(shape=(self.latent_dim,))
        x = Dense(4 * 4 * 256, use_bias=False)(noise)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Reshape((4, 4, 256))(x)
        x = self._upsample_block(
            x,
            128,
            LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self._upsample_block(
            x,
            64,
            LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self._upsample_block(
            x, 1, Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
        )
        # At this point, we have an output which has the same shape as the input, (32, 32, 1).
        # We will use a Cropping2D layer to make it (28, 28, 1).
        x = Cropping2D((2, 2))(x)

        g_model = Model(noise, x, name="generator")
        return g_model

    # ----------------------------------------
    #                  CRITIC
    # ----------------------------------------

    def _conv_block(
            self,
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            use_bn=False,
            use_dropout=True,
            drop_value=0.5,
    ):
        x = Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = Dropout(drop_value)(x)
        return x

    def _get_critic_model(self):
        """
        N.B Avoid batch normalization for critic in WGAN-GP, as explained here:
        https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
        Use dropout instead.
        :return:
        """

        img_input = Input(shape=self.img_shape)
        # Zero pad the input to make the input images size to (32, 32, 1).
        x = ZeroPadding2D((2, 2))(img_input)
        x = self._conv_block(
            x,
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        x = self._conv_block(
            x,
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self._conv_block(
            x,
            256,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self._conv_block(
            x,
            512,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )

        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)

        d_model = Model(img_input, x, name="critic")
        return d_model


