import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, Input, \
    LeakyReLU, Flatten, Layer
from tensorflow.keras.optimizers import RMSprop
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.layers import Dropout
from tqdm import tqdm
from functools import partial

from losses import wasserstein_loss, wass_loss_with_gradient_penalty, gradient_penalty_loss


class RandomWeightedAvg(Layer):
    def __init__(self, batch_size):
        super(RandomWeightedAvg, self).__init__()

        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        x_real = inputs[0]
        x_fake = inputs[1]
        epsilon = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        return epsilon * x_real + (1 - epsilon) * x_fake


class WGANGP:
    """
    Reference: https://github.com/timsainb/tensorflow2-generative-models
    """

    def __init__(self, dim_latent, img_shape, batch_size):
        self.dim_latent = dim_latent
        self.img_shape = img_shape
        self.batch_size = batch_size

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.optimizer = RMSprop(lr=0.00005)

        self.generator, self.critic = self._build_models()

        self.generator.summary()
        self.critic.summary()

    def _build_generator(self):
        model = Sequential()

        # ------ Input Block ------
        model.add(Dense(7 * 7 * 128, input_dim=self.dim_latent))
        model.add(LeakyReLU(0.2))
        model.add(Reshape((7, 7, 128)))
        # ------ Block 1 ------
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        # ------ Block 2 ------
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        # ------ Output Block ------
        model.add(Conv2D(self.img_shape[2], kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        return model

    def _build_critic(self):
        """
        N.B Avoid batch normalization when using WGAN-GP, as explained here:
        https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
        Use dropout instead.
        :return:
        """

        model = Sequential()
        # ------ Conv Block 1 ------
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        # ------ Conv Block 2 ------
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # ------ Conv Block 3 ------
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # ------ Conv Block 4 ------
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="linear"))

        return model

    def _build_models(self):
        generator = self._build_generator()
        critic = self._build_critic()

        return self._compile_models(generator, critic)

    def _compile_models(self, generator, critic):
        #  -------------------------------
        #    Construct the critic model
        #  -------------------------------
        # Must include the (non-trainable) generator in the constructor model to use the wasserstein loss with gradient
        # penalty directly in Keras

        # Freeze generator's layers while training critic
        generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_critic = Input(shape=(self.dim_latent,))
        # Generate image based of noise (fake sample)
        fake_img = generator(z_critic)

        # Random weighted images
        rnd_avg_img = RandomWeightedAvg(self.batch_size)([real_img, fake_img])

        # Compute score for real and generated images
        real_score = critic(real_img)
        fake_score = critic(fake_img)
        rnd_avg_score = critic(rnd_avg_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        wass_gp_loss = partial(wass_loss_with_gradient_penalty,
                                  x_real=real_img, x_fake=fake_img, critic=critic, gp_weight=10)
        wass_gp_loss.__name__ = 'wass_gp'  # Keras requires function names

        partial_gp_loss = partial(gradient_penalty_loss, x_averaged=rnd_avg_img)

        compiled_critic = Model(inputs=[real_img, z_critic], outputs=[real_score, fake_score, rnd_avg_score])
        compiled_critic.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],#wass_loss_with_gradient_penalty(real_img, fake_img, critic),
                                optimizer=self.optimizer, metrics=['accuracy'])

        #  ---------------------------------
        #    Construct the generator model
        #  ---------------------------------
        # Freeze critic's layers while training generator
        critic.trainable = False
        generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.dim_latent,))
        # Generate images based of noise
        img = generator(z_gen)
        # Discriminator determines validity
        score = critic(img)
        # Defines generator model
        compiled_gen = Model(z_gen, score)
        compiled_gen.compile(loss=wasserstein_loss, optimizer=self.optimizer)

        return compiled_gen, compiled_critic

    def train(self, X_train, epochs, batch_size):
        critic_losses = []
        gen_losses = []
        n_batches = X_train.shape[0] // batch_size

        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {n_batches}")

        # Ground truths
        y_real = np.ones((batch_size, 1))
        y_fake = -np.ones((batch_size, 1))
        y_dummy = np.zeros((batch_size, 1))  # Dummy labels for the gradient penalty loss term
        for e in range(1, epochs+1):
            print('-'*15, 'Epoch %d' % e, '-'*15)
            for batch_index in tqdm(range(n_batches)):

                # ---------------------
                #     Train Critic
                # ---------------------

                for _ in range(self.n_critic):
                    noise = np.random.normal(0, 1, size=(batch_size, self.dim_latent))
                    batch_real = X_train[batch_index*batch_size:batch_index*batch_size + batch_size]

                    c_loss = self.critic.train_on_batch([batch_real, noise], [y_real, y_fake, y_dummy])

                # ----------------------
                #    Train Generator
                # ----------------------
                noise = np.random.normal(size=(batch_size, self.dim_latent))
                g_loss = self.generator.train_on_batch(noise, y_real)

            # Store loss of most recent batch from this epoch
            critic_losses.append(c_loss)
            gen_losses.append(g_loss)

            # if e == 1 or e % 5 == 0:
            #     plotGeneratedImages(e)
            #     saveModels(e)


if __name__ == '__main__':
    dim_latent = 100
    img_shape = (28, 28, 1)
    batch_size = 32

    wgan = WGANGP(dim_latent, img_shape, batch_size)
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], *img_shape)
    print(X_train.shape)
    wgan.train(X_train=X_train, epochs=2, batch_size=batch_size)
