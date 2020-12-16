import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, Input, LeakyReLU, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam


class WGAN():
    """
    Reference: https://github.com/timsainb/tensorflow2-generative-models
    """

    def __init__(self, dim_latent, img_shape, **kwargs):
        super(WGAN, self).__init__()
        self.__dict__.update(kwargs)

        self.dim_latent = dim_latent
        self.img_shape = img_shape

        self.generator = self.build_generator()
        self.critic = self.build_critic()

    def build_generator(self):
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

        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, beta_1=0.5), metrics=['accuracy'])

        model.summary()

        return model

    def build_critic(self):
        model = Sequential()

        # ------ Conv Block 1 ------
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        # model.add(Dropout(0.25))
        # ------ Conv Block 2 ------
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # ------ Conv Block 3 ------
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # ------ Conv Block 4 ------
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="linear"))

        model.compile(loss='binary_crossentropy', optimizer=RMSprop(0.0005), metrics=['accuracy'])

        model.summary()

        return model

    def build_wgan(self):
        # Freeze the critic when building the GAN model
        self.critic.trainable = False

        # Random noise vector z
        z = Input(shape=(self.dim_latent, ))

        # Image label
        label = Input(shape=(1, ))

        # Generated image for that label
        img = self.generator([z, label])

        classification = self.critic([img, label])

        # Combined Generator -> Discriminator model
        # G([z, lablel]) = x*
        # D(x*) = classification
        model = Model([z, label], classification)

        model.compile(loss="binary_crossentropy", optimizer=RMSprop(0.0005), metrics=['accuracy'])

        return model
