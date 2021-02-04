import tensorflow as tf
from tensorflow_core.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Softmax


class Discriminator(tf.keras.Model):

    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.list_conv = []
        self.list_batch_norm = []
        self.list_activations = []
        for i in range(opt.num_layer - 1):
            self.list_conv.append(Conv2D(filters=opt.nfc, kernel_size=(opt.ker_size, opt.ker_size), strides=(1, 1),
                                         padding="valid", activation=None,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)))
            self.list_batch_norm.append(BatchNormalization(
                gamma_initializer=tf.keras.initializers.RandomNormal(mean=1., stddev=0.02)))
            self.list_activations.append(LeakyReLU(0.2))

        self.conv_out = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid")

    def call(self, inputs, **kwargs):
        img = inputs

        x = self.list_conv[0](img)
        x = self.list_batch_norm[0](x)
        x = self.list_activations[0](x)

        for i in range(1, len(self.list_conv)):
            x = self.list_conv[i](x)
            x = self.list_batch_norm[i](x)
            x = self.list_activations[i](x)

        return self.conv_out(x)
