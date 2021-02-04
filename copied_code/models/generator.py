import tensorflow as tf
from tensorflow_core.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Softmax


class Generator(tf.keras.Model):

    def __init__(self, opt):
        super(Generator, self).__init__()

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

        self.conv_last = Conv2D(filters=opt.nc_current, kernel_size=(opt.ker_size, opt.ker_size), strides=(1, 1),
                                padding="valid", activation=None)
        self.softmax = Softmax()

    def call(self, inputs, **kwargs):
        noise = inputs[0]
        prev = inputs[1]

        x = self.list_conv[0](noise)
        x = self.list_batch_norm[0](x)
        x = self.list_activations[0](x)

        for i in range(1, len(self.list_conv)):
            x = self.list_conv[i](x)
            x = self.list_batch_norm[i](x)
            x = self.list_activations[i](x)

        x = self.conv_last(x)

        x = self.softmax(x)

        ind = int((prev.shape[1] - x.shape[1]) / 2)
        prev = prev[:, ind:(prev.shape[1] - ind), ind:(prev.shape[2] - ind), :]

        return x + prev
