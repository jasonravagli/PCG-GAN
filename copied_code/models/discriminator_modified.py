import tensorflow as tf
from tensorflow_core.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Softmax


class DiscriminatorModified(tf.keras.Model):

    def __init__(self, opt):
        super(DiscriminatorModified, self).__init__()

        self.gp_weight = opt.lambda_grad

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
        real = inputs[0]
        fake = inputs[0]

        score_real = self._forward(real)

        if fake is not None:
            score_fake = self._forward(fake)
        else:
            score_fake = tf.zeros_like(score_real)

        self.add_loss(self._loss(real, fake, score_real, score_fake))

    def _forward(self, img):
        x = self.list_conv[0](img)
        x = self.list_batch_norm[0](x)
        x = self.list_activations[0](x)

        for i in range(1, len(self.list_conv)):
            x = self.list_conv[i](x)
            x = self.list_batch_norm[i](x)
            x = self.list_activations[i](x)

        return self.conv_out(x)

    def _loss(self, real_img, fake_img, real_score, fake_score):
        # Get the interpolated image
        alpha = tf.random.normal([1, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_img + ((1 - alpha) * fake_img)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the critic output for this interpolated image.
            pred = self._forward(interpolated)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return -tf.reduce_mean(real_score) + tf.reduce_mean(fake_score) + self.gp_weight * gp
