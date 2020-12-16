import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    """
    Implementation of the Wasserstein loss for WGAN as suggested here:
    https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
    Fake images labels (y_true values) must be -1, encouraging the critic to assign greater scores to fake images.
    Real images labels (y_true values) must be 1, encouraging the critic to assign smaller scores to real images.
    Concerning the generator, generated (fake) images are labeled as 1, encouraging the generator to fool the discriminator
    (fake images evaluated as real bring lower losses)
    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.math.reduce_mean(y_true*y_pred)
