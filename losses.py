import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    """
    Implementation of the Wasserstein loss for WGAN as suggested here:
    https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
    Fake images labels (in y_true) must be -1, encouraging the critic to assign greater scores to fake images.
    Real images labels (in y_true) must be 1, encouraging the critic to assign smaller scores to real images.
    Concerning the generator, generated (fake) images are labeled as 1, encouraging the generator to fool the discriminator
    (fake images evaluated as real bring lower losses)

    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.math.reduce_mean(y_true*y_pred)


def generator_wass_loss(fake_score):
    """
    Simplification of the Wasserstein loss for the generator model
    :param fake_score:
    :return:
    """
    return tf.reduce_mean(fake_score)


def critic_wass_loss(real_score, fake_score):
    """
    Simplification of the Wasserstein loss for the critic model
    :param real_score:
    :param fake_score:
    :return:
    """
    return tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)
