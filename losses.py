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
    # return tf.reduce_mean(fake_score)
    return -tf.reduce_mean(fake_score)


def critic_wass_loss(real_score, fake_score):
    """
    Simplification of the Wasserstein loss for the critic model
    :param real_score:
    :param fake_score:
    :return:
    """
    # return tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)
    return -tf.reduce_mean(real_score) + tf.reduce_mean(fake_score)


def gradient_penalty_loss(batch_size, real_images, fake_images, critic):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    # diff = real_images - fake_images
    # interpolated = real_images + alpha * diff
    # diff = fake_images - real_images
    # interpolated = real_images + alpha * diff

    interpolated = alpha * real_images + ((1 - alpha) * fake_images)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the critic output for this interpolated image.
        pred = critic(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[3]))  # axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    # gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1) ** 2)
    return gp


def reconstruction_loss(reconstructed_image, real_image):
    """
    Calculate the reconstruction loss as the mean squared error between the reconstructed and the real image (see the SiGAN
    paper for further details)
    :param reconstructed_image:
    :param real_image:
    :return:
    """
    return tf.reduce_mean(tf.math.squared_difference(reconstructed_image, real_image))
