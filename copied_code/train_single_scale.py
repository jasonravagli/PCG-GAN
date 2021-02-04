import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.layers import ZeroPadding2D
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from copied_code.level_utils import one_hot_to_ascii_level

matplotlib.use('Agg')

from copied_code.draw_concat import draw_concat
from levels.tokens.mario import TOKEN_GROUPS
from levels.utils.conversion import group_to_token
from losses import gradient_penalty_loss
from utils import generate_noise


def update_noise_amplitude(z_prev, real, opt):
    """ Update the amplitude of the noise for the current scale according to the previous noise map. """
    RMSE = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(real, z_prev)))
    return opt.noise_update * RMSE


def train_single_scale(D, G, reals, generators, noise_maps, input_from_prev_scale, noise_amplitudes, opt):
    """ Train one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
    original level, generators and noise_maps contain information from previous scales and will receive information in
    this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
    the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """
    current_scale = len(generators)
    real = reals[current_scale]

    token_group = TOKEN_GROUPS

    nzx = real.shape[1]  # Noise size x
    nzy = real.shape[2]  # Noise size y

    padsize = int(1 * opt.num_layer)  # As kernel size is always 3 currently, padsize goes up by one per layer

    pad_noise = ZeroPadding2D(padsize)
    pad_image = ZeroPadding2D(padsize)

    # setup optimizer ---> MISSING LR SCHEDULERS!
    optimizerD = Adam(lr=opt.lr_d, beta_1=opt.beta1)
    optimizerG = Adam(lr=opt.lr_g, beta_1=opt.beta1)
    ################ MISSING SCHEDULERS #####################

    if current_scale == 0:  # Generate new noise
        z_opt = generate_noise([1, nzx, nzy, opt.nc_current])
        z_opt = pad_noise(z_opt)
    else:  # Add noise to previous output
        z_opt = tf.zeros([1, nzx, nzy, opt.nc_current])
        z_opt = pad_noise(z_opt)

    # Manual fix
    G.build([z_opt.shape]*2)
    D.build(z_opt.shape)
    G.summary()
    D.summary()

    ################
    list_c_wass_loss = []
    list_c_gp_loss = []
    list_g_adv_loss = []
    list_g_rec_loss = []
    ################

    print(f"Training at scale {current_scale}")
    for epoch in tqdm(range(opt.niter)):
        step = current_scale * opt.niter + epoch
        noise_ = generate_noise([1, nzx, nzy, opt.nc_current])
        noise_ = pad_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):

            # train with fake
            if (j == 0) & (epoch == 0):
                if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                    prev = tf.zeros((1, nzx, nzy, opt.nc_current))
                    input_from_prev_scale = prev
                    prev = pad_image(prev)
                    z_prev = tf.zeros((1, nzx, nzy, opt.nc_current))
                    z_prev = pad_noise(z_prev)
                    opt.noise_amp = 1
                else:  # First step in NOT the lowest scale
                    # We need to adapt our inputs from the previous scale and add noise to it
                    prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                       "rand", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        prev = group_to_token(prev, opt.token_list, token_group)

                    prev = tf.image.resize(prev, (real.shape[1], real.shape[2]),
                                          method=tf.image.ResizeMethod.BILINEAR)
                    prev = pad_image(prev)
                    z_prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                         "rec", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        z_prev = group_to_token(z_prev, opt.token_list, token_group)

                    z_prev = tf.image.resize(z_prev, (real.shape[1], real.shape[2]),
                                           method=tf.image.ResizeMethod.BILINEAR)
                    opt.noise_amp = update_noise_amplitude(z_prev, real, opt)
                    z_prev = pad_image(z_prev)
            else:  # Any other step
                prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                   "rand", pad_noise, pad_image, opt)

                # For the seeding experiment, we need to transform from token_groups to the actual token
                if current_scale == (opt.token_insert + 1):
                    prev = group_to_token(prev, opt.token_list, token_group)

                prev = tf.image.resize(prev, (real.shape[1], real.shape[2]),
                                         method=tf.image.ResizeMethod.BILINEAR)
                prev = pad_image(prev)

            # After creating our correct noise input, we feed it to the generator:
            noise = opt.noise_amp * noise_ + prev

            with tf.GradientTape() as tape:
                fake = G([noise, prev], training=True)
                output = D(real, training=True)
                errD_real = -tf.reduce_mean(output)

                output = D(fake, training=True)
                errD_fake = tf.reduce_mean(output)

                # Calculate the gradient penalty
                gp = gradient_penalty_loss(batch_size=1, real_images=real, fake_images=fake,
                                           critic=D)
                # Add the gradient penalty to the original discriminator loss
                loss = errD_real + errD_fake + gp * opt.lambda_grad

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(loss, D.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            optimizerD.apply_gradients(
                zip(d_gradient, D.trainable_variables)
            )

            list_c_wass_loss.append(errD_real + errD_fake)
            list_c_gp_loss.append(gp * opt.lambda_grad)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):

            with tf.GradientTape() as tape:
                fake = G([noise, prev], training=True)
                output = D(fake, training=True)
                errG = -tf.reduce_mean(output)

                Z_opt = opt.noise_amp * z_opt + z_prev
                G_rec = G([Z_opt, z_prev], training=True)
                rec_loss = opt.alpha * tf.reduce_mean(tf.math.squared_difference(G_rec, real))

                loss = errG + rec_loss
            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(loss, G.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            optimizerG.apply_gradients(
                zip(gen_gradient, G.trainable_variables)
            )

            list_g_adv_loss.append(errG)
            list_g_rec_loss.append(rec_loss)

        # Learning Rate scheduler step
        # schedulerD.step()
        # schedulerG.step()

        if epoch % 1000 == 0 or epoch == opt.niter - 1:
            path_folder = os.path.join("imgs", str(current_scale))
            if not os.path.isdir(path_folder):
                os.mkdir(path_folder)

            for i in range(3):
                prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                   "rand", pad_noise, pad_image, opt)
                if current_scale > 0:
                    prev = tf.image.resize(prev, (real.shape[1], real.shape[2]), method=tf.image.ResizeMethod.BILINEAR)
                prev = pad_image(prev)
                noise = opt.noise_amp * noise_ + prev
                fake = G([noise, prev]).numpy()

                ascii_level = one_hot_to_ascii_level(fake, opt.token_list)
                image_rgb = opt.ImgGen.render(ascii_level)
                image_rgb.save(os.path.join(path_folder, f"epoch-{epoch}-{i}.png"), format="png")

    #######################

    iterations = range(len(list_c_wass_loss))

    list_c_wass_loss = np.clip(np.array(list_c_wass_loss), -10, 10)
    list_c_gp_loss = np.clip(np.array(list_c_gp_loss), -10, 10)
    list_g_adv_loss = np.clip(np.array(list_g_adv_loss), -10, 10)
    list_g_rec_loss = np.clip(np.array(list_g_rec_loss), -10, 10)

    fig = plt.figure()
    plt.plot(iterations, list_c_wass_loss, color="r", label="C Wass Loss")
    plt.plot(iterations, list_c_gp_loss, color="b", label="C GP Loss")
    plt.plot(iterations, list_g_adv_loss, color="g", label="G Adv Loss")
    plt.plot(iterations, list_g_rec_loss, color="m", label="G Rec Loss")
    plt.legend()
    fig.savefig(f"imgs/losses-{current_scale}.png")

    #########################

    return z_opt, input_from_prev_scale, G

