import os
import numpy as np
import tensorflow as tf

from copied_code.train_single_scale import train_single_scale
from levels.tokens.mario import TOKEN_GROUPS, TOKEN_DOWNSAMPLING_HIERARCHY
from copied_code.level_utils import one_hot_to_ascii_level
from levels.utils.downsampling import downsample_image
from copied_code.models import init_models


def train(real, opt):
    generators = []
    noise_maps = []
    noise_amplitudes = []

    token_group = TOKEN_GROUPS
    token_hierarchy = TOKEN_DOWNSAMPLING_HIERARCHY

    scales = [[x, x] for x in opt.scales]
    opt.num_scales = len(scales)

    scaled_list = downsampling(scales, real, opt.token_list, token_hierarchy)

    reals = [*scaled_list, real]

    input_from_prev_scale = tf.zeros(reals[0].shape)

    stop_scale = len(reals)
    opt.stop_scale = stop_scale

    ######################################
    if not os.path.isdir("imgs"):
        os.mkdir("imgs")

    for i in range(len(scaled_list)):
        # Convert to ascii level
        level = one_hot_to_ascii_level(scaled_list[i], opt.token_list)

        # Render and save level image
        img = opt.ImgGen.render(level)
        img.save(f"imgs/scaled-{i}.png")
    ######################################

    # Training Loop
    for current_scale in range(0, stop_scale):
        # opt.outf = "%s/%d" % (opt.out_, current_scale)
        # try:
        #     os.makedirs(opt.outf)
        # except OSError:
        #     pass

        # # If we are seeding, we need to adjust the number of channels
        # if current_scale < (opt.token_insert + 1):  # (stop_scale - 1):
        #     opt.nc_current = len(token_group)

        # Initialize models
        D, G = init_models(opt, reals[current_scale].shape)

        # Actually train the current scale
        z_opt, input_from_prev_scale, G = train_single_scale(D, G, reals, generators, noise_maps,
                                                             input_from_prev_scale, noise_amplitudes, opt)

        generators.append(G)
        noise_maps.append(z_opt)
        noise_amplitudes.append(opt.noise_amp)

    return generators, noise_maps, reals, noise_amplitudes


def downsampling(scales, real, token_list, token_hierarchy):
    scaled_images = []
    for scale in scales:
        scale_v = scale[0]
        scale_h = scale[1]

        scaled_image = downsample_image(real[0], (int(real.shape[1]*scale_v), int(real.shape[2]*scale_h)),
                                        token_list, token_hierarchy)

        scaled_image = np.expand_dims(scaled_image, axis=0)

        scaled_images.append(scaled_image)

    scaled_images.reverse()
    return scaled_images
