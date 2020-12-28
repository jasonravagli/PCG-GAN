import os
import matplotlib.pyplot as plt
import numpy as np


def plot_losses(path_file, critic_loss, gen_loss):
    iterations = range(len(critic_loss))

    fig = plt.figure()
    plt.plot(iterations, critic_loss, color="r", label="Critic Loss")
    plt.plot(iterations, gen_loss, color="b", label="Gen. Loss")
    plt.legend()
    fig.savefig(path_file)


def get_all_patches_from_img(img, patch_height, patch_width):
    img_height, img_width, img_channels = img.shape

    # Current patch vertical pixels boundaries
    top_y = 0
    bottom_y = patch_height

    n_patches = (img_width - patch_width)*(img_height - patch_height)
    patches = np.empty((n_patches, patch_height, patch_width, img_channels))
    patch_index = 0
    while bottom_y < img_height:
        # Current patch horizontal pixels boundaries
        left_x = 0
        right_x = patch_width
        while right_x < img_width:
            patches[patch_index] = img[left_x:right_x, top_y:bottom_y, :]

            left_x += 1
            right_x += 1
            patch_index += 1

        top_y += 1
        bottom_y += 1

    return patches
