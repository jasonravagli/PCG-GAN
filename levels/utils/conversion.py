import os
import numpy as np
import tensorflow as tf

from config import cfg
from levels.tokens.mario import TOKEN_GROUPS
from levels.utils.level_image_gen import LevelImageGen


def ascii_to_rgb(level_token):
    """
    Converts the specified image level (a a list of string containing ASCII tokens) to an RGB image using the icons associated to each token
    :param img_token:
    :param dict_token:
    :param tile_dim:
    :return:
    """
    level_img_gen = LevelImageGen(os.path.join(cfg.PATH.RESOURCES, "levels", "mario", "sprites"))
    return level_img_gen.render(level_token)


def ascii_to_one_hot_level(level, tokens):
    """ Converts an ascii level to a full token level tensor. """
    oh_level = np.zeros((len(level), len(level[-1]), len(tokens)), dtype=np.float32)
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in tokens and token != "\n":
                oh_level[i, j, tokens.index(token)] = 1
    return oh_level


def one_hot_to_ascii_level(level, tokens):
    """ Converts a full token level tensor to an ascii level. """
    ascii_level = []
    for i in range(level.shape[0]):
        line = ""
        for j in range(level.shape[1]):
            line += tokens[tf.math.argmax(level[i, j, :])]
        if i < level.shape[0] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level


# Miscellaneous functions to deal with ascii-token-based levels.

def group_to_token(tensor, tokens, token_groups=TOKEN_GROUPS):
    """ Converts a token group level tensor back to a full token level tensor. """
    new_tensor = np.zeros(tensor.shape[0], len(tokens), *tensor.shape[2:])
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, i] = tensor[:, group_idx]
                break
    return new_tensor


def token_to_group(tensor, tokens, token_groups=TOKEN_GROUPS):
    """ Converts a full token tensor to a token group tensor. """
    new_tensor = np.zeros(tensor.shape[0], len(token_groups), *tensor.shape[2:])
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, group_idx] += tensor[:, i]
                break
    return new_tensor

