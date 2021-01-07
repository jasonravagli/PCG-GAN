import os
import numpy as np

from config import cfg
from levels.utils.level_image_gen import LevelImageGen


def ascii_to_one_hot(level_token, tokens):
    """
    Converts the specified image level (a list of string containing ASCII tokens) to a one-hot 3D tensor
    :return:
    """
    height = level_token.shape[0]
    width = level_token.shape[1]
    level_one_hot = np.zeros((height, width, len(tokens)))
    for i in range(height):
        for j in range(width):
            token = level_token[i][j]
            if token != "\n":
                level_one_hot[i, j, tokens.index(token)] = 1
    return level_one_hot


def one_hot_to_ascii(level_one_hot, tokens):
    """
    Converts the specified image level (a 3D tensor made of one-hot encoded elements) to a a list of string containing ASCII tokens
    :return:
    """
    level_ascii = []
    height = level_one_hot.shape[0]
    width = level_one_hot.shape[1]
    for i in range(height):
        line = ""
        for j in range(width):
            line += tokens[level_one_hot[i, j, :].argmax()]
        if i < width - 1:
            line += "\n"
        level_ascii.append(line)
    return level_ascii


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

