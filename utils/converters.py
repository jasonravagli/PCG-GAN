import numpy as np
from PyQt5.QtGui import QImage

from files.tokenset_files import load
from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.utils import load_image_to_numpy
from levels.level import Level
from levels.tokenset import TokenSet


def ascii_to_one_hot_level(level: np.ndarray, unique_tokens: list):
    """ Converts an ascii level to a full token level tensor. """
    oh_level = np.zeros((len(level), len(level[-1]), len(unique_tokens)), dtype=np.float32)
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in unique_tokens and token != "\n":
                oh_level[i, j, unique_tokens.index(token)] = 1
    return oh_level


def one_hot_to_ascii_level(level: np.ndarray, unique_tokens: list):
    """ Converts a full token level tensor to an ascii level. """
    ascii_level = []
    for i in range(level.shape[0]):
        line = []
        for j in range(level.shape[1]):
            line.append(unique_tokens[np.argmax(level[i, j, :])])
        ascii_level.append(line)
    return np.array(ascii_level, dtype=np.object)


def level_model_to_level(level_model: LevelModel, tokenset_name: str) -> Level:
    level = Level()
    level.name = level_model.get_name()
    level.tokenset = load(tokenset_name)
    level.level_size = level_model.get_level_size()
    level.level_ascii = level_model.get_grid_tiles()

    return level


def level_to_level_model(level: Level) -> LevelModel:
    level_model = LevelModel()
    level_model.set_name(level.name)
    rows, columns = level.level_ascii.shape[:2]
    level_model.set_level_size(rows, columns)
    level_model.set_grid_tiles(np.copy(level.level_ascii))

    return level_model


def tokenset_to_tilebox_model(tokenset: TokenSet) -> TileBoxModel:
    tilebox_model = TileBoxModel()
    tilebox_model.set_tokenset_name(tokenset.name)
    tilebox_model.set_selected_tile("-")

    # Make the TilesetModel tiles a dict with the char identifiers as keys and the corresponding tile images as values
    tiles_images = {}
    tiles_np = {}
    for char_tile in tokenset.tokens.keys():
        tiles_images[char_tile] = QImage(tokenset.get_path_token_sprite(char_tile))
        tiles_np[char_tile] = load_image_to_numpy(tokenset.get_path_token_sprite(char_tile))
    tilebox_model.set_tiles_images(tiles_images)
    tilebox_model.set_tiles_np(tiles_np)

    return tilebox_model
