import numpy as np
from PyQt5.QtGui import QImage

from files.tokenset_files import read
from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.utils import load_image_to_numpy
from levels.level import Level
from levels.tokens.tokenset import TokenSet


def level_model_to_level(level_model: LevelModel, tokenset_name: str) -> Level:
    level = Level()
    level.name = level_model.get_name()
    level.tokenset = read(tokenset_name)
    level.level_size = level_model.get_level_size()
    level.level_grid = level_model.get_grid_tiles()

    return level


def level_to_level_model(level: Level) -> LevelModel:
    level_model = LevelModel()
    level_model.set_name(level.name)
    rows, columns = level.level_grid.shape[:2]
    level_model.set_level_size(rows, columns)
    level_model.set_grid_tiles(np.copy(level.level_grid))

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
