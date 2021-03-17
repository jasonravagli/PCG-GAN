import json
import numpy as np

from files import tokenset_files
from levels.level import Level
from utils.converters import ascii_to_one_hot_level


def load(path_file: str) -> Level:
    try:
        with open(path_file) as f:
            dict_level = json.load(f)

        level_size = dict_level["size"]
        level_grid = np.empty(level_size, dtype=np.object)
        unique_tokens = set()
        for row in range(level_size[0]):
            for col in range(level_size[1]):
                token = dict_level["level"][row][col]
                level_grid[row, col] = token
                unique_tokens.add(token)
        unique_tokens = list(unique_tokens)
        # Establishing a sorting is necessary to preserve correspondences between ASCII tokens and one-hot encode
        unique_tokens.sort()

        level = Level()
        level.name = dict_level["name"]
        level.tokenset = tokenset_files.load(dict_level["tokenset"])
        level.unique_tokens = unique_tokens
        level.level_size = level_size
        level.level_ascii = level_grid
        level.level_oh = ascii_to_one_hot_level(level.level_ascii, unique_tokens)

        return level
    except:
        return None


def save(level: Level, path_file: str):
    dict_level = {"name": level.name, "tokenset": level.tokenset.name, "size": level.level_size, "level": []}

    for row in range(level.level_ascii.shape[0]):
        row_string = [token for token in level.level_ascii[row]]
        dict_level["level"].append(row_string)

    with open(path_file, "w") as f:
        json.dump(dict_level, f)
