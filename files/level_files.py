import json
import numpy as np

from files import tokenset_files
from levels.level import Level


def load(path_file: str) -> Level:
    with open(path_file) as f:
        dict_level = json.load(f)

    level_size = dict_level["size"]
    level_grid = np.empty(level_size, dtype=np.object)
    for row in range(level_size[0]):
        for col in range(level_size[1]):
            level_grid[row, col] = dict_level["level"][row][col]

    level = Level()
    level.name = dict_level["name"]
    level.level_size = level_size
    level.level_grid = level_grid
    level.tokenset = tokenset_files.read(dict_level["tokenset"])

    return level


def save(level: Level, path_file: str):
    dict_level = {"name": level.name, "tokenset": level.tokenset.name, "size": level.level_size, "level": []}

    for row in range(level.level_grid.shape[0]):
        row_string = [token for token in level.level_grid[row]]
        dict_level["level"].append(row_string)

    with open(path_file, "w") as f:
        json.dump(dict_level, f)
