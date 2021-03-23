import numpy as np

from gui.model.observable import Observable


class LevelModel(Observable):
    def __init__(self):
        super().__init__()

        self._name = "untitled"
        self._level_size = (16, 64)
        self._grid_tiles = np.full(self._level_size, '-', dtype=np.object)
        self._highlighted_tile = (-1, -1)

    def get_grid_tile(self, row, column) -> str:
        return self._grid_tiles[row, column]

    def get_grid_tiles(self) -> np.ndarray:
        return self._grid_tiles.copy()

    def get_highlighted_tile(self) -> tuple:
        return self._highlighted_tile

    def get_level_size(self):
        return self._level_size

    def get_name(self):
        return self._name

    def set_from_level_model(self, level_model):
        self._name = level_model.get_name()
        self._level_size = level_model.get_level_size()
        self._grid_tiles = level_model.get_grid_tiles()
        self.notify()

    def set_grid_tile(self, row: int, column: int, value: str):
        self._grid_tiles[row, column] = value
        self.notify()

    def set_grid_tiles(self, grid_tiles: np.ndarray):
        self._grid_tiles = np.array(grid_tiles, dtype=np.object, copy=True)
        self.notify()

    def set_level_size(self, rows, columns):
        self._level_size = (rows, columns)
        self._grid_tiles = np.full(self._level_size, '-', dtype=np.object)
        self.notify()

    def set_name(self, name: str):
        self._name = name
        self.notify()

    def set_highlighted_tile(self, tile: tuple):
        self._highlighted_tile = tile
        self.notify()
