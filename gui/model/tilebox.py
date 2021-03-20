from enum import Enum

from PyQt5.QtCore import pyqtSignal, QObject


class TypeChange(Enum):
    ALL = 0
    SELECTED_TILE = 1


class TileBoxModel(QObject):
    value_changed = pyqtSignal(TypeChange)

    def __init__(self):
        super().__init__()

        self._loaded = False
        self._tokenset_name = ""
        self._selected_tile = ""
        self._tiles_images = None
        self._tiles_np = None

    def get_loaded(self) -> bool:
        return self._loaded

    def get_tokenset_name(self) -> str:
        return self._tokenset_name

    def get_selected_tile(self) -> str:
        return self._selected_tile

    def get_tiles_images(self) -> dict:
        return self._tiles_images.copy() if self._loaded is not None else None

    def get_tiles_np(self) -> dict:
        return self._tiles_np.copy() if self._loaded is not None else None

    def set_from_tilebox_model(self, tilebox_model):
        self._loaded = True
        self._tokenset_name = tilebox_model.get_tokenset_name()
        self._tiles_images = tilebox_model.get_tiles_images()
        self._tiles_np = tilebox_model.get_tiles_np()
        self._selected_tile = tilebox_model.get_selected_tile()
        self.notify(TypeChange.ALL)

    def set_loaded(self, loaded: bool):
        self._loaded = loaded

    def set_selected_tile(self, tile_char: str):
        self._selected_tile = tile_char
        self.notify(TypeChange.SELECTED_TILE)

    def set_tiles_images(self, tiles: dict):
        self._tiles_images = tiles.copy()
        self.notify(TypeChange.ALL)

    def set_tiles_np(self, tiles: dict):
        self._tiles_np = tiles.copy()
        self.notify(TypeChange.ALL)

    def set_tokenset_name(self, name: str):
        self._tokenset_name = name
        self.notify(TypeChange.ALL)

    # Observable methods, type: TypeChange
    def observe(self, slot):
        self.value_changed.connect(slot)

    def notify(self, type_change: TypeChange):
        self.value_changed.emit(type_change)
