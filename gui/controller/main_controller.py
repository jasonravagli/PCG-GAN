import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog

from config import cfg
from files import level_files, tokenset_files
from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.view.main_window import MainWindow
from utils import converters


class MainController:
    def __init__(self, main_window: MainWindow, tilebox_model: TileBoxModel, level_model: LevelModel):

        self._main_window = main_window
        self._level_model = level_model
        self._tilebox_model = tilebox_model

        # Connect controller to GUI signals
        main_window.connect_to_button_clear(self.clear_level)
        main_window.connect_to_button_load_level(self.load_level)
        main_window.connect_to_button_save(self.save_level)
        main_window.connect_to_combo_tileset(self.load_tileset)
        main_window.connect_to_edit_level_name(self.set_level_name)
        main_window.connect_to_grid_size_changed(self.set_level_grid_size)
        main_window.widget_tilebox.connect_to_tile_clicked(self.select_tile_from_tilebox)
        main_window.level_grid.connect_to_tile_clicked(self.apply_selected_tile_to_level)

    def apply_selected_tile_to_level(self, row, column):
        self._level_model.set_grid_tile(row, column, self._tilebox_model.get_selected_tile())

    def clear_level(self):
        default_tile = list(self._tilebox_model.get_tiles_images().keys())[0]
        grid_tiles = np.full(self._level_model.get_level_size(), default_tile)
        self._level_model.set_grid_tiles(grid_tiles)
        self._main_window.show_message_on_statusbar("Level Cleared")

    def load_level(self):
        # Show a dialog to select the level file
        start_directory = os.path.join(cfg.PATH.LEVELS_DIR, self._level_model.get_name() + ".json")
        file_path = QFileDialog.getOpenFileName(self._main_window, "Load level file", directory=start_directory,
                                                filter="Level File (*.json)")[0]
        if file_path:
            level = level_files.load(file_path)

            # Set the TilesetModel fields from the Tokenset
            tilebox_model = converters.tokenset_to_tilebox_model(level.tokenset)
            self._tilebox_model.set_from_tilebox_model(tilebox_model)

            # Set the LevelModel with the loaded Level
            level_model = converters.level_to_level_model(level)
            self._level_model.set_from_level_model(level_model)
            self._main_window.show_message_on_statusbar("Level Loaded")

    def load_tileset(self, tokenset_name: str):
        tokenset = tokenset_files.read(tokenset_name)

        # Set the TilesetModel fields from the Tokenset
        tilebox_model = converters.tokenset_to_tilebox_model(tokenset)
        self._tilebox_model.set_from_tilebox_model(tilebox_model)

        # Clear the level after changing the tilebox
        self.clear_level()

    def save_level(self):
        # Show a dialog to select the destination file
        start_directory = cfg.PATH.LEVELS_DIR
        file_path = QFileDialog.getSaveFileName(self._main_window, "Save level file", directory=start_directory,
                                                filter="Level File (*.json)")[0]
        if file_path:
            level = converters.level_model_to_level(self._level_model, self._tilebox_model.get_tokenset_name())
            level_files.save(level, file_path)
            self._main_window.show_message_on_statusbar("Level Saved")

    def select_tile_from_tilebox(self, tile_char: str):
        self._tilebox_model.set_selected_tile(tile_char)

    def set_level_grid_size(self, rows: int, columns: int):
        self._level_model.set_level_size(rows, columns)

    def set_level_name(self, level_name: str):
        self._level_model.set_name(level_name)
