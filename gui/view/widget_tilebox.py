from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout

from gui.model.tilebox import TileBoxModel, TypeChange
from gui.utils import clear_layout
from gui.view.button_tile import ButtonTile


class WidgetTilebox(QWidget):

    N_COLUMNS = 3

    # Signal to notify the click on a button tile of the tilebox
    tile_clicked = pyqtSignal(str)

    def __init__(self, tilebox_model: TileBoxModel):
        super().__init__()

        # Create columns where to place tilebox tiles
        layout = QHBoxLayout()
        self._tilebox_columns = []
        for i in range(self.N_COLUMNS):
            col_layout = QVBoxLayout()
            layout.addLayout(col_layout)
            self._tilebox_columns.append(col_layout)
        self.setLayout(layout)

        self.button_tiles = {}
        self._tilebox_model = tilebox_model
        tilebox_model.observe(self.update_tilebox)

    def connect_to_tile_clicked(self, slot):
        self.tile_clicked.connect(slot)

    def update_tilebox(self, type_change: TypeChange):
        if type_change == TypeChange.ALL:
            # Clear the tilebox
            for i in range(self.N_COLUMNS):
                clear_layout(self._tilebox_columns[i])

            # Whole tileset changed: update the tilebox to display the available tiles
            self.button_tiles = {}
            column = 0
            for tile_char in self._tilebox_model.get_tiles_images().keys():
                button_tile = ButtonTile(tile_char, self._tilebox_model, self.tile_clicked)

                # Highlight the tile selected by default
                if tile_char == self._tilebox_model.get_selected_tile():
                    button_tile.set_highlight_border()
                else:
                    button_tile.set_plain_border()

                self._tilebox_columns[column].addWidget(button_tile)
                self.button_tiles[tile_char] = button_tile

                column = (column + 1) % self.N_COLUMNS
        else:
            # Selected tile changed: highlight new selected tile
            for tile_char, button_tile in self.button_tiles.items():
                if tile_char == self._tilebox_model.get_selected_tile():
                    button_tile.set_highlight_border()
                else:
                    button_tile.set_plain_border()
