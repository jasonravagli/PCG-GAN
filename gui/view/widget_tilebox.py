from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QGridLayout

from gui.model.tilebox import TileBoxModel, TypeChange
from gui.view.button_tile import ButtonTile


class WidgetTilebox(QWidget):

    # Signal to notify the click on a button tile of the tilebox
    tile_clicked = pyqtSignal(str)

    def __init__(self, tilebox_model: TileBoxModel):
        super().__init__()

        self.button_tiles = {}
        self._tilebox_model = tilebox_model
        tilebox_model.observe(self.update_tilebox)

    def connect_to_tile_clicked(self, slot):
        self.tile_clicked.connect(slot)

    def update_tilebox(self, type_change: TypeChange):
        if type_change == TypeChange.ALL:
            # Whole tileset changed: update the tilebox to display the available tiles
            self.button_tiles = {}
            layout = QGridLayout()
            i = 0
            for tile_char in self._tilebox_model.get_tiles_images().keys():
                row = i // 3
                column = i % 3

                button_tile = ButtonTile(tile_char, self._tilebox_model, self.tile_clicked)
                # Highlight the tile selected by default
                if tile_char == self._tilebox_model.get_selected_tile():
                    button_tile.set_highlight_border()
                else:
                    button_tile.set_plain_border()

                layout.addWidget(button_tile, row, column)
                self.button_tiles[tile_char] = button_tile

                i += 1

            self.setLayout(layout)
        else:
            # Selected tile changed: highlight new selected tile
            for tile_char, button_tile in self.button_tiles.items():
                if tile_char == self._tilebox_model.get_selected_tile():
                    button_tile.set_highlight_border()
                else:
                    button_tile.set_plain_border()
