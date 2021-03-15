import os
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QPushButton, QSizePolicy, QToolButton

from config import cfg
from gui.model.tilebox import TileBoxModel



class ButtonTile(QToolButton):

    ICON_SIZE = 40

    def __init__(self, tile_char: str, tilebox_model: TileBoxModel, tile_clicked_signal: pyqtSignal):
        super().__init__()

        self._tile_char = tile_char

        self.setMinimumSize(32, 32)
        self.setMaximumSize(64, 64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        icon_pixmap = QPixmap.fromImage(tilebox_model.get_tiles_images()[tile_char])
        scaled = icon_pixmap.scaled(self.ICON_SIZE, self.ICON_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon = QIcon(scaled)
        self.setIcon(icon)
        self.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))

        self._tile_clicked_signal = tile_clicked_signal
        self.clicked.connect(self.clicked_event)

    def clicked_event(self):
        # On click call the main signal defined inside the parent WidgetTilebox
        self._tile_clicked_signal.emit(self._tile_char)

    def set_highlight_border(self):
        self.setStyleSheet("border: 1px solid rgb(255, 255, 0);")

    def set_plain_border(self):
        self.setStyleSheet("")

