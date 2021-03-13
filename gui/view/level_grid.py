import math
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy
from qtpy import QtGui

from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.utils import level_model_to_qimage


class LevelGrid(QLabel):

    # Signal to notify the click on a tile of the level grid
    tile_clicked = pyqtSignal(int, int)

    def __init__(self, level_model: LevelModel, tilebox_model: TileBoxModel):
        super().__init__()

        self._tilebox_model = tilebox_model

        # Register as observer of the LevelModel to update the displayed grid on state changed
        self._level_model = level_model
        level_model.observe(self.update_grid)

        # Coordinates of the QPixmap (the grid) upper-left corner inside the QLabel.
        # They are required to handle mouse events on tiles
        self.x_pixmap = 0
        self.y_pixmap = 0
        # Keep track of the last tile drawn to handle continuous drawing through mouse dragging
        self._last_tile_drawn_row = -1
        self._last_tile_drawn_col = -1
        # Flag that indicates whether we are drawing on the grid (the mouse is pressed)
        self._drawing = False

        # Display current grid
        self.setObjectName("level_grid")
        self.setStyleSheet("#level_grid { border:1px solid rgb(255, 255, 255); }")

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)  # To allow the QLabel to shrink also with a pixmap attached

        # To enable the mouseMoveEvent tracking
        QWidget.setMouseTracking(self, True)

    def connect_to_tile_clicked(self, slot):
        self.tile_clicked.connect(slot)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        x_pos = ev.pos().x()
        y_pos = ev.pos().y()

        # Check if the mouse position is inside the effective level grid image
        x_grid = x_pos - self.x_pixmap
        y_grid = y_pos - self.y_pixmap
        if 0 <= x_grid < self.pixmap().width() and 0 <= y_grid < self.pixmap().height():
            self.setCursor(Qt.CrossCursor)

            # Check if we are drawing (the mouse is pressed and dragged)
            if self._drawing:
                # Converts the widget coordinates into grid coordinates
                grid_size = self._level_model.get_level_size()
                row = math.floor(y_grid * grid_size[0] / self.pixmap().height())
                col = math.floor(x_grid * grid_size[1] / self.pixmap().width())

                # Check if the event was already handled for this tile (the mouse is moved over the same tile)
                if row != self._last_tile_drawn_row or col != self._last_tile_drawn_col:
                    self.tile_clicked.emit(row, col)
                    self._last_tile_drawn_row = row
                    self._last_tile_drawn_col = col
        else:
            self.unsetCursor()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        x_click = ev.pos().x()
        y_click = ev.pos().y()

        # Converts the widget coordinates into grid coordinates
        x_grid = x_click - self.x_pixmap
        y_grid = y_click - self.y_pixmap
        if 0 <= x_grid < self.pixmap().width() and 0 <= y_grid < self.pixmap().height():
            # Converts the widget coordinates into grid coordinates
            grid_size = self._level_model.get_level_size()
            row = math.floor(y_grid * grid_size[0] / self.pixmap().height())
            col = math.floor(x_grid * grid_size[1] / self.pixmap().width())

            self.tile_clicked.emit(row, col)
            self._last_tile_drawn_row = row
            self._last_tile_drawn_col = col

            # Start continuous drawing
            self._drawing = True

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        # The mouse is released: stop drawing
        self._drawing = False
        self._last_tile_drawn_row = -1
        self._last_tile_drawn_col = -1

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.x_pixmap = (self.width() - self.pixmap().width())//2
        self.y_pixmap = (self.height() - self.pixmap().height())//2
        self.update_grid()

    def update_grid(self):
        qimage = level_model_to_qimage(self._level_model, self._tilebox_model)
        qpixmap = QPixmap.fromImage(qimage)
        # Scale the created QPixmap to fit the widget
        self.setPixmap(qpixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.FastTransformation))

        # Update the QPixmap coordinates
        self.x_pixmap = (self.width() - self.pixmap().width())//2
        self.y_pixmap = (self.height() - self.pixmap().height())//2
