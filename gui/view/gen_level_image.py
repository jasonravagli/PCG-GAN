from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


class GeneratedLevelImage(QLabel):

    def __init__(self):
        super().__init__()
        self.level_pixmap = None
        self.void_pixmap = QPixmap()

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)  # To allow the QLabel to shrink also with a pixmap attached

        self.setObjectName("gen_level_image")
        self.setStyleSheet("#gen_level_image { border:1px solid rgb(255, 255, 255); }")

    def clear_pixmap(self):
        self.level_pixmap = None
        self.setPixmap(self.void_pixmap)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if self.level_pixmap is not None:
            self.setPixmap(self.level_pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def set_level_pixmap(self, level_pixmap: QPixmap):
        self.level_pixmap = level_pixmap
        self.setPixmap(self.level_pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.FastTransformation))
