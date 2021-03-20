import numpy as np
import PIL.Image
from PyQt5.QtGui import QImage

from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel


def clear_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clear_layout(item.layout())


def load_image_to_numpy(img_path: str):
    img_size = (16, 16)

    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    img.thumbnail(img_size, PIL.Image.ANTIALIAS)
    if img.width != img_size[1] or img.height != img_size[0]:
        wrap_image = PIL.Image.new("RGB", img_size, (0, 0, 0))
        wrap_image.paste(img, ((img_size[1] - img.width) // 2, img_size[0] - img.height))
        img = wrap_image
    return np.array(img)


def level_model_to_qimage(level_model: LevelModel, tilebox_model: TileBoxModel):
    img_tile_size = (16, 16)
    n_channels = 3

    available_tiles = tilebox_model.get_tiles_np()
    grid_tiles = level_model.get_grid_tiles()
    rows, columns = level_model.get_level_size()

    # Create empty image matrix
    img_height = img_tile_size[0] * rows
    img_width = img_tile_size[1] * columns
    np_img = np.zeros((img_height, img_width, n_channels), np.uint8)

    for row in range(rows):
        for col in range(columns):
            tile_char = grid_tiles[row, col]
            tile_np = available_tiles[tile_char]

            np_img[row * img_tile_size[0]:(row + 1) * img_tile_size[0], col * img_tile_size[1]:(col + 1) * img_tile_size[1], :] = tile_np

    return QImage(np_img, img_width, img_height, QImage.Format_RGB888)
