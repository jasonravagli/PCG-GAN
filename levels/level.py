from PIL import Image

from config import cfg


class Level:
    def __init__(self):
        self.name = "untitled"
        self.tokenset = None
        self.level_ascii = None
        self.level_oh = None
        self.level_size = None
        self.unique_tokens = None  # List of unique tokens (ordered using char order) appearing in level

    def copy(self):
        level = Level()
        level.name = self.name
        level.tokenset = self.tokenset
        level.level_ascii = self.level_ascii.copy() if self.level_ascii is not None else None
        level.level_oh = self.level_ascii.copy() if self.level_ascii is not None else None
        level.level_size = self.level_size
        level.unique_tokens = self.unique_tokens.copy() if self.unique_tokens is not None else None

        return level

    def render(self):
        """
        Generate a PIL Image from the level
        """
        sprite_size = cfg.LEVEL.TILE_SIZE

        image_height = sprite_size[0]*self.level_size[0]
        image_width = sprite_size[1]*self.level_size[1]
        # Create an empty white image
        level_image = Image.new("RGBA", (image_width, image_height), 255)

        # Place the token sprites on the image
        for row in range(self.level_size[0]):
            for col in range(self.level_size[1]):
                token = self.level_ascii[row, col]
                # Rows and cols are swapped in PIL Image (first the column (width) then the row (height))
                level_image.paste(self.tokenset.token_sprites[token],
                                  (col*sprite_size[1], row*sprite_size[0],
                                   (col + 1)*sprite_size[1], (row + 1)*sprite_size[0]))

        return level_image
