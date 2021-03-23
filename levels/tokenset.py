import os

from config import cfg


class TokenSet:

    def __init__(self):
        self.name = None
        self.token_hierarchy = None
        self.tokens = None
        self.token_sprites_preview = None  # Sprites with transparency to be displayed in a toolbox or in a preview
        self.token_sprites = None  # Sprites with background to be used in a rendered level

    def get_path_token_sprite(self, token: str):
        return os.path.join(cfg.PATH.TOKENSETS, self.name, "sprites", self.tokens[token])

