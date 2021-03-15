import os

from config import cfg


class TokenSet:

    SPRITE_SIZE = (16, 16)

    def __init__(self):
        self.name = None
        self.token_hierarchy = None
        self.tokens = None
        self.token_sprites = None

    def get_path_token_sprite(self, token: str):
        return os.path.join(cfg.PATH.TOKENSETS, self.name, "sprites", self.tokens[token])

