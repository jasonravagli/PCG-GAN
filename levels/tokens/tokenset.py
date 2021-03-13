import os

from config import cfg


class TokenSet:
    def __init__(self):
        self.name = None
        self.token_hierarchy = None
        self.tokens = None

    def get_path_token_sprite(self, token: str):
        return os.path.join(cfg.PATH.TOKENSETS, self.name, "sprites", self.tokens[token])

