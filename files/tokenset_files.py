from collections import OrderedDict

from PIL import Image
import json
import os

from config import cfg
from levels.tokenset import TokenSet


def get_all():
    return [f for f in os.listdir(cfg.PATH.TOKENSETS) if os.path.isdir(os.path.join(cfg.PATH.TOKENSETS, f))]


def load(tokenset_name: str) -> TokenSet:
    with open(os.path.join(cfg.PATH.TOKENSETS, tokenset_name, tokenset_name + ".json")) as f:
        json_data = json.load(f)

    tokenset = TokenSet()
    tokenset.name = json_data["name"]
    tokenset.token_hierarchy = []
    tokenset.tokens = OrderedDict()
    for tk_group in json_data["tk-groups-hierarchy"]:
        dict_group = OrderedDict()
        for token in tk_group["tokens"]:
            image_path = token["image"]
            dict_group[token["char"]] = image_path
        tokenset.token_hierarchy.append(dict_group)
        tokenset.tokens.update(dict_group)

    # Load the token sprites
    tokenset.token_sprites = {}
    for token in tokenset.tokens.keys():
        image_path = tokenset.get_path_token_sprite(token)
        sprite = Image.open(image_path)
        sprite.thumbnail(TokenSet.SPRITE_SIZE)
        if sprite.width != TokenSet.SPRITE_SIZE[1] or sprite.height != TokenSet.SPRITE_SIZE[0]:
            wrap_image = Image.new("RGB", TokenSet.SPRITE_SIZE, (0, 0, 0))
            wrap_image.paste(sprite, ((TokenSet.SPRITE_SIZE[1] - sprite.width) // 2, TokenSet.SPRITE_SIZE[0] - sprite.height))
            sprite = wrap_image
        tokenset.token_sprites[token] = sprite

    return tokenset
