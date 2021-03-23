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
    sprite_size = cfg.LEVEL.TILE_SIZE
    tokenset.token_sprites_preview = {}
    tokenset.token_sprites = {}
    # Load the empty tile: it will be used as background for all the others
    bg_image_path = tokenset.get_path_token_sprite("-")
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.convert("RGBA")
    bg_image = bg_image.resize(sprite_size, Image.LANCZOS)
    for token in tokenset.tokens.keys():
        image_path = tokenset.get_path_token_sprite(token)
        sprite = Image.open(image_path)
        sprite = sprite.convert("RGBA")
        sprite = sprite.resize(sprite_size, Image.LANCZOS)

        # Use the empty tile (-) as a background for all the others
        temp_image = bg_image.copy()
        temp_image.paste(sprite, (0, 0), sprite)
        temp_image = temp_image.convert("RGB")

        tokenset.token_sprites_preview[token] = sprite
        tokenset.token_sprites[token] = temp_image

    return tokenset
