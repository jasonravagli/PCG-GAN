from collections import OrderedDict
import json
import os

from config import cfg
from levels.tokens.tokenset import TokenSet


def get_all():
    return [f for f in os.listdir(cfg.PATH.TOKENSETS) if os.path.isdir(os.path.join(cfg.PATH.TOKENSETS, f))]


def read(tokenset_name: str) -> TokenSet:
    with open(os.path.join(cfg.PATH.TOKENSETS, tokenset_name, tokenset_name + ".json")) as f:
        json_data = json.load(f)

    tokenset = TokenSet()
    tokenset.name = json_data["name"]
    tokenset.token_hierarchy = []
    tokenset.tokens = OrderedDict()
    for tk_group in json_data["tk-groups-hierarchy"]:
        dict_group = OrderedDict()
        for token in tk_group["tokens"]:
            dict_group[token["char"]] = token["image"]
        tokenset.token_hierarchy.append(dict_group)
        tokenset.tokens.update(dict_group)

    return tokenset