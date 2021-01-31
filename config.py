import os
from yacs.config import CfgNode as CN

import levels
from levels.tokens.mario import TOKEN_GROUPS

_C = CN()
# Receptive field of the convolutional filters in the last convolutional block
_C.CONV_RECEPTIVE_FIELD = 11
_C.SCALE_FACTOR = 0.8

_C.PATH = CN()
_C.PATH.ROOT = os.path.dirname(os.path.abspath(__file__))
_C.PATH.RESOURCES = os.path.join(_C.PATH.ROOT, "resources")
_C.PATH.LEVEL = os.path.join(_C.PATH.RESOURCES, "levels", "mario", "lvl_1-1.txt")
_C.PATH.SPRITES = os.path.join(_C.PATH.RESOURCES, "levels", "mario", "sprites")

_C.PATH.TRAIN = CN()
_C.PATH.TRAIN.DIR = os.path.join(_C.PATH.RESOURCES, "training")
_C.PATH.TRAIN.SCALED_IMGS = os.path.join(_C.PATH.TRAIN.DIR, "scaled-imgs")
_C.PATH.TRAIN.MONITOR_IMGS = os.path.join(_C.PATH.TRAIN.DIR, "training-imgs")

_C.LEVEL = CN()
_C.LEVEL.TYPE = "mario"


cfg = _C
