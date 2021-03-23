import os
from yacs.config import CfgNode as CN

_C = CN()
# Receptive field of the convolutional filters in the last convolutional block
_C.CONV_RECEPTIVE_FIELD = 11
_C.SCALE_FACTOR = 0.8

_C.PATH = CN()
_C.PATH.ROOT = os.path.dirname(os.path.abspath(__file__))
_C.PATH.RESOURCES = os.path.join(_C.PATH.ROOT, "resources")
_C.PATH.LEVELS_DIR = os.path.join(_C.PATH.RESOURCES, "levels")
_C.PATH.LEVEL = os.path.join(_C.PATH.RESOURCES, "levels", "mario", "lvl_1-1.txt")
_C.PATH.SPRITES = os.path.join(_C.PATH.RESOURCES, "levels", "mario", "sprites")
_C.PATH.PROJECTS = os.path.join(_C.PATH.RESOURCES, "projects")
_C.PATH.TOKENSETS = os.path.join(_C.PATH.RESOURCES, "tokensets")
_C.PATH.OUTPUT = os.path.join(_C.PATH.ROOT, "output")

_C.PATH.TRAIN = CN()
_C.PATH.TRAIN.DIR = os.path.join(_C.PATH.OUTPUT, "training")
_C.PATH.TRAIN.SCALED_IMGS = os.path.join(_C.PATH.TRAIN.DIR, "scaled-imgs")
_C.PATH.TRAIN.MONITOR_IMGS = os.path.join(_C.PATH.TRAIN.DIR, "training-imgs")
_C.PATH.TRAIN.LOSSES = os.path.join(_C.PATH.TRAIN.DIR, "losses")
_C.PATH.TRAIN.PROJECT_NAME = "toadgan-project"

_C.LEVEL = CN()
_C.LEVEL.TYPE = "default"
_C.LEVEL.NAME = "ex-1"
_C.LEVEL.TILE_SIZE = (32, 32)

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 1000
_C.TRAIN.NOISE_UPDATE = 0.1

cfg = _C
