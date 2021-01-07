import os
from yacs.config import CfgNode as CN


_C = CN()
# Receptive field of the convolutional filters in the last convolutional block
_C.CONV_RECEPTIVE_FIELD = 11
_C.SCALE_FACTOR = 0.75

_C.PATH = CN()
_C.PATH.ROOT = os.path.dirname(os.path.abspath(__file__))
_C.PATH.RESOURCES = os.path.join(_C.PATH.ROOT, "resources")


cfg = _C
