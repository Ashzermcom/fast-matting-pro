from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.META_ARCHITECTURE = "SAM"
_C.MODEL.PRETRAIN = True
_C.MODEL.PRETRAIN_PATH = ""
_C.MODEL.IS_STRICT = False
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_IMAGE = [1024, 1024]
