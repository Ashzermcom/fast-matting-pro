# encoding: utf-8
"""
@author:
"""
from fastmatting.utils.registry import Registry

MATTE_DECODE_REGISTRY = Registry("DECODER")
MATTE_DECODE_REGISTRY.__doc__ = """
Registry for decoder in matting model.
"""


def build_decoder(cfg):
    """
    Build matting decoder defined by `cfg.MODEL.MATTING_HEADS.NAME`.
    """
    head_name = cfg.MODEL.MATTING_DECODER.NAME
    return MATTE_DECODE_REGISTRY.get(head_name)(cfg)
