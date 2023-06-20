# encoding: utf-8
from fastmatting.utils.registry import Registry


IMAGE_ENCODER_REGISTRY = Registry("IMAGE_ENCODER")
IMAGE_ENCODER_REGISTRY.__doc__ = """
Registry for image encoder, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.
It must returns an instance of :class:`IMAGE_ENCODER`.
"""


def build_image_encoder(cfg):
    """
    Build image encoder from `cfg.MODEL.IMAGE_ENCODER.NAME`.
    Returns:
        an instance of :class:`IMAGE_ENCODER`
    """
    image_encoder_name = cfg.MODEL.IMAGE_ENCODER.NAME
    image_encoder = IMAGE_ENCODER_REGISTRY.get(image_encoder_name)(cfg)
    return image_encoder
