from fastmatting.utils.registry import Registry


PROMPT_ENCODER_REGISTRY = Registry("PROMPT_ENCODER")
PROMPT_ENCODER_REGISTRY.__doc__ = """
Registry for prompt encoder, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.
It must returns an instance of :class:`PROMPT_ENCODER`.
"""


def build_prompt_encoder(cfg):
    """
    Build prompt encoder from `cfg.MODEL.PROMPT_ENCODER.NAME`.
    Returns:
        an instance of :class:`PROMPT_ENCODER`
    """
    prompt_encoder_name = cfg.MODEL.PROMPT_ENCODER.NAME
    prompt_encoder = PROMPT_ENCODER_REGISTRY.get(prompt_encoder_name)(cfg)
    return prompt_encoder
