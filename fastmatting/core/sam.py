import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel


class SAM(nn.Module):
    def __init__(self, cfg):
        super(SAM, self).__init__()
        self._cfg = cfg
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self.tokenizer = AutoTokenizer.from_pretrained("pretrains/CLIP")
        self.text_model = CLIPTextModel.from_pretrained("pretrains/CLIP")


