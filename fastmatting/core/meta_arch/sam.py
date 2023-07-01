import torch
import numpy as np
import torch.nn.functional as func
from torch import nn

from typing import Dict, Tuple
from fastmatting.core.encoder.image_encoder import build_image_encoder
from fastmatting.core.encoder.prompt_encoder import build_prompt_encoder
from fastmatting.core.decoder import build_decoder
from transformers import AutoTokenizer, CLIPTextModel
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class SAM(nn.Module):
    def __init__(self, cfg):
        super(SAM, self).__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self.tokenizer = AutoTokenizer.from_pretrained("pretrains/CLIP")
        self.text_model = CLIPTextModel.from_pretrained("pretrains/CLIP")
        self.image_encoder = build_image_encoder(cfg)
        self.prompt_encoder = build_prompt_encoder(cfg)
        self.mask_decoder = build_decoder(cfg)
        is_pretrain = cfg.MODEL.PRETRAIN
        pretrain_path = cfg.MODEL.PRETRAIN_PATH
        is_strict = cfg.MODEL.IS_STRICT
        if is_pretrain:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict, strict=is_strict)

    def forward(self, inputs: Dict, multimask_output: bool = False):
        input_images = self.process_image(inputs)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for cur_embedding, image_record in zip(image_embeddings, inputs["prompt"]):
            points_pair = None
            if "point_set" in image_record and "point_label" in image_record:
                point_coords = image_record["point_set"]
                point_labels = image_record["point_label"]
                if point_coords is not None and point_labels is not None:
                    point_coords = point_coords[None, :, :]
                    point_labels = point_labels[None, :]
                    points_pair = (point_coords.to(self.device), point_labels.to(self.device))
            boxes = None
            if "box" in image_record:
                boxes = image_record["box"]
                if boxes is not None:
                    boxes = boxes[None, :]
                    boxes = boxes.to(self.device)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points_pair,
                boxes=boxes,
                masks=image_record.get("mask_input", None)
            )

            if "text" in inputs["prompt"]:
                text_inputs = self.tokenizer(image_record["text"], padding=True, return_tensors="pt")
                input_ids = text_inputs["input_ids"].to(self.device)
                attention_mask = text_inputs["attention_mask"].to(self.device)
                with torch.no_grad():
                    text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                    text_features = text_features.pooler_output[:, None, :]
                    text_features = func.interpolate(text_features, 256, mode="linear")
                    sparse_embeddings = torch.cat([sparse_embeddings, text_features], dim=1)

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=cur_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            outputs.append(low_res_masks)

        pred_matte = torch.cat(outputs, dim=0)
        outs = {"pred_matte": pred_matte}
        return outs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...]
    ):
        masks = func.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False
        )
        masks = masks[..., : input_size[0], :input_size[1]]
        return masks

    def process_image(self, inputs):
        """
        Normalize and batch the input images
        """
        if isinstance(inputs, dict):
            images = inputs["images"].to(self.device)
        elif isinstance(inputs, torch.Tensor):
            images = inputs.to(self.device)
        else:
            raise TypeError("batch_inputs must be dict or torch.Tensor, but get `{}`.".format(type(inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    @property
    def device(self):
        return self.pixel_mean.device

