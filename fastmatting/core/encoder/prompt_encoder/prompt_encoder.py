# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn

from typing import Any, Tuple, Type, Optional
from fastmatting.layers import LayerNorm2d
from fastmatting.core.encoder.prompt_encoder.build import PROMPT_ENCODER_REGISTRY


@PROMPT_ENCODER_REGISTRY.register()
class PromptEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        activation: Type[nn.Module] = nn.GELU
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          cfg:
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super(PromptEncoder, self).__init__()
        self._cfg = cfg
        self.embed_dim = cfg.MODEL.PROMPT_ENCODER.EMBED_DIM
        self.input_image_size = cfg.MODEL.PROMPT_ENCODER.INPUT_IMAGE_SIZE
        self.image_embedding_size = cfg.MODEL.PROMPT_ENCODER.IMAGE_EMBED_SIZE
        mask_in_channels = cfg.MODEL.PROMPT_ENCODER.MASK_IN_CHANNELS
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings_list = [nn.Embedding(1, self.embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings_list)
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)

        self.mask_input_size = (4 * self.image_embedding_size[0], 4*self.image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels // 4),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels),
            activation(),
            nn.Conv2d(mask_in_channels, self.embed_dim, kernel_size=1)
        )
        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0]), 1, device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embeds box prompts
            boxes:
        Returns:
        """
        boxes = boxes + 0.5  # shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embeds mask inputs.
            masks:
        Returns:
        """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    @staticmethod
    def _get_batch_size(
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor]
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
            points:
            boxes:
            masks:
        Returns:
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.
        Args:
            points (tuple[torch.Tensor, torch.Tensor] or None): point coordinates
            boxes (torch.Tensor or None): boxes to embed
            masks (torch.Tensor or None): masks to embed
        Returns:
            torch.Tensor: sparse embeddings for the points and boxes, with shape B x N x embed_dim, where N is
            determined by the number of input points and boxes.
            torch.Tensor: dense embeddings for the mask, in the shape B x embed_dim x embed_H x embed_W
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embedding = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(bs is None))
            sparse_embedding = torch.cat([sparse_embedding, point_embeddings], dim=1)
        if boxes is not None:
            box_embedding = self._embed_boxes(boxes)
            sparse_embedding = torch.cat([sparse_embedding, box_embedding], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embedding, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super(PositionEmbeddingRandom, self).__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
