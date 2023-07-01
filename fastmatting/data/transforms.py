import cv2
import torch
import numpy as np
from typing import Tuple
from copy import deepcopy
from torchvision import transforms


class Resize(object):
    def __init__(self, size, interpolation):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return cv2.resize(img, self.size, interpolation = self.interpolation)


class CenterAlign(object):
    def __init__(self, dst_size):
        self.dst_size = dst_size

    def __call__(self, img):
        ih, iw = img.shape[:2]
        dw, dh = self.dst_size
        scale = min(dw/iw, dh/ih)
        inv_affine_matrix = np.array([
            [scale, 0, -scale * iw * 0.5 + dw * 0.5],
            [0, scale, -scale * ih * 0.5 + dh * 0.5]
        ])
        dst = cv2.warpAffine(img, inv_affine_matrix, self.dst_size)
        return dst


class ToTensor(object):
    def __call__(self, img):
        # img = torch.from_numpy(img.transpose(2, 0, 1))
        img = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        return img


def build_transforms(cfg):
    img_transform = []
    size_image = cfg.INPUT.SIZE_IMAGE
    # img_transform.append(Resize(size_image, interpolation=cv2.INTER_CUBIC))
    img_transform.append(CenterAlign(size_image))
    img_transform.append(ToTensor())
    return transforms.Compose(img_transform)


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides methods for resizing coordinates and boxes.
    Provides methods for transforming both numpy array and batched torch tensors.
    """
    def __init__(self, target_length: int = 1024) -> None:
        self.target_length = target_length

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Excepts a numpy array shape Bx4. Requires the original image size in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww
