import cv2
import torch


class Resize(object):
    def __init__(self, size, interpolation):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return cv2.resize(img, self.size, interpolation = self.interpolation)


class ToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img


def build_transforms(cfg):
    img_transform = []
    size_image = cfg.INPUT.SIZE_IMAGE
    img_transform.append(Resize(size_image, interpolation=cv2.INTER_CUBIC))
    img_transform.append(ToTensor())
    return img_transform

