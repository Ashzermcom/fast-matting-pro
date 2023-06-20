import torch
import argparse
from fastmatting.config import get_cfg
from fastmatting.core.meta_arch import build_model
from fastmatting.data.transforms import build_transforms
from flask import Flask, Response, request, render_template


app = Flask(__name__)


def argument_parser():
    """
    """
    parser = argparse.ArgumentParser(description="fast-matting webui")
    parser.add_argument("--config-file", default="configs/webui.yml", metavar="FILE", help="path to config file")
    return parser


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


class MattingInfer(object):
    def __init__(self, args):
        self.cfg = setup(args)
        self.device = torch.device(args.device)
        self.checkpoint_path = args.weight_path
        self.transform = self.build_transform()
        self.model = self.build_model()

    def build_transformer(self):
        img_transformers = build_transforms(self.cfg)
        return img_transformers

    def build_model(self):
        model = build_model(self.cfg)
        weights = torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(weights["model"], strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def matte(self, inputs):
        # {"image": <image_file>, "prompt": {"point": {"positive": [x1,y1, x2,y2, x3,y3...], "negative": [x5,y5....]}, "text": "文本", "box": [x1,y1, x2,y2]}}
        img = inputs["image"]
        img = self.transform(img)
        img = img[None, :, :, :]
        inputs = {
            "image": img,
        }
        with torch.no_grad():
            result = self.model(inputs)
            matte = result["pred_matte"]
        return matte


