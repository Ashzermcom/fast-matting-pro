import torch
import argparse
from fastmatting.config import get_cfg
from fastmatting.data.transforms import build_transforms
from flask import Flask, Response, request, render_template


app = Flask(__name__)


def argument_parser():
    """
    """
    parser = argparse.ArgumentParser(description="fast-matting webui")
    parser.add_argument("--config-file", default="configs/webui.yml", metavar="FILE", help="path to config file")
    parser.add_argument("--device", type=str, default="cuda:0")


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

    def matte(self, img):
        img = self.transform(img)
        img = img[None, :, :, :]
        inputs = {"image": img}
        with torch.no_grad():
            result = self.model(inputs)
            matte = result["pred_matte"]
        return matte


