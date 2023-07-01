import cv2
import json
import torch
import imageio
import argparse
import numpy as np
from fastmatting.config import get_cfg
from fastmatting.core.meta_arch import build_model
from fastmatting.data.transforms import build_transforms, ResizeLongestSide
from flask import Flask, Response, request
from fastmatting.web import RequestStatus, log_response_status, preprocess_request
from gevent import pywsgi

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
        self.device = torch.device(self.cfg.MODEL.DEVICE)
        self.checkpoint_path = self.cfg.MODEL.PRETRAIN_PATH
        self.img_transform = self.build_transform()
        self.prompt_transform = ResizeLongestSide()
        self.model = self.build_model()

    def build_transform(self):
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
        """
        Args:
            inputs: dict with format
                {
                    "image": np.ndarray,
                    "prompt": {
                        "point_set": np.ndarray,
                        "point_label": np.ndarray,
                        "text": "string",
                        "box": np.ndarray
                    }
                }
        Returns:
        """
        img = inputs["image"]
        origin_size = img.shape[:2]
        img = self.img_transform(img)
        img = img[None, :, :, :]
        prompt_points = inputs["prompt"]["point_set"]
        point_label = inputs["prompt"]["point_label"]
        if prompt_points is not None and point_label is not None:
            prompt_points = self.prompt_transform.apply_coords(prompt_points, origin_size)
            prompt_points = torch.as_tensor(prompt_points, dtype=torch.float)
            point_label = torch.as_tensor(point_label, dtype=torch.float)
        box_prompt = inputs["prompt"]["box"]
        if box_prompt is not None:
            box_prompt = self.prompt_transform.apply_coords(box_prompt, origin_size)
            box_prompt = torch.as_tensor(box_prompt, dtype=torch.float)
        fm_inputs = {
            "images": img,
            "prompt": [{
                "text": inputs["prompt"]["text"],
                "box": box_prompt,
                "point_set": prompt_points,
                "point_label": point_label
            }]
        }
        with torch.no_grad():
            result = self.model(fm_inputs)
            matte = result["pred_matte"]
        return matte


fm_args = argument_parser().parse_args()
fm_infer = MattingInfer(fm_args)


@app.route("/", methods=["POST"])
def matting():
    req_flag, inputs = preprocess_request()
    if req_flag != RequestStatus.SUC:
        return Response(status=log_response_status(req_flag))
    pred_matte = fm_infer.matte(inputs)
    dst_img = pred_matte.detach().cpu().numpy().squeeze()
    dst_img = dst_img > 0
    dst_img = 255 * dst_img
    encoded_img = cv2.imencode(".png", dst_img)
    resp = encoded_img[1].tobytes()
    return Response(response=resp, status=log_response_status(req_flag), mimetype="image/png")


if __name__ == '__main__':
    # server = pywsgi.WSGIServer(('127.0.0.1', 2048), app)
    # server.serve_forever()
    app.run(debug=True, host="0.0.0.0")
