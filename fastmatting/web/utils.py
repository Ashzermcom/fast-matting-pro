import json
import imageio
import numpy as np
from flask import request
from .status import RequestStatus


def preprocess_request():
    image = request.files.get("image")
    inputs = None
    if image is None:
        return RequestStatus.IMAGE_LOSS, inputs
    try:
        src_img = imageio.imread_v2(image)
    except Exception as e:
        return RequestStatus.IMAGE_FORMAT_ERROR, inputs
    prompt = request.form.get("prompt")
    if prompt is None:
        return RequestStatus.PROMPT_LOSS, inputs
    try:
        prompt = json.loads(prompt)
    except Exception as e:
        return RequestStatus.POINT_FORMAT_ERROR, inputs
    point = prompt.get("point")
    if point is None:
        return RequestStatus.POINT_LOSS, inputs

    point_set = []
    point_label = []
    positive = point.get("positive")
    negative = point.get("negative")
    if not isinstance(positive, list) or not isinstance(negative, list):
        return RequestStatus.POINT_SET_LOSS, inputs
    if len(positive):
        point_set.extend(positive)
        point_label.extend([1] * len(positive))
    if len(negative):
        point_set.extend(negative)
        point_label.extend([0] * len(negative))
    if len(point_set) and len(point_label):
        point_set = np.array(point_set)
        point_label = np.array(point_label)
    else:
        point_set = None
        point_label = None
    box = prompt.get("box")
    if not isinstance(box, list):
        return RequestStatus.BOX_LOSS, inputs
    if len(box):
        box = np.array(prompt["box"])
    else:
        box = None
    text = prompt.get("text")
    if not isinstance(text, str):
        return RequestStatus.TEXT_LOSS, inputs

    inputs = {
        "image": src_img,
        "prompt": {
            "text": text,
            "box": box,
            "point_set": point_set,
            "point_label": point_label
        }
    }
    return RequestStatus.SUC, inputs


def log_response_status(req_flag):
    if req_flag == RequestStatus.SUC:
        status_info = "200 SUCCESS"
    elif req_flag == RequestStatus.IMAGE_LOSS:
        status_info = "406 IMAGE FILE LOSS"
    elif req_flag == RequestStatus.IMAGE_FORMAT_ERROR:
        status_info = "406 IMAGE FILE FORMAT ERROR"
    elif req_flag == RequestStatus.PROMPT_LOSS:
        status_info = "406 PROMPT LOSS"
    elif req_flag == RequestStatus.POINT_LOSS:
        status_info = "406 POINT LOSS"
    elif req_flag == RequestStatus.POINT_FORMAT_ERROR:
        status_info = "406 POINT FORMAT ERROR"
    elif req_flag == RequestStatus.POINT_SET_LOSS:
        status_info = "406 POINT SET ERROR"
    elif req_flag == RequestStatus.POINT_LABEL_LOSS:
        status_info = "406 POINT LABEL LOSS"
    elif req_flag == RequestStatus.BOX_LOSS:
        status_info = "406 BOX LOSS"
    elif req_flag == RequestStatus.TEXT_LOSS:
        status_info = "406 TEXT LOSS"
    else:
        status_info = "406 UNKNOWN ERROR"
    return status_info
