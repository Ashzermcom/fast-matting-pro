import cv2
import base64
import numpy as np
from flask import request
from .status import RequestStatus


def preprocess_request():
    req_json = request.get_json()
    image = req_json.get("image")
    inputs = None
    if image is None:
        return RequestStatus.IMAGE_LOSS, inputs
    try:
        src_img_base64_str = image.split(",")[-1]
        src_img_base64_decode = base64.b64decode(src_img_base64_str)
        src_img = np.fromstring(src_img_base64_decode, np.uint8)
        src_img = cv2.imdecode(src_img, cv2.COLOR_BGR2RGB)
        src_h, src_w = src_img.shape[:2]
    except Exception as e:
        return RequestStatus.IMAGE_FORMAT_ERROR, inputs

    prompt = req_json.get("prompt")
    if prompt is None:
        return RequestStatus.PROMPT_LOSS, inputs
    if not isinstance(prompt, dict):
        return RequestStatus.POINT_FORMAT_ERROR, inputs

    web_size = req_json.get("size")
    if web_size is None:
        return RequestStatus.IMAGE_WEB_SIZE_LOSS, inputs
    web_h = web_size.get("height")
    web_w = web_size.get("width")
    if not isinstance(web_h, int) or not isinstance(web_w, int):
        return RequestStatus.IMAGE_WEB_SIZE_FORMAT_ERROR, inputs
    scale = np.array([src_w, src_h]) / np.array([web_w, web_h])

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
        point_set = np.array(point_set) * scale
        point_label = np.array(point_label)
    else:
        point_set = None
        point_label = None
    box = prompt.get("box")
    if not isinstance(box, list):
        return RequestStatus.BOX_LOSS, inputs
    if len(box):
        box = np.array(prompt["box"])
        box[0] = box[0] * src_w / web_w
        box[1] = box[1] * src_h / web_h
        box[2] = box[2] * src_w / web_w
        box[3] = box[3] * src_h / web_h
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
    # if True:
    #     show_img = inputs["image"]
    #     show_img = cv2.rectangle(show_img, inputs["prompt"]["box"][:2], inputs["prompt"]["box"][2:], (0,255,0), 2)
    #     col = inputs["prompt"]["point_set"].shape[0]
    #     ps = inputs["prompt"]["point_set"]
    #
    #     for _ in range(col):
    #         show_img = cv2.circle(show_img, (int(ps[_][0]), int(ps[_][1])), 2, (255,0,0), 2)
    #     cv2.imwrite("req.png", show_img)
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
