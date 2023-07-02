from enum import Enum, unique


@unique
class RequestStatus(Enum):
    SUC = 0
    IMAGE_LOSS = 1
    IMAGE_FORMAT_ERROR = 2
    PROMPT_LOSS = 3
    IMAGE_WEB_SIZE_LOSS = 4
    IMAGE_WEB_SIZE_FORMAT_ERROR = 5
    POINT_LOSS = 6
    POINT_FORMAT_ERROR = 7
    POINT_SET_LOSS = 8
    POINT_LABEL_LOSS = 9
    BOX_LOSS = 10
    TEXT_LOSS = 11


@unique
class ResponseStatus(Enum):
    OK = 200
    REQUEST_ERROR = 406
