from enum import Enum, unique


@unique
class RequestStatus(Enum):
    SUC = 0
    IMAGE_LOSS = 1
    IMAGE_FORMAT_ERROR = 2
    PROMPT_LOSS = 3
    POINT_LOSS = 4
    POINT_FORMAT_ERROR = 5
    POINT_SET_LOSS = 6
    POINT_LABEL_LOSS = 7
    BOX_LOSS = 8
    TEXT_LOSS = 9


@unique
class ResponseStatus(Enum):
    OK = 200
    REQUEST_ERROR = 406
