from typing import Any, Generic, NewType, TypedDict, TypeVar

import numpy as np

TModelKey = NewType("TModelKey", str)
TTagName = NewType("TTagName", str)
TAlarmType = NewType("TAlarmType", str)
TAlarmSettingDB = NewType("TAlarmSetting", dict[str, Any])
TTagSetting = NewType("TTagSetting", dict[str, Any])


class TModelTagData(TypedDict):
    learedValue: float
    statusCodeEnum: int
    index: float
    statusIndex: float


class TModelData(TypedDict):
    unixTime: int
    data: dict[TTagName, TModelTagData]


class TRequestData(TypedDict):
    unixTime: int
    data: dict[TTagName, dict]


Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    pass
